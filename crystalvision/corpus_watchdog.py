import os
import time
import json
import logging
import uuid
from pathlib import Path
from typing import Optional, Dict, Type, List

from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient, models as qmodels
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

try:
    from .lang import CORPUS_DIR, CORPUS_JSON, gather_corpus
    from .lang.docs import LOADER_MAP, gather_documents
    from .lang.embeddings import FastEmbedEmbeddingsGPU
except (ModuleNotFoundError, ImportError):
    from crystalvision.lang import CORPUS_DIR, CORPUS_JSON, gather_corpus
    from crystalvision.lang.docs import LOADER_MAP, gather_documents
    from crystalvision.lang.embeddings import FastEmbedEmbeddingsGPU


log = logging.getLogger("corpus_watchdog")


class CorpusWatchdog:
    CORPUS_DIRECTORY = Path(os.getenv("CORPUS_DIR", CORPUS_DIR)).resolve()
    CORPUS_FILE = Path(os.getenv("CORPUS_URIS_FILE", CORPUS_JSON)).resolve()

    def __init__(
        self,
        embeddings: Optional[FastEmbedEmbeddingsGPU] = None,
    ):
        assert embeddings, "No embeddings provided"

        if not self.CORPUS_FILE.exists():
            raise FileNotFoundError(f"Could not find {self.CORPUS_FILE}")

        if not self.CORPUS_DIRECTORY.exists():
            self.CORPUS_DIRECTORY.mkdir()

        self.observer = PollingObserver()
        self.event_handler = CorpusHandler(
            self.CORPUS_FILE, self.CORPUS_DIRECTORY, embeddings
        )

    async def run(self):
        log.debug(self.event_handler)
        self.observer.schedule(
            self.event_handler, str(self.CORPUS_DIRECTORY) + os.sep, recursive=True
        )
        self.observer.schedule(self.event_handler, str(self.CORPUS_FILE))
        self.observer.start()
        log.info("CorpusWatchdog over %s/, %s", self.CORPUS_DIRECTORY, self.CORPUS_FILE)
        await self.event_handler.gather_corpus()
        log.info("Watching")
        try:
            while True:
                time.sleep(5)
        except Exception:
            self.observer.stop()
            log.warning("CorpusWatchdog Observer Stopped")

        self.observer.join()


class CorpusHandler(FileSystemEventHandler):
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "crystalvision-discordbot")

    def __init__(
        self,
        corpus_uris_file: Path,
        corpus_dir: Path,
        embeddings: FastEmbedEmbeddingsGPU,
        distance: Type[qmodels.Distance] = qmodels.Distance.COSINE,
    ):
        super().__init__()

        self._corpus_uris_file: Path = corpus_uris_file
        self.corpus_dir: Path = corpus_dir

        self.embeddings = embeddings
        self.vector_store_client: QdrantClient = QdrantClient(
            url=os.getenv("QDRANT_HOST"), prefer_grpc=True
        )

        if not self.vector_store_client.collection_exists(self.COLLECTION_NAME):
            log.info("Creating vectorstore: %s", self.COLLECTION_NAME)
            log.info(self.embeddings)
            embedding_vector = self.embeddings.embed_query("Getting the dimesionality!")
            log.debug(
                "'Getting the dimesionality!' -> VectorSize of %s",
                len(embedding_vector),
            )
            self.vector_store_client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(
                    size=len(embedding_vector), distance=distance
                ),
            )
        self.vector_store: QdrantVectorStore = QdrantVectorStore(
            self.vector_store_client,
            collection_name=self.COLLECTION_NAME,
            embedding=self.embeddings,
            distance=distance,
        )

        self.data: Dict = self.reload_data()

    def __repr__(self):
        return f"<CorpusHandler ({self._corpus_uris_file} -> {self.COLLECTION_NAME})>"

    async def gather_corpus(self):
        remainders = await gather_corpus()
        log.debug("remainder: %s", remainders)

        missing_docs = []
        missing_uuids = []
        for d in remainders:
            missings = await self.astore_document_by_fname(d["fname"])
            missing_docs += missings[0]
            missing_uuids += missings[1]

        if missing_docs:
            await self.vector_store.aadd_documents(
                documents=missing_docs, ids=missing_uuids
            )
            log.info(
                "Added %s missing to the vectordb '%s'",
                len(missing_docs),
                self.COLLECTION_NAME,
            )
        log.info("Finished downloading corpus")

    def reload_data(self) -> Dict:
        with self._corpus_uris_file.open("r") as f:
            try:
                corpus_data = json.load(f)
            except json.JSONDecodeError as err:
                log.error(err)
                return

        self.data = {}
        for corpus in corpus_data:
            key = corpus.get("fname")
            if key is None:
                key = corpus["uri"].split("/")[-1]

            if corpus.get("disabled", False):
                if (self.corpus_dir / key).exists():
                    self.remove_documents_by_fname(key)
                log.debug("Skipping %s", corpus["uri"])
                continue

            if (loader := LOADER_MAP.get(corpus["loader"], None)) is None:
                log.error(
                    "No loader found (%s) for %s", corpus["loader"], corpus["uri"]
                )
                continue
            corpus["loader"] = loader

            self.data[key] = corpus

        return self.data

    def on_created(self, event) -> None:
        if event.is_directory:
            return

        missing_docs, missing_uuids = self.store_document_by_fname(
            (fname := event.src_path.split(os.sep)[-1])
        )

        if missing_docs:
            self.vector_store.add_documents(documents=missing_docs, ids=missing_uuids)
            log.info("Added '%s' to the vectordb '%s'", fname, self.COLLECTION_NAME)

    def remove_documents_by_fname(self, fname: str) -> List[str]:
        filter_conditions = Filter(
            must=[
                FieldCondition(key="metadata.fname", match=MatchValue(value=fname)),
            ]
        )

        results = self.vector_store_client.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=filter_conditions,
            limit=100,
        )

        old_ids = []
        if old_documents := results[0]:
            old_ids = [doc.id for doc in old_documents]
            self.vector_store.delete(old_ids)

        return old_ids

    @staticmethod
    def get_uuid_from_document(doc: Document, fname: str) -> str:
        if (q_uuid := doc.metadata.get("id", None)) is None:
            q_uuid = fname
            if (page_num := doc.metadata.get("page", None)) is not None:
                q_uuid += f"/{page_num}"
            if (title := doc.metadata.get("title", None)) is not None:
                q_uuid += f"/{title}"
        q_uuid = uuid.uuid5(uuid.NAMESPACE_URL, name=q_uuid)
        return str(q_uuid)

    def remove_old_versions_by_uuid(
        self, doc: Document, quuid: str, fname: str
    ) -> bool:
        result = self.vector_store.get_by_ids([quuid])
        if result:
            log.debug(
                "%s vs %s", result[0].metadata["version"], doc.metadata["version"]
            )
            if result[0].metadata["version"] == doc.metadata["version"]:
                log.debug("%s (%s) is already in the vectorstore", quuid, doc.metadata)
                return False
            filter_conditions = Filter(
                must=[
                    FieldCondition(
                        key="metadata.version",
                        match=MatchValue(value=result[0].metadata["version"]),
                    ),
                    FieldCondition(key="metadata.fname", match=MatchValue(value=fname)),
                ]
            )

            results = self.vector_store_client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=filter_conditions,
                limit=250,
            )
            if old_documents := results[0]:
                old_ids = [doc.id for doc in old_documents]
                self.vector_store.delete(old_ids)
        return True

    def store_document_by_fname(self, fname: str):
        if fname not in self.data:
            log.warning("Cannot find %s registered with corpus data", fname)
            return

        data = self.data[fname]
        if data.get("disabled", False):
            log.debug("Skipping %s", fname)
            return

        document = data["loader"](data["uri"], **data.get("kwargs", {}))

        missing_docs = []
        missing_uuids = []
        for doc in document.lazy_load():
            doc.metadata["version"] = data.get("version")
            doc.metadata["fname"] = fname

            q_uuid = self.get_uuid_from_document(doc, fname)

            if not self.remove_old_versions_by_uuid(doc, q_uuid, fname):
                continue

            log.info("Adding %s (%s) to the vector store", q_uuid, doc.metadata)
            missing_docs.append(doc)
            missing_uuids.append(q_uuid)

        return missing_docs, missing_uuids

    async def astore_document_by_fname(self, fname: str):
        if fname not in self.data:
            log.warning("Cannot find %s registered with corpus data", fname)
            return

        data = self.data[fname]
        if data.get("disabled", False):
            log.debug("Skipping %s", fname)
            return

        document = data["loader"](data["uri"], **data.get("kwargs", {}))

        missing_docs = []
        missing_uuids = []
        async for doc in document.alazy_load():
            doc.metadata["version"] = data.get("version")
            doc.metadata["fname"] = fname

            q_uuid = self.get_uuid_from_document(doc, fname)

            if not self.remove_old_versions_by_uuid(doc, q_uuid, fname):
                continue

            log.debug("Adding %s (%s) to the vector store", q_uuid, doc.metadata)
            missing_docs.append(doc)
            missing_uuids.append(q_uuid)

        return missing_docs, missing_uuids

    def on_deleted(self, event):
        if event.is_directory:
            return

        fname = event.src_path.split(os.sep)[-1]
        if fname not in self.data:
            log.warning("Cannot find %s registered with corpus data", fname)

        log.warning("Removing vectordb entries for: %s", fname)
        old_ids = self.remove_documents_by_fname(fname)
        log.debug("Removed `%s` old ids: %s", fname, old_ids)

    def on_any_event(self, event):
        if event.is_directory:
            return

        if event.src_path.endswith("corpus_uris.json"):
            self.reload_data()
            log.debug("Re-gathering corpus data")
            remainder = asyncio.run(gather_corpus(self.data.values()))
            log.debug("remainder: %s", remainder)
            gather_documents(remainder)
            log.debug("Finished gathering corups data")
            return


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    assert (embed_model := os.getenv("FASTEMBED_TEXT_MODEL")), "No embed model provided"
    log.info("Embedding Model: %s", embed_model)

    watch = CorpusWatchdog(embeddings=FastEmbedEmbeddingsGPU(model_name=embed_model))
    asyncio.run(watch.run())

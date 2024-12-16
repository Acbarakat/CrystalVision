import json
import logging
from typing import List, Optional

from langchain_community.document_loaders import (
    PyPDFLoader,
    JSONLoader,
    AsyncChromiumLoader,
)

try:
    from . import CORPUS_DIR, CORPUS_JSON
    from .loaders import ChromiumJsonLoader, CardJsonLoader
except (ModuleNotFoundError, ImportError):
    from crystalvision.lang import CORPUS_DIR, CORPUS_JSON
    from crystalvision.lang.loaders import ChromiumJsonLoader, CardJsonLoader


log = logging.getLogger("lang.loaders")
log.setLevel(logging.INFO)

LOADER_MAP = {
    "PyPDF": PyPDFLoader,
    "ChromiumJson": ChromiumJsonLoader,
    "JSON": JSONLoader,
    "Card": CardJsonLoader,
    "Chromium": AsyncChromiumLoader,
}


def gather_documents(corpus_uris: Optional[List] = None) -> List:
    if corpus_uris is None:
        with open(CORPUS_JSON, "r") as fp:
            corpus_uris = json.load(fp)

    docs = []
    for corpus in corpus_uris:
        if (loader := LOADER_MAP.get(corpus["loader"], None)) is None:
            log.error("No loader found (%s) for %s", corpus["loader"], corpus["uri"])
            continue

        if corpus.get("disabled", False):
            continue

        uri = corpus["uri"]
        if isinstance(uri, str) and uri.startswith("http") and uri.endswith(".pdf"):
            uri = CORPUS_DIR / corpus.get("fname", uri.split("/")[-1])

        kwargs = corpus.get("kwargs", {})
        log.info("Attempting to %s: %s", loader, uri)
        docs.append(loader(uri, **kwargs))

    return docs

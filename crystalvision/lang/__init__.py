import logging
import json
import os
from pathlib import Path
import aiohttp
import asyncio


log = logging.getLogger("crystalvision.lang")
log.setLevel(logging.INFO)


SRC_DIR = Path(os.path.dirname(__file__), "..").resolve()
CORPUS_DIR = Path(SRC_DIR, "..", "data", "corpus").resolve()
CORPUS_JSON = (CORPUS_DIR / ".." / "corpus_uris.json").resolve()
PROMPTS_JSON = (CORPUS_DIR / ".." / "prompts.json").resolve()
CORPUS_URIS = {}

if CORPUS_JSON.exists():
    with open(CORPUS_JSON, "r") as fp:
        CORPUS_URIS = json.load(fp)
else:
    log.error("Could not find %s", CORPUS_JSON)


async def download_file(session, uri, dst):
    if dst.exists():
        log.info("Found file: %s", dst.name)
    else:
        log.info("Downloading file: %s", dst.name)
        async with session.get(uri) as response:
            if response.status == 200:
                # Open a file in binary write mode
                with open(dst, "wb+") as fp:
                    # Write the content of the response to the file
                    fp.write(await response.read())
                log.debug("%s downloaded successfully", dst.name)
            else:
                log.error(
                    "Failed to download %s. Status code: %s", dst.name, response.status
                )


async def gather_corpus():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for corpus in CORPUS_URIS:
            uri = corpus.get("uri")
            if isinstance(uri, str) and uri.startswith("http") and uri.endswith("pdf"):
                fname = uri.split("/")[-1]
                dst = (CORPUS_DIR / fname).resolve()
                dst.parent.mkdir(parents=True, exist_ok=True)
                tasks.append(download_file(session, uri, dst))
        await asyncio.gather(*tasks)


asyncio.run(gather_corpus())
log.info("Finished downloading corpus")

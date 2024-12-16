import logging
import json
import os
import xattr
from pathlib import Path
import asyncio
from typing import Optional, List

import aiohttp


log = logging.getLogger("crystalvision.lang")
log.setLevel(logging.INFO)


SRC_DIR = Path(os.path.dirname(__file__), "..").resolve()
CORPUS_DIR = Path(SRC_DIR, "..", "data", "corpus").resolve()
CORPUS_JSON = (CORPUS_DIR / ".." / "corpus_uris.json").resolve()
PROMPTS_JSON = (CORPUS_DIR / ".." / "prompts.json").resolve()
EMOJI_JSON = (CORPUS_DIR / ".." / "emoji.json").resolve()


if not CORPUS_JSON.exists():
    log.error("Could not find %s", CORPUS_JSON)


async def download_file(session, uri, dst, version: Optional[str] = None):
    if dst.exists():
        try:
            dver = xattr.getxattr(dst, b"user.version", symlink=True).decode()
        except OSError:
            dver = None
        log.info("Found file: %s (%s)", dst.name, dver)

        if dver == version:
            return

    log.info("Downloading file: %s (%s)", dst.name, version)
    async with session.get(uri) as response:
        if response.status == 200:
            # Open a file in binary write mode
            with open(dst, "wb+") as fp:
                # Write the content of the response to the file
                fp.write(await response.read())
            log.debug("%s downloaded successfully", dst.name)
            if version is not None:
                xattr.setxattr(dst, b"user.version", version.encode(), symlink=True)
                log.debug("%s xattrs set 'user.version' to %s", dst.name, version)
        else:
            log.error(
                "Failed to download %s. Status code: %s", dst.name, response.status
            )


async def gather_corpus(corpus_uris: Optional[List] = None):
    if corpus_uris is None:
        with open(CORPUS_JSON, "r") as fp:
            corpus_uris = json.load(fp)

    remainder_corpus = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for corpus in corpus_uris:
            if corpus.get("disabled", False):
                continue

            uri = corpus.get("uri")
            if isinstance(uri, str) and uri.startswith("http") and uri.endswith("pdf"):
                fname = corpus.get("fname", uri.split("/")[-1])
                dst = (CORPUS_DIR / fname).resolve()
                dst.parent.mkdir(parents=True, exist_ok=True)
                tasks.append(
                    download_file(session, uri, dst, version=corpus.get("version"))
                )
            else:
                remainder_corpus.append(corpus)
        await asyncio.gather(*tasks)

    return remainder_corpus

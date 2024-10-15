import logging

from langchain_community.document_loaders import (
    PyPDFLoader,
    JSONLoader,
    AsyncChromiumLoader,
)

try:
    from . import CORPUS_URIS, CORPUS_DIR
    from .loaders import ChromiumJsonLoader, CardJsonLoader
except (ModuleNotFoundError, ImportError):
    from crystalvision.lang import CORPUS_URIS, CORPUS_DIR
    from crystalvision.lang.loaders import ChromiumJsonLoader, CardJsonLoader


log = logging.getLogger("lang.loaders")
log.setLevel(logging.INFO)

DOCS = []
LOADER_MAP = {
    "PyPDF": PyPDFLoader,
    "ChromiumJson": ChromiumJsonLoader,
    "JSON": JSONLoader,
    "Card": CardJsonLoader,
    "Chromium": AsyncChromiumLoader,
}


for corpus in CORPUS_URIS:
    if (loader := LOADER_MAP.get(corpus["loader"], None)) is None:
        log.error("No loader found (%s) for %s", corpus["loader"], corpus["uri"])
        continue

    if corpus.get("disabled", False):
        continue

    uri = corpus["uri"]
    if isinstance(uri, str) and uri.startswith("http") and uri.endswith(".pdf"):
        uri = CORPUS_DIR / uri.split("/")[-1]

    kwargs = corpus.get("kwargs", {})
    DOCS.append(loader(uri, **kwargs))

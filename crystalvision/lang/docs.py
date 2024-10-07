import logging
import json
import asyncio
from typing import List, Iterator, AsyncIterator

import jq
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, AsyncChromiumLoader

try:
    from . import CORPUS_URIS, CORPUS_DIR
except (ModuleNotFoundError, ImportError):
    from crystalvision.lang import CORPUS_URIS, CORPUS_DIR


log = logging.getLogger("lang.loaders")
log.setLevel(logging.INFO)

DOCS = []


class ChromiumJsonLoader(AsyncChromiumLoader):
    def __init__(
        self,
        urls: List[str],
        *,
        headless: bool = True,
        user_agent: str | None = None,
        extra_http_headers: dict | None = None,
        cache_name: str | None = None,
        page_content_string: str = "{body}",
        jq_query: str = ".",
    ):
        super().__init__(urls, headless=headless, user_agent=user_agent)

        if isinstance(urls, str):
            self.urls = [urls]

        self.extra_http_headers: dict = (
            extra_http_headers if extra_http_headers is not None else {}
        )
        self.cache_name: str | None = cache_name
        self.handle_script: str = "body => body.querySelector('pre').innerHTML"
        self.page_content_string: str = page_content_string
        self.jq_query: jq._Program = jq.compile(jq_query)

    async def ascrape_playwright(self, url: str) -> str:
        """
        Asynchronously scrape the content of a given URL using Playwright's async API.

        Args:
            url (str): The URL to scrape.

        Returns:
            str: The scraped HTML content or an error message if an exception occurs.

        """
        from playwright.async_api import async_playwright

        log.info("Starting scraping...")
        results = ""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            try:
                page = await browser.new_page(
                    user_agent=self.user_agent,
                    extra_http_headers=self.extra_http_headers,
                )
                await page.goto(url)
                results = await page.content()  # Simply get the HTML content
                log.info("Content scraped")
                a_handle = await page.evaluate_handle("document.body")
                result_handle = await page.evaluate_handle(self.handle_script, a_handle)
                results = await result_handle.json_value()
                await result_handle.dispose()
            except Exception as err:
                log.exception(err)
                return []
            finally:
                await browser.close()

        results = json.loads(results)

        if self.cache_name:
            with open(CORPUS_DIR / self.cache_name, "w+") as fp:
                json.dump(results, fp, indent=4)

        return self.jq_query.input_value(results).first()

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazily load text content from the provided URLs.

        This method yields Documents one at a time as they're scraped,
        instead of waiting to scrape all URLs before returning.

        Yields:
            Document: The scraped content encapsulated within a Document object.

        """
        for url in self.urls:
            html_content = asyncio.run(self.ascrape_playwright(url))
            for content in html_content:
                content = content["document"]
                metadata = {
                    "title": content["title"],
                    "source": content["source"],
                    "id": content["id"],
                }
                yield Document(
                    page_content=self.page_content_string.format(**content),
                    metadata=metadata,
                )

    async def alazy_load(self) -> AsyncIterator[Document]:
        """
        Asynchronously load text content from the provided URLs.

        This method leverages asyncio to initiate the scraping of all provided URLs
        simultaneously. It improves performance by utilizing concurrent asynchronous
        requests. Each Document is yielded as soon as its content is available,
        encapsulating the scraped content.

        Yields:
            Document: A Document object containing the scraped content, along with its
            source URL as metadata.
        """
        tasks = [self.ascrape_playwright(url) for url in self.urls]
        results = await asyncio.gather(*tasks)
        for url, contents in zip(self.urls, results):
            log.debug("Retrieved content from %s", url)
            for content in contents:
                content = content["document"]
                metadata = {
                    "title": content["title"],
                    "source": content["source"],
                    "id": content["id"],
                }
                yield Document(
                    page_content=self.page_content_string.format(**content),
                    metadata=metadata,
                )


LOADER_MAP = {
    "PyPDF": PyPDFLoader,
    "ChromiumJson": ChromiumJsonLoader,
}


for corpus in CORPUS_URIS:
    if (loader := LOADER_MAP.get(corpus["loader"], None)) is None:
        log.error("No loader found (%s) for %s", corpus["loader"], corpus["uri"])
        continue

    uri = corpus["uri"]
    if uri.startswith("http") and uri.endswith(".pdf"):
        uri = CORPUS_DIR / uri.split("/")[-1]

    kwargs = corpus.get("kwargs", {})
    DOCS.append(loader(uri, **kwargs))

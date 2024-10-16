import re
import json
import asyncio
from pathlib import Path
from typing import AsyncIterator, Iterator, List, Optional

import jq
from jinja2 import Environment, FileSystemLoader, Template, select_autoescape
from langchain_community.document_loaders import AsyncChromiumLoader, JSONLoader
from langchain_core.documents import Document

try:
    from ..data.card_database import make_database
    from . import CORPUS_DIR, log
except (ModuleNotFoundError, ImportError):
    from crystalvision.data.card_database import make_database
    from crystalvision.lang import CORPUS_DIR, log


JENV = Environment(
    loader=FileSystemLoader(CORPUS_DIR / ".." / "jtemplate"),
    autoescape=select_autoescape(),
)


kanji_to_english = {
    "火": "Fire",
    "水": "Water",
    "土": "Earth",
    "風": "Wind",
    "雷": "Lightning",
    "氷": "Ice",
    "闇": "Dark",
    "光": "Light",
    "ダル": "Dull",
    "《": "[",
    "》": "]",
}


TEXTEN_REGEX = re.compile(
    "|".join(re.escape(kanji) for kanji in kanji_to_english.keys())
)
CARD_CODE = re.compile(r"(?:\d{1,2})-\d{3}[CRHLS]|PR-\d{3}")


def explain_database():
    with Path(__file__).parent / "df_description.json" as fp:
        description = json.loads(fp.read_bytes())

    df = make_database().drop(
        ["id", "thumbs", "element", "power", "multicard", "mono"], axis=1
    )
    df = df[~df["code"].str.contains("C-")]
    for lang in ("de", "fr", "es", "it", "ja"):
        df.drop(
            [f"name_{lang}", f"text_{lang}", f"job_{lang}", f"type_{lang}"],
            axis=1,
            inplace=True,
        )

    df.rename({"element_v2": "element_ja", "power_v2": "power"}, inplace=True, axis=1)
    df["element"] = df["element_ja"].apply(
        lambda row: {kanji_to_english.get(kanji, kanji) for kanji in row}
    )
    df["numElements"] = df["element"].apply(lambda row: len(row))
    df["text_en"] = (
        df["text_en"].str.replace(r"\[\[br\]\]", "\u2029", regex=True).str.strip()
    )
    df["images"] = df["images"].apply(
        lambda x: (
            f"https://fftcg.cdn.sewest.net/images/cards/full/{x[0]}" if x else None
        )
    )
    df["cost"] = df["cost"].astype(int)
    df["power"] = df["power"].astype(float)

    for col in df.columns:
        if col_attrs := description.get(col):
            df[col].attrs.update(col_attrs)

    return df


class CardJsonLoader(JSONLoader):
    def __init__(
        self,
        file_path: str | Path,
        jq_schema: str = ".cards[]",
        content_key: Optional[str] = None,
        is_content_key_jq_parsable: Optional[bool] = False,
        json_lines: bool = False,
    ):
        super().__init__(
            file_path,
            jq_schema,
            content_key,
            is_content_key_jq_parsable,
            None,
            True,
            json_lines,
        )

        self.template = JENV.get_template("cards.jinja")

    @classmethod
    def make_metadata(cls, record: dict, source: str | Path) -> dict:
        metadata = {
            "source": str(source),
            "uuid": record["code"],
            "code": record["code"],
            "card_name": record["name_en"],
        }

        return metadata

    def _parse(self, content: str, index: int) -> Iterator[Document]:
        """Convert given content to documents."""
        data = self._jq_schema.input(json.loads(content))

        # Perform some validation
        # This is not a perfect validation, but it should catch most cases
        # and prevent the user from getting a cryptic error later on.
        if self._content_key is not None:
            self._validate_content_key(data)
        self._validate_metadata_func(data)

        for i, sample in enumerate(data, index + 1):
            if sample["element"] is None or not CARD_CODE.match(sample["code"]):
                log.warning("Skipping %s" % sample["code"])
                continue
            text = self._get_text(sample=sample)
            metadata = self.make_metadata(sample, self.file_path)
            yield Document(page_content=text, metadata=metadata)

    def _get_text(self, sample):
        """Convert sample to string format"""
        if self._content_key is not None:
            if self._is_content_key_jq_parsable:
                compiled_content_key = self.jq.compile(self._content_key)
                content = compiled_content_key.input(sample).first()
            else:
                content = sample[self._content_key]
        else:
            content = sample

        content["element"] = [
            kanji_to_english.get(kanji, kanji) for kanji in content["element"]
        ]
        content["power"] = int(content["power"])
        content["text_en"] = TEXTEN_REGEX.sub(
            lambda match: kanji_to_english[match.group(0)], content["text_en"]
        )
        content["multicard"] = content["multicard"] in ("\u25cb", "1")

        return self.template.render(**content)


class ChromiumJsonLoader(AsyncChromiumLoader):
    def __init__(
        self,
        urls: List[str],
        *,
        headless: bool = True,
        user_agent: str | None = None,
        extra_http_headers: dict | None = None,
        cache_name: str | None = None,
        page_content_string: str = "",
        page_content_template: str = "",
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
        self.page_content_template: Template | None = page_content_template
        if page_content_template:
            self.page_content_template = JENV.get_template(page_content_template)
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
                code = CARD_CODE.search(content["title"])
                if code:
                    code = code.group(0)
                    metadata["code"] = code

                page_content = content
                if self.page_content_string:
                    page_content = self.page_content_string.format(code=code, **content)
                elif self.page_content_template:
                    page_content = self.page_content_template.render(
                        code=code, **content
                    )

                yield Document(
                    page_content=page_content,
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
                code = CARD_CODE.search(content["title"])
                if code:
                    code = code.group(0)
                    metadata["code"] = code

                page_content = content
                if self.page_content_string:
                    page_content = self.page_content_string.format(code=code, **content)
                elif self.page_content_template:
                    page_content = self.page_content_template.render(
                        code=code, **content
                    )

                yield Document(
                    page_content=page_content,
                    metadata=metadata,
                )

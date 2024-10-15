"""
Serve a FFTCG discord bot.

TODO:
- Specialize the rules comp intos its own thing
- Use message history as context, with a max limit deque,
only pulling from the last checkpoint, maybe on disk db? or in mem only
- Decode content <@###> into user names

"""

import os
import json
import logging
import asyncio
import hashlib
import re
from typing import Optional
from functools import cached_property, wraps

import discord
import pandas as pd
from discord import Intents, ChannelType, channel
from ollama import Client, AsyncClient
from langchain.agents.agent import AgentExecutor
from langchain_core.language_models import BaseLLM
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.tools.retriever import create_retriever_tool

from crystalvision.lang.loaders import explain_database

try:
    from .lang import PROMPTS_JSON, CORPUS_DIR
    from .lang.docs import DOCS
except (ModuleNotFoundError, ImportError):
    from crystalvision.lang import PROMPTS_JSON, CORPUS_DIR
    from crystalvision.lang.docs import DOCS


log = logging.getLogger("discord.crystalvision")
log.setLevel(logging.DEBUG)

intents = Intents.default()
intents.members = True
intents.message_content = True
intents.reactions = True
intents.guilds = True

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)

EMOJI_JSON = (CORPUS_DIR / ".." / "emoji.json").resolve()


def thinking(timeout: int = 999):
    def decorator(func):
        async def thinking_reaction(
            message: discord.Message, user: discord.User
        ) -> None:
            """Add a thinking reaction until timeout to a message"""
            try:
                await message.add_reaction("🤔")
                async with message.channel.typing():
                    await asyncio.sleep(timeout)
            except Exception:
                pass
            finally:
                await message.remove_reaction("🤔", user)

        @wraps(func)
        async def wrapper(
            cls: discord.Client, message: discord.Member, *args, **kwargs
        ):
            task = asyncio.create_task(thinking_reaction(message, cls.user))
            try:
                return await func(cls, message, *args, **kwargs)
            except Exception as err:
                log.exception(err)
            finally:
                # Ensure the reaction is removed even in case of an error or timeout
                task.cancel()

        return wrapper

    return decorator


class CrystalClient(discord.Client):
    """A discord bot for FFTCG"""

    def __init__(
        self,
        *args,
        ollama: Optional[AsyncClient] = None,
        embeddings: Optional[BaseLLM] = None,
        llm: Optional[BaseLLM] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        assert ollama, "No ollama AsyncClient provided"
        assert embeddings, "No embeddings provided"
        assert llm, "No llm provided"

        self.ollama: Optional[AsyncClient] = ollama
        self.embeddings: Optional[BaseLLM] = embeddings
        self.llm: Optional[BaseLLM] = llm
        self.vector_store: VectorStore = Chroma(
            collection_name="crystalvision-discordbot",
            embedding_function=self.embeddings,
            persist_directory=str(CORPUS_DIR / ".." / "chroma_langchain_db"),
        )
        self.model: str = os.getenv("OLLAMA_CHAT_MODEL")
        self.prompts: dict = {}
        self.emoji_mapping: dict = {}
        self.df: pd.DataFrame = explain_database()

        self._ready: bool = False

    @cached_property
    def agent(self) -> AgentExecutor:
        kwargs = self.prompts.get("discord", {})

        retriever_tool = create_retriever_tool(
            self.retriever,
            "rules_search",
            kwargs.get("rules_search", "Search for information"),
        )

        prefix = kwargs.get("df_prefix1", "You are a pandas agent.")
        for col in self.df.columns:
            if col_desc := self.df[col].attrs.get("description", ""):
                prefix += f"'{col}' refers to {col_desc}. "

        prefix += kwargs.get("df_prefix2", "")

        return create_pandas_dataframe_agent(
            self.llm,
            self.df,
            verbose=True,
            include_df_in_prompt=None,
            allow_dangerous_code=True,
            prefix=prefix,
            extra_tools=[retriever_tool],
        )

    @cached_property
    def retriever(self) -> VectorStoreRetriever:
        return self.vector_store.as_retriever()

    async def on_ready(self) -> None:
        """Trigger when bot is ready/online"""
        if PROMPTS_JSON.exists():
            with open(PROMPTS_JSON, "r") as fp:
                self.prompts = json.load(fp)
        else:
            log.error("Could not find prompts json (%s)", PROMPTS_JSON)

        if EMOJI_JSON.exists():
            with open(EMOJI_JSON, "r") as fp:
                self.emoji_mapping = json.load(fp)
        else:
            log.error("Could not find emoji json (%s)", EMOJI_JSON)

        missing_docs = []
        missing_uuids = []
        for document in DOCS:
            async for doc in document.alazy_load():
                if (uuid := doc.metadata.get("id", None)) is None:
                    uuid = hashlib.blake2b(
                        doc.metadata["source"].encode(), digest_size=10
                    ).hexdigest()
                    if (page_num := doc.metadata.get("page", None)) is not None:
                        uuid += f"-{page_num}"
                    if (title := doc.metadata.get("title", None)) is not None:
                        title = hashlib.blake2b(
                            title.encode(), digest_size=6
                        ).hexdigest()
                        uuid += f"-{title}"

                result = self.vector_store.get(ids=[uuid])
                if uuid in result["ids"]:
                    log.debug(
                        "%s (%s) is already in the vectorstore", uuid, doc.metadata
                    )
                else:
                    log.info("Adding %s (%s) to the vector store", uuid, doc.metadata)
                    missing_docs.append(doc)
                    missing_uuids.append(uuid)

        if missing_docs:
            await self.vector_store.aadd_documents(
                documents=missing_docs, ids=missing_uuids
            )
        del missing_docs
        del missing_uuids

        activity = discord.Activity(
            name="CrystalVision", state="Ask me", type=discord.ActivityType.custom
        )
        await self.change_presence(activity=activity)

        log.info("Logged in as '%s' (ID: %s)", self.user, self.user.id)

        self._ready = True

    def decode_message(self, message: discord.Message) -> str:
        return message.content

    CARD_ITALICS = re.compile(r"\[\[i\]\](.*?)\[\[/\]\]")
    EX_BURST = re.compile(r"\[\[ex\]\]EX BURS[T|T ]\[\[/\]\]")

    def format_message(self, message: str) -> str:
        answer = self.EX_BURST.sub("《EX》", message)
        answer = re.sub(
            "|".join(self.emoji_mapping.keys()),
            lambda match: self.emoji_mapping[match.group(0)],
            answer,
        )
        answer = re.sub(r"\u2029\s+|\u2029\s", "\n", answer)
        answer = self.CARD_ITALICS.sub(r"*\1*", answer)
        return answer

    async def generate(self, content, context) -> str:
        result = await self.agent.ainvoke(content)
        answer = self.format_message(result["output"])

        # TODO: Maybe send 2+ messages instead
        if len(answer) > 2000:
            log.warning("The initial response is too long")

        return answer

    async def generate_thread_title(self, initial_message, response) -> str:
        kwargs = self.prompts.get("title_thread", {})

        result = await self.ollama.generate(
            model=self.model,
            prompt=kwargs.get("prompt"),
            system=kwargs.get("system").format(
                initial_message=initial_message, response=response
            ),
            keep_alive=-1,
        )
        log.debug(result)

        return result["response"]

    async def on_raw_reaction_add(
        self, payload: discord.RawReactionActionEvent
    ) -> None:
        """Triggers when a reaction is added to a message"""
        channel = await self.fetch_channel(payload.channel_id)
        message = await channel.fetch_message(payload.message_id)
        user = await self.fetch_user(payload.message_author_id)
        emoji = payload.emoji.name

        if user == self.user:
            log.debug(
                "User '%s' added reaction %s  in channel '%s'", user, emoji, channel
            )

            if emoji == "❌":
                await message.delete()

    @thinking()
    async def reply_to_message(self, message: discord.Message) -> None:
        resp_channel = message.channel

        response = await self.generate(
            self.decode_message(message),
            None,
        )

        mention_author = True
        if resp_channel.type == ChannelType.text:
            thread_name = await self.generate_thread_title(message.content, response)
            resp_channel = await resp_channel.create_thread(
                name=thread_name, message=message, auto_archive_duration=60
            )
            mention_author = False

        await resp_channel.send(response, mention_author=mention_author)

    async def on_message(self, message: discord.Message) -> None:
        if not self._ready:
            log.warning("%s is not ready", self.user)
            return

        if self.user == message.author:
            # don't respond to ourselves
            return

        log.debug(message)
        if (
            self.user.mentioned_in(message)
            or f"<@!{self.user.id}>" in message.content
            or isinstance(message.channel, channel.DMChannel)
        ):
            await self.reply_to_message(message)


if __name__ == "__main__":
    from langchain_ollama import OllamaEmbeddings, OllamaLLM

    client = Client()

    assert (embed_model := os.getenv("OLLAMA_EMBED_MODEL")), "No embed model provided"
    assert (chat_model := os.getenv("OLLAMA_CHAT_MODEL")), "No chat model provided"

    for model in (embed_model, chat_model):
        if model not in client.list():
            log.warning("Downloading model: %s", model)
            client.pull(model)
        log.info("Found model: %s", model)

    bot = CrystalClient(
        ollama=AsyncClient(),
        intents=intents,
        embeddings=OllamaEmbeddings(model=embed_model),
        llm=OllamaLLM(model=chat_model, temperature=0.0),
    )
    bot.run(os.getenv("DISCORD_TOKEN"))

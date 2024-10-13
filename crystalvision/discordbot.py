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
from typing import Optional
from functools import cached_property, wraps

import discord
from discord import Intents, ChannelType, channel
from ollama import Client, AsyncClient
from langchain_core.language_models import BaseLLM
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

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

        self._ready: bool = False

    async def on_ready(self) -> None:
        """Trigger when bot is ready/online"""
        if PROMPTS_JSON.exists():
            with open(PROMPTS_JSON, "r") as fp:
                self.prompts = json.load(fp)
        else:
            log.error("Could not find prompts json (%s)", PROMPTS_JSON)

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

    @cached_property
    def retriever(self) -> VectorStoreRetriever:
        return self.vector_store.as_retriever()

    def decode_message(self, message: discord.Message) -> str:
        return message.content

    async def generate(self, content, context) -> str:
        kwargs = self.prompts.get("general", {})
        print(content)

        system_prompt = (
            "Use the following pieces of retrieved context to inform your response."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("system", kwargs.get("system", "You are a chatbot.")),
                ("human", "{input}"),
            ]
        )

        qa_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(self.vector_store.as_retriever(), qa_chain)

        result = await rag_chain.ainvoke({"input": content})
        print(result)
        answer = result["answer"]

        if len(answer) > 2000:
            log.warning("The initial response is too long: \n%s", answer)
            sub_result = await self.ollama.generate(
                model=self.model,
                prompt=answer,
                system="Make this response more concise but still formatted for discord. It must be less than 2000 characters long.",
            )
            answer = sub_result["response"]

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

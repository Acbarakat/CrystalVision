"""
Serve a FFTCG discord bot.

TODO:
- Use RAG Model and context from message
- Use message history as context, with a max limit deque,
only pulling from the last checkpoint, maybe on disk db? or in mem only

"""

import os
import json
import logging
import asyncio
from typing import Optional
from functools import wraps

import discord
from discord import Intents, ChannelType, channel
from ollama import Client, AsyncClient

try:
    from lang import PROMPTS_JSON
except (ModuleNotFoundError, ImportError):
    from crystalvision.lang import PROMPTS_JSON


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

    def __init__(self, *args, ollama: Optional[AsyncClient] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert ollama, "No ollama AsyncClient provided"

        self.ollama: Optional[AsyncClient] = ollama
        self.model: str = os.getenv("OLLAMA_CHAT_MODEL")
        self.prompts: dict = {}

    async def on_ready(self) -> None:
        """Trigger when bot is ready/online"""
        if PROMPTS_JSON.exists():
            with open(PROMPTS_JSON, "r") as fp:
                self.prompts = json.load(fp)
        else:
            log.error("Could not find prompts json (%s)", PROMPTS_JSON)

        activity = discord.Activity(
            name="CrystalVision", state="Ask me", type=discord.ActivityType.custom
        )
        await self.change_presence(activity=activity)

        log.info("Logged in as '%s' (ID: %s)", self.user, self.user.id)

    async def generate(self, content, context) -> str:
        kwargs = self.prompts.get("general", {})

        result = await self.ollama.generate(
            model=self.model, prompt=content, system=kwargs.get("system"), keep_alive=-1
        )
        log.debug(result)

        return result["response"]

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
            message.content,
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
        if self.user == message.author:
            # don't respond to ourselves
            return

        if self.user.mentioned_in(message) or isinstance(
            message.channel, channel.DMChannel
        ):
            await self.reply_to_message(message)

        else:
            log.debug(message)


if __name__ == "__main__":
    client = Client(host="http://ollama:11434")

    assert (embed_model := os.getenv("OLLAMA_EMBED_MODEL")), "No embed model provided"
    assert (chat_model := os.getenv("OLLAMA_CHAT_MODEL")), "No chat model provided"

    for model in (embed_model, chat_model):
        if model not in client.list():
            log.warning("Downloading model: %s", model)
            client.pull(model)
        log.info("Found model: %s", model)

    bot = CrystalClient(ollama=AsyncClient(host="http://ollama:11434"), intents=intents)
    bot.run(os.getenv("DISCORD_TOKEN"))

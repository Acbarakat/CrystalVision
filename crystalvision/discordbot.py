"""
Serve a FFTCG discord bot.

TODO:
- Use RAG Model and context from message
- Use message history as context, with a max limit deque,
only pulling from the last checkpoint, maybe on disk db? or in mem only

"""

import os
import logging
import asyncio
from typing import Optional

import discord
from discord import Intents, ChannelType, channel
import discord.message
from ollama import Client, AsyncClient


log = logging.getLogger("discord.crystalvision")
log.setLevel(logging.DEBUG)

intents = Intents.default()
intents.members = True
intents.message_content = True
intents.reactions = True
intents.guilds = True


class CrystalClient(discord.Client):
    """A discord bot for FFTCG"""

    def __init__(self, *args, ollama: Optional[AsyncClient] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert ollama, "No ollama AsyncClient provided"

        self.ollama: Optional[AsyncClient] = ollama
        self.model: str = os.getenv("OLLAMA_CHAT_MODEL")

    async def on_ready(self) -> None:
        """Trigger when bot is ready/online"""
        activity = discord.Activity(
            name="CrystalVision", state="Ask me", type=discord.ActivityType.custom
        )
        await self.change_presence(activity=activity)

        log.info("Logged in as '%s' (ID: %s)", self.user, self.user.id)

    async def thinking(self, message: discord.Message, timeout: int = 999) -> None:
        """Add a thinking reaction until timeout to a message"""
        try:
            await message.add_reaction("🤔")
            async with message.channel.typing():
                await asyncio.sleep(timeout)
        except Exception:
            pass
        finally:
            await message.remove_reaction("🤔", self.user)

    async def generate(self, content, context) -> str:
        result = await self.ollama.generate(
            model=self.model, prompt=content, context=context, keep_alive=-1
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

    async def on_message(self, message: discord.Message) -> None:
        if self.user == message.author:
            # don't respond to ourselves
            return

        if self.user.mentioned_in(message) or isinstance(
            message.channel, channel.DMChannel
        ):
            resp_channel = message.channel

            task = asyncio.create_task(self.thinking(message))
            response = await self.generate(
                "How are you? Respond in the most eloquent way possible and formatted for discord.",
                None,
            )

            mention_author = True
            if resp_channel.type == ChannelType.text:
                resp_channel = await resp_channel.create_thread(
                    name="CrystalVision Says", message=message, auto_archive_duration=60
                )
                mention_author = False

            await resp_channel.send(response, mention_author=mention_author)
            task.cancel()

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

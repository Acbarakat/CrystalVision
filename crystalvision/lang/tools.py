from typing import Optional, List, Type, Any, Tuple

import pandas as pd
from langchain.tools import BaseTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from pydantic import BaseModel, Field, field_validator, field_serializer

try:
    from .loaders import CARD_CODE
except (ModuleNotFoundError, ImportError):
    from crystalvision.lang.loaders import CARD_CODE


class MultipleImageInput(BaseModel):
    card_codes: List[str] = Field(description="A list of card code(s)")

    @field_validator("card_codes", mode="before")
    @classmethod
    def validate_card_codes(cls, v: Any):
        # Check if input is a string that looks like a list, e.g., "['1-001H']"
        if isinstance(v, str):
            # Evaluate the string to convert it to a list
            if (v := CARD_CODE.findall(v)) and len(v) < 1:
                raise ValueError("No valid card codes provided.")
        return v

    @field_serializer("card_codes")
    @classmethod
    def parse_card_codes(cls, v: Any):
        # Check if input is a string that looks like a list, e.g., "['1-001H']"
        if isinstance(v, str):
            # Evaluate the string to convert it to a list
            v = CARD_CODE.findall(v)
        return v


class MultiImageEmbedTool(BaseTool):
    name: str = "MultiImageEmbedTool"
    # description: str = "Shows multiple images from the given a list of card code(s)."
    description: str = (
        "Shows multiple images or retrieve image(s) URLs from the given a list of card code(s) formatted for discord."
    )
    return_direct: bool = False
    df: pd.DataFrame = pd.DataFrame()
    response_format: str = "content_and_artifact"

    # Define the schema for arguments that the tool will accept
    args_schema: Type[BaseModel] = MultipleImageInput

    def __init__(self, df: pd.DataFrame = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.df = df

    def _run(
        self, card_codes: Any, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Tuple[str, List[Any]]:
        """Fetch images from the URLs and return a list of embeds suitable for Discord."""
        from discord import Embed

        card_codes = MultipleImageInput.parse_card_codes(card_codes)

        content = []
        artifacts = []
        for _, row in self.df.query("`code` in @card_codes").iterrows():
            content.append(f"({row['name_en'].upper()} {row['code']})[{row['images']}]")
            embed = Embed(title=f"{row['name_en'].upper()} {row['code']}")
            embed.set_image(url=row["images"])
            artifacts.append(embed)

        # content = "\u2029".join(content)
        content = "\r\n".join(content)

        return (content, artifacts)

    async def _arun(
        self,
        card_codes: Any,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[str, List[Any]]:
        """Asynchronous version of the tool."""
        from discord import Embed

        card_codes = MultipleImageInput.parse_card_codes(card_codes)

        content = []
        artifacts = []
        for _, row in self.df.query("`code` in @card_codes").iterrows():
            content.append(f"[{row['name_en'].upper()} {row['code']}]({row['images']})")
            embed = Embed(title=f"{row['name_en'].upper()} {row['code']}")
            embed.set_image(url=row["images"])
            artifacts.append(embed)

        content = "\u2029".join(content)

        return (content, artifacts)

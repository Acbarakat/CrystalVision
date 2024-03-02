# -*- coding: utf-8 -*-
"""
Gather card API data and download images.

Attributes:
    DATA_DIR (str): The root data folder
    CARD_API_FILEPATH (str): Save card API data as JSON

Todo:
    * Archive all image data
    * Fill data with missing Anniversary Only cards
    * Mist Dragon (Full Art) is missing
    * Add Illustrator data (http://www.square-enix-shop.com/jp/ff-tcg/card/illust_index.html)

"""
import asyncio
import json
import os
import typing
from io import BytesIO

from tqdm.asyncio import tqdm
import aiohttp
import aiofiles
import aiofiles.os as aioos
import requests
import pandas as pd
from PIL import ImageFile, Image

try:
    from .base import MISSING_CARDS_FILEPATH, CARD_API_FILEPATH, DATA_DIR
except ImportError:
    from crystalvision.data.base import (
        MISSING_CARDS_FILEPATH,
        CARD_API_FILEPATH,
        DATA_DIR,
    )


def download_and_save() -> dict:
    """
    Download and return FFTCG Card data.

    Returns:
        dict: a json/dict of FFTCG Card API
    """
    with requests.get("https://fftcg.square-enix-games.com/en/get-cards") as url:
        data = url.json()

    with open(MISSING_CARDS_FILEPATH) as fp:
        missing_cards = json.load(fp)
        for card in missing_cards:
            card = {key.lower(): value for key, value in card.items()}
            data["cards"].append(card)

    codes = set()
    duplicates = []
    for idx, c in enumerate(data["cards"]):
        key = "Code" if "Code" in c else "code"
        if c[key] in codes:
            duplicates.append((idx, c[key]))
        else:
            codes.add(c[key])
        for d in ("thumbs", "full"):
            extra = []
            for v in c["images"][d]:
                for lang in ("_de", "_es", "_fr", "_it"):
                    extra.append(
                        v.replace("_eg.jpg", f"{lang}.jpg").replace("_eg_", f"{lang}_")
                    )

            c["images"][d] += extra

        if "image" in c:
            del c["image"]

    if duplicates:
        for d, code in duplicates[::-1]:
            print(f"Found duplicate: {code}")
            del data["cards"][d]

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    with open(CARD_API_FILEPATH, "w+") as fp:
        json.dump(data, fp, indent=4)

    return data


async def download_image(
    img_url: str,
    subfolder: str = "img",
    fname: typing.Any = None,
    crop: typing.Any = None,
    resize: typing.Any = None,
) -> str:
    """
    Download image and return on-disk destination.

    Args:
        img_url (str): The URL of the image
        subfolder (str): The subfolder location the image
            will downloaded into
            (default is 'img')
        fname (str): The name of the image
            (default is derived from img_url)
        crop (tuple): The crop endpoint (left, top, right, bottom)
            (default is `None`)
        resize (tuple): The size to rescale image to
            (default is `None`)

    Returns:
        str: the on-disk filepath of downloaded image
    """
    if fname is None:
        fname = img_url.split("/")[-1]
    dst = os.path.join(DATA_DIR, subfolder, fname)
    if await aioos.path.exists(dst):
        return dst

    async with aiohttp.ClientSession() as session:
        async with session.get(img_url, allow_redirects=True) as resp:
            content = await resp.read()

            if resp.status == 404:
                print(f"Failed to download {img_url}")
                return

    p = ImageFile.Parser()
    p.feed(content)

    # Convert to jpg
    img = p.close().convert("RGB")

    if crop:
        img = img.crop(crop)

    if resize:
        img.resize(resize, Image.LANCZOS)

    # img.save(dst)
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    async with aiofiles.open(dst, "wb") as file:
        await file.write(buffer.getbuffer())

    return dst


async def main(pargs) -> None:
    """Download FFTCG API data and download any missing card images."""
    data = download_and_save()

    img_urls = []
    for card in data["cards"]:
        img_urls += card["images"]["full"]

    images = await tqdm.gather(
        *[download_image(img_url) for img_url in img_urls],
        desc="full card images",
        unit="cards",
    )

    thumb_urls = []
    for card in data["cards"]:
        thumb_urls += card["images"]["thumbs"]

    images = await tqdm.gather(
        *[download_image(thumb_url, "thumb") for thumb_url in thumb_urls],
        desc="thumb card images",
        unit="cards",
    )

    df = pd.read_table(
        "http://www.square-enix-shop.com/jp/ff-tcg/card/data/list_card.txt", header=None
    )
    df.rename({0: "code", 1: "element", 2: "name_ja", 7: "image"}, axis=1, inplace=True)

    # Special case flip
    df.replace({"code": "PR-051/11-083R"}, {"code": "11-083R/PR-051"}, inplace=True)
    df.replace({"code": "PR-055/11-062R"}, {"code": "11-062R/PR-055"}, inplace=True)

    cleared_codes = []
    images = []
    for d in data["cards"]:
        key = "Code" if "Code" in d else "code"
        # Ignore Boss Deck, Crystal Cards
        if d[key].startswith("B-") or d[key].startswith("C-"):
            continue

        rows = df.query(
            f"code == '{d[key]}' or (code.str.endswith('/{d[key]}') and code.str.startswith('PR'))"
        )
        if rows.empty and d[key] not in cleared_codes:
            raise Exception(f"Can't find '{d[key]}'")
        cleared_codes.append(d[key])
        df.query(f"code != '{d[key]}'", inplace=True)
        df.query(
            f"~(code.str.endswith('/{d[key]}') and code.str.startswith('PR'))",
            inplace=True,
        )

        for _, row in rows.iterrows():
            img_loc = row["image"]
            if "_FL" in img_loc:
                fname = f"{d[key].split('/')[0]}_FL_jp.jpg"
            elif img_loc.startswith("pr/"):
                fname = f"{d[key].split('/')[0]}_PR_jp.jpg"
            else:
                fname = f"{d[key].split('/')[0]}_jp.jpg"

            if "image" in d:
                del d["image"]

            d["images"]["thumbs"].append(
                f"http://www.square-enix-shop.com/jp/ff-tcg/card/cimg/thumb/{fname}"
            )
            images.append(
                download_image(
                    f"http://www.square-enix-shop.com/jp/ff-tcg/card/cimg/thumb/{img_loc}",
                    "thumb",
                    fname,
                    resize=(179, 250),
                    crop=(0, 0, 143, 200),
                )
            )

            d["images"]["full"].append(
                f"http://www.square-enix-shop.com/jp/ff-tcg/card/cimg/large/{fname}"
            )
            images.append(
                download_image(
                    f"http://www.square-enix-shop.com/jp/ff-tcg/card/cimg/large/{img_loc}",
                    "img",
                    fname,
                    resize=(429, 600),
                )
            )

    # Remainder of Cards
    print(df)

    with open(CARD_API_FILEPATH, "w+") as fp:
        json.dump(data, fp, indent=4)

    images = await tqdm.gather(*images, desc="JP card images", unit="cards")

    # Download testdata images
    df = pd.read_json(pargs.file)

    images = [
        download_image(row["uri"], "test", f"{idx}.jpg") for idx, row in df.iterrows()
    ]
    images = await tqdm.gather(*images, desc="testing images", unit="cards")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Gather all assets")
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        required=True,
        help="Path to the JSON validation assets file.",
    )

    args = parser.parse_args()

    if not args.file.exists():
        raise FileNotFoundError(str(args.file))

    asyncio.run(main(args))

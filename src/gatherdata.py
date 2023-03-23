# -*- coding: utf-8 -*-
"""
Gather card API data and download images

Attributes:
    DATA_DIR (str): The root data folder
    CARD_API_FILEPATH (str): Save card API data as JSON

Todo:
    * Archive all image data

"""
import asyncio
import json
import os
import typing

import requests
import pandas as pd
from PIL import ImageFile, Image


DATA_DIR = os.path.join(".", "data")
CARD_API_FILEPATH = os.path.join(DATA_DIR, "cards.json")


def download_and_save() -> dict:
    '''
    Download and return FFTCG Card data.

    Returns:
        dict: a json/dict of FFTCG Card API
    '''

    with requests.get("https://fftcg.square-enix-games.com/en/get-cards") as url:
        data = url.json()
        for c in data["cards"]:
            for d in ("thumbs", "full"):
                extra = []
                for v in c["images"][d]:
                    for lang in ("_de", "_es", "_fr", "_it"):
                        extra.append(v.replace("_eg.jpg", f"{lang}.jpg").replace("_eg_", f"{lang}_"))

                c["images"][d] += extra


        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        with open(CARD_API_FILEPATH, "w+") as fp:
            json.dump(data, fp, indent=4)

    return data


async def download_image(img_url: str,
                         subfolder: str='img',
                         fname: typing.Any=None,
                         crop: typing.Any=None,
                         resize: typing.Any=None) -> str:
    '''
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
    '''
    if fname is None:
        fname = img_url.split("/")[-1]
    dst = os.path.join(DATA_DIR, subfolder, fname)
    if os.path.exists(dst):
        return dst

    p = ImageFile.Parser()
    with requests.get(img_url, allow_redirects=True) as url:
        p.feed(url.content)

    # Convert to jpg
    img = p.close().convert('RGB')

    if crop:
        img = img.crop(crop)

    if resize:
        img.resize(resize, Image.LANCZOS)

    img.save(dst)

    return dst


async def main() -> None:
    '''
    Download FFTCG API data and download any missing card images
    '''
    data = download_and_save()

    img_urls = []
    for card in data["cards"]:
        img_urls += card["images"]["full"]

    images = asyncio.gather(*[download_image(img_url) for img_url in img_urls])
    await images
    print(images)

    thumb_urls = []
    for card in data["cards"]:
        thumb_urls += card["images"]["thumbs"]

    images = asyncio.gather(*[download_image(thumb_url, "thumb") for thumb_url in thumb_urls])
    await images
    print(images)

    df = pd.read_table("http://www.square-enix-shop.com/jp/ff-tcg/card/data/list_card.txt", header=None)
    df.rename({0: "Code", 1: "Element", 2: "Name", 7: "image"}, axis=1, inplace=True) #, "Element", "Name", "Opus", "Category_1", "Category_2", "6", "img", "8", "9", "10", "11"], inplace=True)
    # print(df['image'])
    images = []
    for d in data["cards"]:
        # Ignore Boss Deck, Crystal Cards
        if d["Code"].startswith("B-") or d["Code"].startswith("C-"):
            continue
        rows = df.query(f"Code == '{d['Code']}'")
        if rows.empty:
            raise Exception(f"Can't find '{d['Code']}'")
        for idx, row in rows.iterrows():
            img_loc = row['image']
            if "_FL" in img_loc:
                fname = f"{d['Code'].split('/')[0]}_FL_jp.jpg"
            else:
                fname = f"{d['Code'].split('/')[0]}_jp.jpg"
            d["images"]["thumbs"].append(f"http://www.square-enix-shop.com/jp/ff-tcg/card/cimg/thumb/{fname}")
            images.append(download_image(f"http://www.square-enix-shop.com/jp/ff-tcg/card/cimg/thumb/{img_loc}",
                                        "thumb",
                                        fname,
                                        resize=(179, 250),
                                        crop=(0, 0, 143, 200)))
            d["images"]["full"].append(f"http://www.square-enix-shop.com/jp/ff-tcg/card/cimg/large/{fname}")
            images.append(download_image(f"http://www.square-enix-shop.com/jp/ff-tcg/card/cimg/large/{img_loc}",
                                        "img",
                                        fname,
                                        resize=(429, 600)))

    with open(CARD_API_FILEPATH, "w+") as fp:
        json.dump(data, fp, indent=4)

    images = asyncio.gather(*images)
    await images
    print(images)


if __name__ == "__main__":
    asyncio.run(main())

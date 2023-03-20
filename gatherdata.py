import asyncio
import json
import os

import requests


def download_and_save() -> dict:
    '''
    Download and return FFTCG Card data.

    Returns:
        dict: a json/dict of FFTCG Card API
    '''
    with requests.get("https://fftcg.square-enix-games.com/en/get-cards") as url:
        data = url.json()
        for c in data["cards"]:
            extra = []
            for v in c["images"]["thumbs"]:
                for lang in ("_de", "_es", "_fr", "_it"):
                    extra.append(v.replace("_eg.jpg", f"{lang}.jpg").replace("_eg_", f"{lang}_"))

            c["images"]["thumbs"] += extra

        with open("cards.json", "w+") as fp:
            json.dump(data, fp, indent=4)

    # TODO: Get data from JP
    # http://www.square-enix-shop.com/jp/ff-tcg/card/data/list_card.txt
    # http://www.square-enix-shop.com/jp/ff-tcg/card/cimg/large/opus19/19-001R.png (400 x 559)
    # http://www.square-enix-shop.com/jp/ff-tcg/card/cimg/thumb/opus19/19-001R.png (143 x 249) Extra mirror on bottom

    return data


async def download_image(img_url, prefix='.\\img') -> str:
    '''
    Download image and return on-disk destination.

    Args:
        img_url (str): The URL of the image
        prefix (str): The location the image will downloaded to
            (default is'.\\img')

    Returns:
        str: the on-disk filepath of downloaded image
    '''
    fname = img_url.split("/")[-1]
    dst = f"{prefix}\\{fname}"
    if os.path.exists(dst):
        return dst

    with requests.get(img_url, allow_redirects=True) as url:
        with open(f"{prefix}\\{fname}", "wb+") as fp:
            fp.write(url.content)

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

    images = asyncio.gather(*[download_image(thumb_url, ".\\thumb") for thumb_url in thumb_urls])
    await images
    print(images)


if __name__ == "__main__":
    asyncio.run(main())

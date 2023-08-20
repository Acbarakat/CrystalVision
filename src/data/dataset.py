# -*- coding: utf-8 -*-
import json
from typing import Tuple

import pandas as pd
import tensorflow as tf
from keras import layers

try:
    from __init__ import CARD_API_FILEPATH
except ModuleNotFoundError:
    from data import CARD_API_FILEPATH


def make_database() -> pd.DataFrame:
    """
    Load card data and clean up any issue found in the API.

    Returns:
        Card API dataframe
    """
    with open(CARD_API_FILEPATH) as fp:
        data = json.load(fp)["cards"]

    df = pd.DataFrame(data)
    df["thumbs"] = df["images"].apply(lambda i: [j.split("/")[-1] for j in i["thumbs"]])
    df["images"] = df["images"].apply(lambda i: [j.split("/")[-1] for j in i["full"]])
    df["ex_burst"] = df["ex_burst"].apply(lambda i: i == "\u25cb" or i == "1").astype(bool)
    df["multicard"] = df["multicard"].apply(lambda i: i == "\u25cb" or i == "1").astype(bool)
    df["mono"] = df["element"].apply(lambda i: len(i) == 1 if i else True).astype(str)
    df["element"] = df["element"].str.join("_")
    df["power"] = df["power"].str.replace(" ", "").replace("\u2015", "").replace("\uff0d", "")

    return df


def extendDataset(ds: tf.data.Dataset,
                  seed: int | None = None,
                  name: str | None = None,
                  batch_size: int | None = 32,
                  shuffle: bool = True,
                  reshuffle_each_iteration: bool = True,
                  flip_horizontal: bool = False,
                  flip_vertical: bool = True,
                  brightness: float = 0.2,
                  contrast: Tuple[float] | None = (0.5, 1.25),
                  saturation: Tuple[float] | None = (0.65, 1.75),
                  hue: float = 0.025) -> tf.data.Dataset:
    """
    Preprocess and add any extra augmented entries to the dataset.

    Args:
        ds (tf.data.DataFrame): All tensorflow image Dataset
        seed (int): An optional integer used to create a random seed
            (default is None)
        name (str): Optional name for the Dataset
            (default is None)
        batch_size (int): Size of the batches of data.
            If `None`, the data will not be batched
            (default is 32)
        shuffle (bool): Whether to shuffle the data.
            If set to False, sorts the data in alphanumeric order.
            (default is True)
        reshuffle_each_iteration: Whether the shuffle order should
            be different for each epoch
            (default is True)
        flip_horizontal (list): Add additional horizontal flipped images
            (default is False)
        flip_vertical (list): Add additional vertical flipped images
            (default is True)
        brightness (float): A delta randomly picked in the interval
            [-max_delta, max_delta) applied across dataset
            (default is 0.2)
        contast (tuple[flost]): a contrast_factor randomly picked
            in the interval [lower, upper) applied across the dataset
            (default is (0.5, 1.25))
        saturation (tuple[flost]):  a saturation_factor randomly picked
            in the interval [lower, upper) applied across the dataset
            (default is (0.65, 1.75))
        hue (float): a delta randomly picked in the interval
            [-max_delta, max_delta) applied across the dataset
            (default is 0.025)

    Returns:
        Dataset
    """
    assert brightness >= 0.0, "brightness must be >= 0.0"

    preprocess_layer = layers.Rescaling(1. / 255)
    # preprocess_layer = layers.Rescaling(scale=1./127.5, offset=-1)

    if name:
        ds.element_spec[0]._name = f"orig_{name}"

    ds = ds.map(tf.autograph.experimental.do_not_convert(
        lambda x, y: (preprocess_layer(x), y)),
        name=name
    )
    if flip_horizontal:
        raise NotImplementedError("flip_horizontal")

    if flip_vertical:
        veritcal_ds = ds.map(
            lambda x, y: (tf.image.flip_up_down(x), y),
            name=f"vertical_{name}"
        )
        ds = ds.concatenate(veritcal_ds)

    ds = ds.cache()

    effects = []

    if brightness:
        ds = ds.map(
            lambda x, y: (tf.clip_by_value(tf.image.random_brightness(x, brightness, seed=seed), 0.0, 1.0), y),
            name=f"brightness_{name}"
        )

    if contrast:
        ds = ds.map(
            lambda x, y: (tf.clip_by_value(tf.image.random_contrast(x, *contrast, seed=seed), 0.0, 1.0), y),
            name=f"contrast_{name}"
        )

    if saturation:
        ds = ds.map(
            lambda x, y: (tf.clip_by_value(tf.image.random_saturation(x, *saturation, seed=seed), 0.0, 1.0), y),
            name=f"saturated_{name}"
        )

    if hue:
        ds = ds.map(
            lambda x, y: (tf.clip_by_value(tf.image.random_hue(x, hue, seed=seed), 0.0, 1.0), y),
            name=f"hue_{name}"
        )

    for effect in effects:
        ds = ds.concatenate(effect)

    if shuffle:
        ds = ds.shuffle(buffer_size=ds.cardinality(),
                        seed=seed,
                        reshuffle_each_iteration=reshuffle_each_iteration,
                        name=f"shuffled_{name}")

    if batch_size:
        ds = ds.batch(batch_size, name=f"batch_{name}")
        # ds.batch_size

    ds = ds.prefetch(tf.data.AUTOTUNE)

    ds.element_spec[0]._name = name

    return ds

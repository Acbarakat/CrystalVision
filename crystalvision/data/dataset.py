# -*- coding: utf-8 -*-
"""
Getting and transforming dataframes/datasets.

Attributes:
    SRC_DIR (str): Where the src folder is
    MODEL_DIR (str): Where the models are stored

Todo:
    * N/A

"""
import os
import json
from copy import deepcopy
from typing import Tuple, List, Any

import pandas as pd
import numpy as np
from dogpile.cache import make_region
from keras import layers, backend

try:
    from .base import CARD_API_FILEPATH, DATA_DIR
except ImportError:
    from crystalvision.data.base import CARD_API_FILEPATH, DATA_DIR


if backend.backend() == "tensorflow":
    from keras.src.utils.image_dataset_utils import (
        paths_and_labels_to_dataset as paths_and_labels_to_dataset_tf,
    )
    from keras.src.utils.module_utils import tensorflow as tf

    IterableDataset = object


elif backend.backend() == "torch":
    import torch
    from torch import manual_seed
    from torch.utils.data import IterableDataset, DataLoader, ChainDataset
    from torchvision import transforms, set_image_backend
    from torchvision.datasets.folder import default_loader
    from torchvision.transforms.functional import InterpolationMode

    torch.autograd.profiler.profile(enabled=False)
    torch.autograd.profiler.emit_itt(enabled=False)
    torch.autograd.profiler.emit_nvtx(enabled=False)
    torch.autograd.set_detect_anomaly(mode=False)
    torch.autograd.gradcheck.fast_mode = True
    torch.autograd.gradgradcheck.fast_mode = True
    torch.backends.cudnn.set_flags(_benchmark=True)

    set_image_backend("accimage")


# Define a cache region
cache = make_region().configure(
    "dogpile.cache.memory", expiration_time=3600  # Cache expiration time in seconds
)


def make_database(clear_extras: bool = False) -> pd.DataFrame:
    """
    Load card data and clean up any issue found in the API.

    Returns:
        Card API DataFrame
    """
    with open(CARD_API_FILEPATH, "r") as fp:
        data = json.load(fp)["cards"]

    df = pd.DataFrame(data)
    # Remove the extra lang columns
    if clear_extras:
        for lang in ("_es", "_de", "_fr", "_it", "_ja"):
            df = df.loc[:, ~df.columns.str.endswith(lang)]

    df["thumbs"] = df["images"].apply(lambda i: [j.split("/")[-1] for j in i["thumbs"]])
    df["images"] = df["images"].apply(lambda i: [j.split("/")[-1] for j in i["full"]])
    df["ex_burst"] = (
        df["ex_burst"].apply(lambda i: i == "\u25cb" or i == "1").astype(bool)
    )
    df["multicard"] = (
        df["multicard"].apply(lambda i: i == "\u25cb" or i == "1").astype(bool)
    )
    df["limit_break"] = df["text_en"].str.contains("Limit Break --")
    df["icons"] = df[["ex_burst", "multicard", "limit_break"]].apply(
        lambda row: tuple(row[row].index) if not row[row].index.empty else ("no_icon",),
        axis=1,
    )
    df["mono"] = df["element"].apply(lambda i: len(i) == 1 if i else True).astype(bool)
    df["element_v2"] = df["element"].apply(
        lambda x: tuple(x) if x is not None else tuple()
    )
    df["element"] = df["element"].str.join("_")
    df["power"] = (
        df["power"].str.replace(" ", "").replace("\u2015", "").replace("\uff0d", "")
    )

    return df


def imagine_database(image: str = "thumbs", clear_extras: bool = False) -> pd.DataFrame:
    """
    Explode the database based on `image` kwarg.

    - Filter out `Crystal` cards
    - Filter out `Boss` cards
    - Filter out Full Art cards
    - Filter out Promo cards
    - Filter out some language cards

    Args:
        image (str): The column to explode containing lists of image sources.
            (default is `thumbs`)

    Returns:
        Card API DataFrame
    """
    assert image in ("thumbs", "images"), f"'{image}' is not valid"
    df = make_database().explode(image)
    # df["Class"] = df['thumbs'].apply(lambda x: 'Full Art' if '_fl' in x.lower() else '')

    # Ignore Crystal Tokens
    df.query("type_en != 'Crystal'", inplace=True)

    # Ignore Boss Deck cards
    df.query("rarity != 'B'", inplace=True)

    # Ignore Full Art Cards
    df.query(
        f"~{image}.str.contains('_FL') and ~{image}.str.contains('_2_')", inplace=True
    )

    # Ignore Promo Cards, they tend to be Full Art
    df.query(f"~{image}.str.contains('_PR')", inplace=True)

    # Ignore
    df.query(f"~{image}.str.contains('_premium')", inplace=True)

    # WA: Bad Download/Image from server
    df.query(
        f"{image} not in ('8-080C_es.jpg', '11-138S_fr.jpg', '12-049H_fr_Premium.jpg', '13-106H_de.jpg')",
        inplace=True,
    )

    # Source image folder
    df = df.copy()  # WA: for pandas modification on slice
    if image == "images":
        image_dir = os.path.abspath(os.path.join(DATA_DIR, "img"))
        df.rename({"images": "filename"}, axis=1, inplace=True)
    else:
        image_dir = os.path.abspath(os.path.join(DATA_DIR, "thumb"))
        df.rename({"thumbs": "filename"}, axis=1, inplace=True)
    df["filename"] = image_dir + os.sep + df["filename"]

    # Remove the extra lang columns
    if clear_extras:
        for lang in ("_es", "_de", "_fr", "_it", "_ja"):
            df = df.loc[:, ~df.columns.str.endswith(lang)]

    return df


def extend_dataset_tf(
    ds: Any,
    seed: int | None = None,
    name: str | None = None,
    batch_size: int | None = 32,
    shuffle: bool = False,
    reshuffle_each_iteration: bool = True,
    flip_horizontal: bool = False,
    flip_vertical: bool = True,
    brightness: float = 0.1,
    contrast: Tuple[float] | None = (0.80, 1.25),
    saturation: Tuple[float] | None = (0.65, 1.75),
    hue: float | None = 0.025,
    additive: bool = True,
    **kwargs,
) -> Any:
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
            (default is False)
        reshuffle_each_iteration: Whether the shuffle order should
            be different for each epoch
            (default is True)
        flip_horizontal (list): Add additional horizontal flipped images
            (default is False)
        flip_vertical (list): Add additional vertical flipped images
            (default is True)
        brightness (float): A delta randomly picked in the interval
            [-max_delta, max_delta) applied across dataset
            (default is 0.1)
        contast (tuple[flost]): a contrast_factor randomly picked
            in the interval [lower, upper) applied across the dataset
            (default is (0.80, 1.25))
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

    if not name:
        name = "unknown"

    ds.element_spec[0]._name = f"orig_{name}"  # pylint: disable=W0212

    ds = ds.map(
        lambda x, y: (layers.Rescaling(1.0 / 255)(x), y),
        name=name,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if flip_horizontal:
        raise NotImplementedError("flip_horizontal")

    if flip_vertical:
        vertical_ds = ds.map(
            lambda x, y: (tf.image.flip_up_down(x), y),
            name=f"vertical_{name}",
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        cardinality = ds.cardinality() * 2
        # ds = ds.concatenate(veritcal_ds)
        ds = (
            tf.data.Dataset.from_tensor_slices([ds, vertical_ds])
            .interleave(
                lambda x: x,
                cycle_length=2,
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .apply(tf.data.experimental.assert_cardinality(cardinality))
        )

    ds = ds.cache()

    effects = []

    if brightness:
        effects.append(
            lambda ads: ads.map(
                lambda x, y: (
                    tf.clip_by_value(
                        tf.image.random_brightness(x, brightness, seed=seed), 0.0, 1.0
                    ),
                    y,
                ),
                name=f"brightness_{name}",
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        )

    if contrast:
        effects.append(
            lambda ads: ads.map(
                lambda x, y: (
                    tf.clip_by_value(
                        tf.image.random_contrast(x, *contrast, seed=seed), 0.0, 1.0
                    ),
                    y,
                ),
                name=f"contrast_{name}",
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        )

    if saturation:
        effects.append(
            lambda ads: ads.map(
                lambda x, y: (
                    tf.clip_by_value(
                        tf.image.random_saturation(x, *saturation, seed=seed), 0.0, 1.0
                    ),
                    y,
                ),
                name=f"saturated_{name}",
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        )

    if hue:
        effects.append(
            lambda ads: ads.map(
                lambda x, y: (
                    tf.clip_by_value(tf.image.random_hue(x, hue, seed=seed), 0.0, 1.0),
                    y,
                ),
                name=f"hue_{name}",
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        )

    if additive:
        effects = [effect(ds) for effect in effects]

    for effect in effects:
        if additive:
            ds = ds.concatenate(effect)
        else:
            ds = effect(ds)

    if shuffle:
        # buffer_size = batch_size * 2 if batch_size else ds.cardinality()
        ds = ds.shuffle(
            buffer_size=ds.cardinality(),
            seed=seed,
            reshuffle_each_iteration=reshuffle_each_iteration,
            name=f"shuffled_{name}",
        )

    if batch_size:
        ds = ds.batch(
            batch_size,
            name=f"batch_{name}",
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.prefetch(tf.data.AUTOTUNE)

    ds.element_spec[0]._name = name  # pylint: disable=W0212

    return ds


@cache.cache_on_arguments()
def get_image(path) -> Any:
    return default_loader(path)


class CustomDataset(IterableDataset):
    def __init__(
        self,
        paths,
        size,
        labels,
        name="customdata",
        interpolation="bilinear",
    ):
        interpolation = (
            InterpolationMode[interpolation.upper()]
            if isinstance(interpolation, str)
            else interpolation
        )

        self.paths = paths
        self.labels = np.array(labels)
        self.name = name
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size, interpolation),
            ]
        )

    def __repr__(self) -> str:
        return f"<Dataset name={self.name} lenght={len(self)}>"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = self.preprocess_image(self.paths[idx])
        return image, self.labels[idx]

    def preprocess_image(self, path):
        image = get_image(path)  # You need to define get_image function
        if self.transform:
            image = self.transform(image)
        return image.permute(1, 2, 0)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        for idx in range(worker_info.id, len(self), worker_info.num_workers):
            yield self.preprocess_image(self.paths[idx]), self.labels[idx]
        # print(worker_info, cache)

    def batch(self, batch_size: int):
        return DataLoader(
            self,
            batch_size=batch_size,
            pin_memory=batch_size < 256 if batch_size else True,
            num_workers=8,
            persistent_workers=os.environ.get("TORCH_DS_PERSISTENT_WORKS", "0") == "1",
        )


def paths_and_labels_to_dataset_torch(
    image_paths: List[str] = [],
    image_size: Tuple[int, int] = (448, 224),
    labels: List[Any] = [],
    interpolation="bilinear",
    **kwargs,
):
    interpolation = InterpolationMode[interpolation.upper()]

    return CustomDataset(image_paths, image_size, labels, interpolation=interpolation)


def extend_dataset_torch(
    ds: Any,
    seed: int | None = None,
    name: str | None = None,
    batch_size: int | None = 32,
    shuffle: bool = False,
    flip_horizontal: bool = False,
    flip_vertical: bool = True,
    brightness: float | Tuple[float, float] | None = 0.1,
    contrast: float | Tuple[float, float] | None = (0.80, 1.25),
    saturation: float | Tuple[float, float] | None = (0.65, 1.75),
    hue: float = 0.025,
    perspective: float | None = None,
    additive: bool = True,
    **kwargs,
):
    if seed:
        manual_seed(seed)

    if not name:
        name = "unknown"

    ds.name = f"orig_{name}"

    def append_transform(transform, tname=None, aug_ds=None):
        if aug_ds is None:
            aug_ds = deepcopy(ds)

        if isinstance(aug_ds, ChainDataset):
            for ads in aug_ds.datasets:
                append_transform(transform, tname=tname, aug_ds=ads)
        else:
            aug_ds.transform.transforms.append(transform)
            if tname:
                aug_ds.name = tname.format(name=name, old_name=aug_ds.name)
        return aug_ds

    if flip_horizontal:
        ds += append_transform(transforms.RandomHorizontalFlip(1.0), "fliph_{name}")

    if flip_vertical:
        ds += append_transform(transforms.RandomVerticalFlip(1.0), "flipv_{name}")

    augments = []

    if brightness is not None:
        augments.append(
            (transforms.ColorJitter(brightness=brightness), "bright_{name}")
        )

    if contrast is not None:
        augments.append((transforms.ColorJitter(contrast=contrast), "contrast_{name}"))

    if saturation is not None:
        augments.append(
            (transforms.ColorJitter(saturation=saturation), "saturation_{name}")
        )

    if hue is not None:
        augments.append((transforms.ColorJitter(hue=hue), "hue_{name}"))

    if additive:
        ds = ChainDataset([ds, *(append_transform(*aug) for aug in augments)])
    else:
        for ads in ds.datasets:
            ads.transform.transforms.extend([aug[0] for aug in augments])

    if perspective:
        ds += append_transform(
            transforms.RandomPerspective(perspective, p=1.0),
            tname="perspective_{old_name}",
        )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=batch_size < 256 if batch_size else True,
        num_workers=8,
        persistent_workers=os.environ.get("TORCH_DS_PERSISTENT_WORKS", "0") == "1",
    )


if backend.backend() == "tensorflow":
    paths_and_labels_to_dataset = paths_and_labels_to_dataset_tf
    extend_dataset = extend_dataset_tf
elif backend.backend() == "torch":
    paths_and_labels_to_dataset = paths_and_labels_to_dataset_torch
    extend_dataset = extend_dataset_torch
else:
    raise NotImplementedError(f"{backend.backend()} is not implemented")

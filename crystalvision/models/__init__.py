# -*- coding: utf-8 -*-
"""
Methods to work with models.

Todo:
    * ???

"""
import os
import argparse
from pathlib import Path

import pandas as pd
from keras.models import load_model

SRC_DIR = Path(os.path.dirname(__file__), "..")
MODEL_DIR = Path(SRC_DIR, "..", "data", "model")
TEST_IMG_DIR = Path(MODEL_DIR, "..", "test")

try:
    from .mixins.compiles import BinaryMixin
    from .base import CardModel
    from ..data.dataset import imagine_database, make_database
except ImportError:
    from crystalvision.models.mixins.compiles import BinaryMixin
    from crystalvision.models.base import CardModel
    from crystalvision.data.dataset import imagine_database, make_database


def tune_model(model: CardModel, num: int = 5, save_models: bool = True) -> None:
    parser = argparse.ArgumentParser(description="Model tuning command-line tool")
    parser.add_argument(
        "--num", "-n", type=int, default=num, help="The number of best models to keep"
    )
    parser.add_argument(
        "--random-state", "-r", type=int, default=23, help="The random state number"
    )
    parser.add_argument(
        "--disable-clear-cache",
        "-d",
        action="store_true",
        help="Disable the clearing of the tuning cache",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose mode"
    )

    args = parser.parse_args()

    df = imagine_database(clear_extras=True)

    vdf = pd.read_json(os.path.join(SRC_DIR, "testmodels.json"))
    vdf["filename"] = str(TEST_IMG_DIR) + os.sep + vdf["uri"].index.astype(str) + ".jpg"
    vdf.fillna({"full_art": 0, "foil": 1, "focal": 1}, inplace=True)
    vdf = vdf.merge(make_database(clear_extras=True), on="code", how="left", sort=False)
    vdf.drop(["thumbs", "images", "uri", "id"], axis=1, inplace=True)
    vdf.query("full_art != 1 and focal == 1", inplace=True)

    null_vdf = vdf[vdf["name_en"].isna()]
    if not null_vdf.empty:
        print(null_vdf)
        raise ValueError("There are null values in test dataset")

    m = model(df, vdf)

    if args.verbose:
        print(vdf)
        if isinstance(m, BinaryMixin):
            print(
                vdf.query("@m.feature_key == True")
                .groupby(m.stratify_cols)
                .count()["code"]
            )
        else:
            print(vdf.groupby(m.stratify_cols).count()["code"])

    if not args.disable_clear_cache:
        m.clear_cache()

    m.tune_and_save(
        num_models=args.num, random_state=args.random_state, save_models=save_models
    )

    if args.verbose:
        for i in range(1, args.num + 1):
            best_model = load_model(os.path.join(MODEL_DIR, f"{m.name}_{i}.h5"))
            print(best_model.summary())

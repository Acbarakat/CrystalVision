# -*- coding: utf-8 -*-
"""
Methods to work with models.

Todo:
    * ???

"""
import os
import argparse
import yaml
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


def tune_model(
    model: CardModel, num: int = 5, save_models: bool = True, clear_cache=False
) -> None:
    parser = argparse.ArgumentParser(description="Model tuning command-line tool")
    parser.add_argument(
        "--num", "-n", type=int, default=num, help="The number of best models to keep"
    )
    parser.add_argument(
        "--random-state", "-r", type=int, default=23, help="The random state number"
    )
    parser.add_argument(
        "--clear-cache",
        "-c",
        action="store_true",
        help="Disable the clearing of the tuning cache",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose mode"
    )
    parser.add_argument(
        "--validation",
        type=Path,
        help="The validation data JSON file",
        default=MODEL_DIR / ".." / "testmodels.json",
    )
    parser.add_argument(
        "--tune",
        type=Path,
        help="The tuning YAML file",
        default=SRC_DIR / "models" / "tune.yml",
    )

    args = parser.parse_args()

    if not args.validation.exists():
        raise FileNotFoundError(str(args.validation))

    if not args.tune.exists():
        raise FileNotFoundError(str(args.tune))

    with args.tune.open("r") as fp:
        tune_data = yaml.safe_load(fp)
        print(tune_data)

    df = imagine_database(clear_extras=True)
    df.attrs["seed"] = args.random_state

    vdf = pd.read_json(str(args.validation))
    vdf["filename"] = str(TEST_IMG_DIR) + os.sep + vdf["uri"].index.astype(str) + ".jpg"
    vdf.fillna({"full_art": 0, "foil": 1, "focal": 1}, inplace=True)
    vdf = vdf.merge(make_database(clear_extras=True), on="code", how="left", sort=False)
    vdf.drop(["thumbs", "images", "uri", "id"], axis=1, inplace=True)
    vdf.query("full_art != 1 and focal == 1", inplace=True)

    null_vdf = vdf[vdf["name_en"].isna()]
    if not null_vdf.empty:
        print(null_vdf)
        raise ValueError("There are null values in test dataset")

    m = model(df, vdf, **tune_data)

    if args.clear_cache or clear_cache:
        m.clear_cache()
        m.save_multilabels()

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

    m.tune_and_save(
        num_models=args.num, random_state=args.random_state, save_models=save_models
    )

    if args.verbose:
        for i in range(1, args.num + 1):
            best_model = load_model(os.path.join(MODEL_DIR, f"{m.name}_{i}.h5"))
            print(best_model.summary())

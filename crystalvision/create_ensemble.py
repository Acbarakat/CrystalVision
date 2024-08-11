# -*- coding: utf-8 -*-
"""
Test models against real-world (internet data).

Attributes:
    CATEGORIES (tuple): A tuple of the types of ``models`` trained
    IMAGES (np.ndarray): A list or URIs converted to loaded and resized
        image data
    DF (pd.DataFrame): A dataframe full of accurate card data which
        we will compare model predictions against

Todo:
    * Find more varied data, such as off-center or card with background
    * Explore minimize function further

"""
import json
import logging
from typing import Tuple

import numpy as np
import pandas as pd
from keras.models import load_model
from PIL import Image
from skimage.io import imread
from skimage.transform import resize as imresize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from scipy.optimize._minimize import MINIMIZE_METHODS

from crystalvision.data.base import DATA_DIR
from crystalvision.data.dataset import imagine_database, paths_and_labels_to_dataset
from crystalvision.models.base import MODEL_DIR
from crystalvision.models.ext.ensemble import hard_activation, MyEnsembleVoteClassifier


log = logging.getLogger()

IMAGE_DF: pd.DataFrame = pd.read_json(DATA_DIR / "testmodels.json")


def load_image(
    url: str, img_fname: str = "", resize: Tuple[int, int] | None = None
) -> np.ndarray:
    """
    Load image (and cache it).

    Args:
        url (str): The image URL
        img_fname (str): The file name we will save to
            (default will be derived from url)

    Returns:
        Image as ndarray
    """
    if not img_fname:
        img_fname = url.split("/")[-1]
        img_fname = img_fname.split("%2F")[-1]

    dst = DATA_DIR / "test" / img_fname
    log.debug("Loading %s", dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        data = imread(dst)
        if resize:
            data = imresize(data, resize, anti_aliasing=True, preserve_range=True)
        data = data[:, :, :3]
        return data * (1.0 / 255)

    if url.startswith("blob:"):
        url = url[5:]
    data = imread(url)
    log.info("Downloading %s", url)
    if img_fname.endswith(".jpg"):
        Image.fromarray(data).convert("RGB").save(dst)
    else:
        Image.fromarray(data).save(dst)

    if resize:
        data = imresize(data, resize, anti_aliasing=True, preserve_range=True)

    return data[:, :, :3] * (1.0 / 255)


def apply_load_image(row: pd.Series) -> np.ndarray:
    return load_image(row["uri"], f"{row.name}.jpg", resize=(250, 179))


def find_threshold(X, y_true, labels, method="nelder-mead"):
    def function_to_minimize(threshold):
        y_pred = hard_activation(X, threshold[0])
        y_pred = [labels[y] for y in y_pred]

        # this is the mean accuracy
        score = accuracy_score(y_true, y_pred)

        # change accuracy to error so that smaller is better
        score_to_minimize = 1.0 - score

        return score_to_minimize

    threshold = X.mean()

    best_score, best_threshold = 1.0, 1.0
    for method in MINIMIZE_METHODS:
        try:
            results = minimize(
                function_to_minimize,
                threshold,
                bounds=[(X.min(), X.max())],
                method=method,
            )
            print(results)
        except ValueError as err:
            print(method)
            print(err)
            continue
        if results["fun"] < best_score:
            best_score = results["fun"]
            best_threshold = results["x"]

    return best_threshold


def create_models(margs) -> pd.DataFrame:
    """
    Run all models and apply values in the dataframe.

    Returns:
        ImageData dataframe with yhat(s)
    """
    df: pd.DataFrame = IMAGE_DF.copy().set_index("code")
    missing_cards = set(df.index.unique())

    cols = [
        "name_en",
        "element",
        "element_v2",
        "type_en",
        "cost",
        "power",
        "ex_burst",
        "multicard",
        "limit_break",
        "icons",
        "filename",
    ]
    mdf: pd.DataFrame = imagine_database(clear_extras=True).set_index("code")[cols]
    mdf = mdf[~mdf.index.duplicated(keep="first")]

    df = df.merge(mdf, on="code", how="left", sort=False)

    missing_cards = missing_cards - set(df.index.unique())
    if missing_cards:
        log.warning("missing cards:\n%s", missing_cards)

    df["uid"] = range(df.shape[0])
    df["path"] = df["uid"].apply(lambda x: str(DATA_DIR / "test" / f"{x}.jpg"))
    df.set_index("uid", inplace=True)
    if margs.data_filter:
        df.query(margs.data_filter, inplace=True)

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Cannot find '{MODEL_DIR}', skipping")

    for category in margs.models:
        label_fname = (MODEL_DIR / f"{category}.json").resolve()
        if not label_fname.exists():
            log.warning("Cannot find %s, skipping...", label_fname)
            continue

        with open(label_fname) as fp:
            labels = json.load(fp)

        if margs.label_mode == "category":
            if not pd.api.types.is_categorical_dtype(df[category]):
                df[category] = pd.Categorical(df[category], categories=labels)
        elif margs.label_mode == "multilabel":
            from crystalvision.models.multioutput import MultiLabel

            mdf = MultiLabel(mdf, df, reuse_labels=False)
            assert np.equal(
                labels, mdf.labels
            ).all(), "Save labels and current labels dont match"
            df[category] = [tuple(row) for row in mdf.vdf_codes]
        else:
            raise RuntimeError(f"Unknown label_mode: {margs.label_mode}")

        del mdf

        ds = paths_and_labels_to_dataset(
            image_paths=df["path"].tolist(),
            image_size=(250, 179),
            num_channels=3,
            labels=df[category],
            label_mode="categorical",  # pylint: disable=E1101
            num_classes=len(labels),
            data_format="channels_last",
            interpolation=margs.interpolation,
        )

        ensemble_fp = MODEL_DIR / f"{category}_ensemble.keras"
        if not ensemble_fp.exists() or ensemble_fp.stat().st_size == 0:
            log.warning("Creating ensemble: %s", ensemble_fp)

            models = [
                load_model(mpath) for mpath in MODEL_DIR.glob(f"{category}_*.keras")
            ]
            assert len(models) > 1, f"Not enough models found ({category})"

            voting = MyEnsembleVoteClassifier(
                models,
                labels=labels,
                weights=margs.init_weights,
                voting=margs.voting,
                activation=hard_activation if margs.hard_activation else None,
                activation_kwargs={"threshold": 0.0} if margs.hard_activation else None,
            )

            # dtype = ""
            # if category in ("ex_burst", "multicard", "limit_break", "mono"):
            #     dtype = "uint8"

            # x = voting.predict(ds, dtype=dtype)
            # print(voting.find_activation(IMAGES, df[category]))
            pre_score, models_predict, y_hat = voting.scores(
                ds.batch(256), ds.labels, with_dataframe=True, with_Y=True
            )
            log.info("Prediction Scores:\n%s", pre_score)
            log.info("Prediction DF:\n%s", models_predict)

            df[f"{category}_yhat"] = pd.Categorical(y_hat, categories=labels)

            mismatches = ~(df[category] == df[f"{category}_yhat"])
            mismatches = df[mismatches]
            log.info("Mismatches DF:\n%s", mismatches[[category, f"{category}_yhat"]])

            disagreements = ~models_predict.apply(
                lambda row: row.nunique() == 1, axis=1
            )
            disagreements = models_predict[disagreements]
            log.info("Disagreements DF:\n%s", disagreements)

            if not disagreements.empty and margs.vote_minimize:
                dis_df = df.iloc[disagreements.index]
                log.info("Disagreements:\n%s", dis_df[[category, f"{category}_yhat"]])

                X_train, X_test, y_train, y_test = train_test_split(
                    df["path"],  # dis_df["path"],
                    df[category],  # dis_df[category],
                    test_size=margs.test_size,
                    stratify=df[margs.stratify] if margs.stratify is not None else None,
                    random_state=23,
                )

                X_train = paths_and_labels_to_dataset(
                    image_paths=X_train.tolist(),
                    image_size=(250, 179),
                    num_channels=3,
                    labels=y_train,
                    label_mode="categorical",  # pylint: disable=E1101
                    num_classes=len(labels),
                    data_format="channels_last",
                    interpolation=margs.interpolation,
                )

                X_test = paths_and_labels_to_dataset(
                    image_paths=X_test.tolist(),
                    image_size=(250, 179),
                    num_channels=3,
                    labels=y_test,
                    label_mode="categorical",  # pylint: disable=E1101
                    num_classes=len(labels),
                    data_format="channels_last",
                    interpolation=margs.interpolation,
                )

                dis_ds = paths_and_labels_to_dataset(
                    image_paths=dis_df["path"].tolist(),
                    image_size=(250, 179),
                    num_channels=3,
                    labels=dis_df[category],
                    label_mode="categorical",  # pylint: disable=E1101
                    num_classes=len(labels),
                    data_format="channels_last",
                    interpolation=margs.interpolation,
                )

                if margs.use_disagreement:
                    pre_score = voting.scores(dis_ds.batch(256), dis_ds.labels)
                pre_weights = voting.weights.cpu().numpy()
                log.debug(
                    voting.find_weights(
                        X_train.batch(256), X_test.batch(256), y_train, y_test
                    )
                )

                if margs.use_disagreement:
                    scores = voting.scores(dis_ds.batch(256), dis_ds.labels)
                else:
                    scores = voting.scores(ds.batch(256), ds.labels)
                print(pre_score)
                print(pre_weights)
                print(scores)
                print(voting.weights)

            log.warning("Saving Model: %s", ensemble_fp)
            voting.save_model(ensemble_fp)

        ensemble_model = load_model(ensemble_fp)
        log.debug(ensemble_model.summary())

        yhat = ensemble_model.predict(ds.batch(256))
        log.debug("y_hat: %s", yhat)

        df[f"{category}_yhat"] = pd.Categorical.from_codes(yhat, categories=labels)
        log.debug("Y vs. y_hat:\n%s", df[[category, f"{category}_yhat"]])

        # data = df[[category, f"{category}_yhat"]].groupby(category)
        # print(data.min(), data.max())
        # data = data.sample(sample_size)
        # print(data)
        # threshold = find_threshold(data[f"{category}_yhat"], data[category], labels)
        # print(threshold)
        # # x = [labels[np.argmax(y)] for y in x]
        # x = [labels[y[0]] for y in hard_activation(x, threshold=threshold)]

        # df[f"{category}_yhat"] = x
        # # if len(labels) <= 2: df[f"{category}_yhat"] = df[f"{category}_yhat"].astype('UInt8')

    return df


def main(margs) -> None:
    """Test the various models."""
    df = create_models(margs).reset_index()

    for category in margs.models:
        key = f"{category}_yhat"
        if key not in df:
            continue
        comp = df[category] == df[key]
        comp = comp.value_counts(normalize=True)

        log.info("%s accuracy: %.2f%%", category, comp.get(True, 0.0) * 100)

    # df.sort_index(axis=1, inplace=True)
    # print(df)


if __name__ == "__main__":
    import yaml
    import argparse
    from pathlib import Path
    from keras import backend
    from crystalvision.models import SRC_DIR

    if backend.backend() == "torch":
        from torchvision.transforms.functional import InterpolationMode

    parser = argparse.ArgumentParser(description="Model tuning command-line tool")
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        help="The ensemble YAML file",
        default=SRC_DIR / "models" / "ensemble.yml",
    )
    parser.add_argument(
        "--models", "-m", type=str, nargs="+", default=[], help="The models"
    )
    parser.add_argument(
        "--interpolation",
        "-i",
        type=InterpolationMode,
        default=InterpolationMode.NEAREST_EXACT,
        choices=list(InterpolationMode),
        help="An up/downsampling method.",
    )
    parser.add_argument("--vote-minimize", action="store_true")
    parser.add_argument("--voting", type=str, default="hard", help="The voting type")
    parser.add_argument(
        "--hard-activation", action="store_true", help="Use hard_activation"
    )
    parser.add_argument(
        "--data-filter",
        default="full_art != 1 and focal == 1",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose mode"
    )
    parser.add_argument("--label-mode", default="category")
    parser.add_argument(
        "--stratify",
        nargs="+",
    )
    parser.add_argument("--test-size", default=0.25, type=float)
    parser.add_argument("--init-weights", type=float, nargs="+")
    parser.add_argument("--use-disagreement", action="store_true")

    args = parser.parse_args()
    args.config = args.config.resolve()

    if not args.config.exists():
        raise FileNotFoundError(str(args.config.resolve()))

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s [%(funcName)s] %(message)s",
        datefmt="%d/%b/%Y %H:%M:%S",
        encoding="utf-8",
    )

    with args.config.open("r") as fp:
        for key, value in yaml.safe_load(fp).items():
            if getattr(args, key) in (None, parser.get_default(key)):
                setattr(args, key, value)

    log.info(args)

    main(args)

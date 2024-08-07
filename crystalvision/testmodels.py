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
import os
from glob import iglob
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from PIL import Image
from skimage.io import imread
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
from scipy.optimize._minimize import MINIMIZE_METHODS

from crystalvision.data.base import DATA_DIR
from crystalvision.data.dataset import make_database
from crystalvision.models.base import MODEL_DIR
from crystalvision.models.ensemble import hard_activation, MyEnsembleVoteClassifier


CATEGORIES: Tuple[str] = (
    "name_en",
    "element",
    "type_en",
    "cost",
    "power",
    "ex_burst",
    "multicard",
    "limit_break",
    "mono",
)
IMAGE_DF: pd.DataFrame = pd.read_json(
    os.path.join(os.path.dirname(__file__), "testmodels.json")
)
# print(IMAGE_DF)


def load_image(url: str, img_fname: str = "") -> np.ndarray:
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

    dst = os.path.join(DATA_DIR, "test")
    if not os.path.exists(dst):
        os.makedirs(dst)

    dst = os.path.join(dst, img_fname)
    if os.path.exists(dst):
        data = imread(dst)[:, :, :3]
        return data * (1.0 / 255)

    if url.startswith("blob:"):
        url = url[5:]
    data = imread(url)
    if img_fname.endswith(".jpg"):
        Image.fromarray(data).convert("RGB").save(dst)
    else:
        Image.fromarray(data).save(dst)

    return data[:, :, :3] * (1.0 / 255)


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


def test_models() -> pd.DataFrame:
    """
    Run all models and apply values in the dataframe.

    Returns:
        ImageData dataframe with yhat(s)
    """
    df: pd.DataFrame = IMAGE_DF.copy().set_index("code")
    cols = [
        "name_en",
        "element",
        "type_en",
        "cost",
        "power",
        "ex_burst",
        "multicard",
        "limit_break",
    ]
    mdf: pd.DataFrame = make_database().set_index("code")[cols]
    df = df.merge(mdf, on="code", how="left", sort=False)
    # df['ex_burst'] = df['ex_burst'].astype('uint8')
    # df['multicard'] = df['multicard'].astype('uint8')
    # df["mono"] = df["element"].apply(lambda i: len(i) == 1 if i else True).astype(bool)

    df["uid"] = range(df.shape[0])
    df["images"] = df.apply(
        lambda x: tf.image.resize(load_image(x["uri"], f"{x['uid']}.jpg"), (250, 179)),
        axis=1,
    )

    df.query("full_art != 1 and focal == 1", inplace=True)
    IMAGES = np.array(df.pop("images").tolist())

    df.drop(["uid", "uri"], axis=1, inplace=True)
    print(df)

    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"Cannot find '{MODEL_DIR}', skipping")

    for category in CATEGORIES:
        label_fname = os.path.join(MODEL_DIR, f"{category}.json")
        if not os.path.exists(label_fname):
            print(f"Cannot find {label_fname}, skipping...")
            continue

        with open(label_fname) as fp:
            labels = json.load(fp)

        models = [
            load_model(mpath)
            for mpath in iglob(str(MODEL_DIR) + os.sep + f"{category}_*.h5")
            if "_model" not in mpath
        ]
        # ensemble_path = os.path.join(MODEL_DIR, f"{category}_ensemble.h5")
        if len(models) > 1:
            # if os.path.exists(ensemble_path):
            #   model = tf.keras.models.load_model(ensemble_path)
            #   x = model(IMAGES, training=False)
            #   if category != "Ex_Burst":
            #       x = [labels[y] for y in x]
            # else:
            if category in ("ex_burst", "multicard", "mono"):
                voting = MyEnsembleVoteClassifier(
                    models,
                    labels=labels,
                    activation=hard_activation,
                    activation_kwargs={"threshold": 0.0},
                )
                x = voting.predict(IMAGES, "uint8")
                print(voting.find_activation(IMAGES, df[category]))
            else:
                voting = MyEnsembleVoteClassifier(models, labels=labels)
                x = voting.predict(IMAGES)
            # print(voting._predict(IMAGES))
            # print(df[category])
            # labels = df[category].apply(lambda x: labels[x])
            # print(voting._predict(IMAGES))
            # print(voting.find_weights(IMAGES, df[category]))
            scores = voting.scores(IMAGES, df[category])
            print(scores)
            # voting.save_model(ensemble_path)
            x = x.to_numpy()
        else:
            # print(models[0].summary())
            # print(models[0](IMAGES, training=False))
            x = models[0].predict(IMAGES)
            # print(x)
            df[f"{category}_yhat"] = x
            sample_size = min(df[category].value_counts())
            data = df[[category, f"{category}_yhat"]].groupby(category)
            print(data.min(), data.max())
            data = data.sample(sample_size)
            print(data)
            threshold = find_threshold(data[f"{category}_yhat"], data[category], labels)
            print(threshold)
            # x = [labels[np.argmax(y)] for y in x]
            x = [labels[y[0]] for y in hard_activation(x, threshold=threshold)]

        df[f"{category}_yhat"] = x
        # if len(labels) <= 2: df[f"{category}_yhat"] = df[f"{category}_yhat"].astype('UInt8')

    return df


def main() -> None:
    """Test the various models."""
    df = test_models().reset_index()

    for category in CATEGORIES:
        key = f"{category}_yhat"
        if key not in df:
            continue
        comp = df[category] == df[key]
        comp = comp.value_counts(normalize=True)
        # print(comp)

        print(f"{category} accuracy: {comp.get(True, 0.0) * 100}%")
        # print(xf)

    df.sort_index(axis=1, inplace=True)
    print(df)


if __name__ == "__main__":
    main()

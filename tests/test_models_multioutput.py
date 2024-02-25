import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import cv2


@pytest.fixture(scope="module")
def database():
    from crystalvision.data.dataset import (
        imagine_database,
    )

    df = imagine_database("images")
    df.query("filename.str.contains('_eg')", inplace=True)  # English
    # df = df[df['element_v2'].apply(lambda x: isinstance(x, tuple) and len(x) > 1)]

    return df


@pytest.fixture(scope="module")
def cards(database):
    return database.sample(n=10)


@pytest.fixture(scope="module")
def images(cards):
    return [cv2.imread(card.filename) for _, card in cards.iterrows()]


@pytest.fixture(scope="module")
def blobs(images):
    blob = []
    for img in images:
        img_blob = cv2.dnn.blobFromImage(
            img,
            scalefactor=1 / 255.0,
            size=(179, 250),
            mean=(0, 0, 0),
            swapRB=False,
            crop=False,
        )

        img_blob = np.transpose(img_blob, (0, 2, 3, 1))
        blob.append(img_blob)

    return np.concatenate(blob)


@pytest.fixture(scope="module")
def card(database):
    return database.sample(n=1).iloc[0]


@pytest.fixture(scope="module")
def image(cards):
    return cv2.imread(card.filename)


@pytest.fixture(scope="module")
def labels():
    label_file = Path("./data/model/multilabel.json")
    with label_file.open("r") as lbfp:
        return json.load(lbfp)


@pytest.fixture(scope="module")
def mlb(labels):
    from crystalvision.models import MODEL_DIR

    mlb_file = Path(MODEL_DIR / "multilabel" / "multilabel_mlb.pkl")
    with mlb_file.open("rb") as f:
        return pickle.load(f)[0]


# Test to check specific ONNX models
@pytest.mark.parametrize(
    "key, start, end",
    [
        ("type_en", 0, 4),
        ("cost", 4, 15),
        ("power", 23, 34),
        ("icons", 34, 37),
        ("element_v2", 15, 23),
    ],
)
@pytest.mark.parametrize(
    "model_file",
    [
        "multilabel_1.onnx",
        "multilabel_2.onnx",
        "multilabel_3.onnx",
        "multilabel_4.onnx",
        "multilabel_5.onnx",
        "multilabel_6.onnx",
    ],
)
def test_specific_onnx_model(key, start, end, model_file, cards, blobs, mlb):
    model_path = Path("data/model") / model_file
    if not model_path.exists():
        pytest.skip(f"{model_path} does not exist.")

    net = cv2.dnn.readNetFromONNX(str(model_path))
    assert not net.empty(), f"{model_path} is not a valid ONNX model."

    # Set the input
    net.setInput(blobs)

    # Forward pass
    output = pd.DataFrame(net.forward(), columns=mlb.classes).map(
        lambda x: 1.0 if x >= 0.95 else 0.0
    )
    assert output.shape == (cards.shape[0], len(mlb.classes))

    print(mlb.inverse_transform(output.to_numpy()))

    df = output.iloc[:, start:end]

    if key == "icons":
        y_true = list(icons[0] for icons in cards[key].tolist())
    elif key == "element_v2":
        y_true = list(set(card["element_v2"]) for _, card in cards.iterrows())
    else:
        y_true = cards[key].tolist()

    if key == "element_v2":
        y_pred = list(
            set(card.index[card == 1.0].tolist()) for _, card in df.iterrows()
        )
    else:
        y_pred = df.idxmax(axis=1).tolist()

    print(y_true)
    print(y_pred)
    assert y_true == y_pred, df

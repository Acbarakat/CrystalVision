import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import cv2


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


@pytest.fixture(scope="module")
def database():
    from crystalvision.data.dataset import (
        imagine_database,
    )

    df = imagine_database("images")
    df.query("filename.str.contains('_eg')", inplace=True)  # English
    # df = df[df['element'].apply(lambda x: isinstance(x, tuple) and len(x) > 1)]

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
            swapRB=True,
            crop=False,
        )

        img_blob = np.transpose(img_blob, (0, 2, 3, 1))
        blob.append(img_blob)

    return np.concatenate(blob)


@pytest.fixture(scope="module")
def card(database):
    return database.sample(n=1).iloc[0]


@pytest.fixture(scope="module")
def image(card):
    return cv2.imread(card.filename)


@pytest.fixture(scope="module")
def labels():
    label_file = Path("./data/model/element.json")
    with label_file.open("r") as lbfp:
        return json.load(lbfp)


# Test to check specific ONNX models
@pytest.mark.parametrize(
    "model_file",
    [
        "element_1.onnx",
        "element_2.onnx",
        "element_3.onnx",
        "element_4.onnx",
        "element_5.onnx",
        "element_6.onnx",
    ],
)
def test_specific_onnx_model(model_file, cards, blobs, labels):
    model_path = Path("data/model") / model_file
    if not model_path.exists():
        pytest.skip(f"{model_path} does not exist.")

    net = cv2.dnn.readNetFromONNX(str(model_path))
    assert not net.empty(), f"{model_path} is not a valid ONNX model."

    # Set the input
    net.setInput(blobs)

    # Forward pass
    output = pd.DataFrame(net.forward(), columns=labels)
    print(output)

    y_pred = output.idxmax(axis=1).tolist()
    print("y_pred", y_pred)
    assert output.shape[1] == len(labels)

    y_true = cards["element"].tolist()
    print("y_true", y_true)

    assert y_true == y_pred, output

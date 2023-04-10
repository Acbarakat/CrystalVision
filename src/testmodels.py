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

"""
import json
import os
from glob import iglob
from typing import Any, Callable

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.layers import Activation, Flatten, Input, Layer
from keras.models import Model, load_model
from keras.utils.generic_utils import get_custom_objects
from mlxtend.classifier import EnsembleVoteClassifier
from PIL import Image
from skimage.io import imread
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

from gatherdata import DATA_DIR
from generatemodels import make_database

CATEGORIES: tuple = (
    "Name_EN", "Element", "Type_EN", "Cost", "Power", "Ex_Burst"
)
IMAGE_DF: pd.DataFrame = pd.read_json(os.path.join(os.path.dirname(__file__),
                                                   "testmodels.json"))


class MyEnsembleVoteClassifier(EnsembleVoteClassifier):
    def __init__(self,
                 clfs,
                 voting: str = "hard",
                 weights: Any | None = None,
                 verbose: int = 0,
                 use_clones: bool = True,
                 fit_base_estimators: bool = False,
                 activation: Callable | None = None,
                 activation_kwargs: dict | None = None):
        super().__init__(clfs,
                         voting,
                         weights,
                         verbose,
                         use_clones,
                         fit_base_estimators)
        self.clfs_ = clfs
        self.activation = tf.keras.activations.linear
        if activation is not None:
            self.activation = activation
        self.activation_kwargs = {}
        if activation_kwargs is not None:
            self.activation_kwargs = activation_kwargs

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Collect results from clf.predict calls."""
        if not self.fit_base_estimators:
            predictions = np.asarray([clf(X) for clf in self.clfs_]).T
            if predictions.shape[0] == 1:
                return self.activation(predictions[0],
                                       **self.activation_kwargs)

            predictions = np.array([
                np.argmax(p, axis=1) for p in predictions.T
            ])
            return self.activation(predictions.T, **self.activation_kwargs)

        return np.asarray(
            [self.le_.transform(clf(X)) for clf in self.clfs_]
        ).T

    def _predict_probas(self, X) -> np.ndarray:
        """Collect results from clf.predict_proba calls."""
        probas = [clf(X, training=False) for clf in self.clfs_]
        if self.weights:
            probas = [np.dot(c, w) for c, w in zip(probas, self.weights)]
        probas = np.sum(probas, axis=0) / len(self.clfs_)
        return normalize(probas, axis=1, norm='l1')

    def predict(self, X: np.ndarray, dtype: str = '') -> np.ndarray:
        """
        Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.

        """
        predictions = self.transform(X)
        if hasattr(predictions, "numpy"):
            predictions = predictions.numpy()
        if dtype:
            predictions = predictions.astype(dtype)

        if self.voting == "soft":
            maj = np.argmax(predictions, axis=1)

        else:  # 'hard' voting
            maj = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                axis=1,
                arr=predictions,
            )

        if self.fit_base_estimators:
            maj = self.le_.inverse_transform(maj)

        return maj

    def scores(self, X, y, sample_weight=None) -> pd.Series:
        """
        Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of ``self.predict(X)`` w.r.t. `y`.
        """
        result = self._predict(X)
        if hasattr(result, "numpy"):
            result = result.numpy()
        result = result.T

        result = np.array([
            accuracy_score(y, p, sample_weight=sample_weight) for p in result
        ] + [self.score(X, y)])

        return pd.Series(result,
                         index=[c.name for c in self.clfs_] + ['ensemble'],
                         name="accuracy")

    def generate_model(self) -> Model:
        """Converts/creates our ensemble as a single Model."""
        if self.voting == 'soft':
            raise NotImplementedError("Cannot generate a 'soft' voting model")

        name = self.clfs_[0].name.replace("_1", "_ensemble")
        model_input = Input(shape=self.clfs_[0].input_shape[1:])
        y_models = [model(model_input, training=False) for model in self.clfs_]

        active = self.activation(y_models, **self.activation_kwargs)
        if self.clfs_[0].output_shape[1] == 1:
            flatten = Flatten()(active)
            outputs = HardBinaryVote(vote_weights=self.weights)(flatten)
        else:
            outputs = HardClassVote(vote_weights=self.weights)(active)

        return Model(inputs=model_input, outputs=outputs, name=name)

    def save_model(self, file_path: str) -> None:
        """Save the model to Tensorflow SavedModel or a single HDF5 file."""
        return self.generate_model().save(file_path)


def load_image(url: str,
               img_fname: str = '') -> np.ndarray:
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
        return imread(dst)[:, :, :3]

    data = imread(url)
    if img_fname.endswith(".jpg"):
        Image.fromarray(data).convert("RGB").save(dst)
    else:
        Image.fromarray(data).save(dst)

    return data[:, :, :3]


IMAGES = IMAGE_DF.pop("URI")
IMAGES = [load_image(image, f"{i}.jpg") for i, image in enumerate(IMAGES)]
IMAGES = np.array([tf.image.resize(image, (250, 179)) for image in IMAGES])


class HardBinaryVote(Layer):
    def __init__(self,
                 trainable=False,
                 name="hard_vote",
                 dtype=None,
                 dynamic=False,
                 vote_weights: Any | None = None,
                 **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.vote_weights = None
        if vote_weights is not None:
            self.vote_weights = tf.convert_to_tensor(vote_weights,
                                                     name="votes")

    def call(self, inputs):
        inputs = K.transpose(inputs)

        return K.tf.map_fn(
            lambda z: K.cast(K.argmax(K.tf.math.bincount(z, weights=self.vote_weights)), 'int32'),  # noqa: E501
            inputs,
        )


class HardClassVote(HardBinaryVote):
    def call(self, inputs):
        inputs = K.transpose(K.cast(K.argmax(inputs), 'int32'))

        return K.tf.map_fn(
            lambda z: K.cast(K.argmax(K.tf.math.bincount(z, weights=self.vote_weights)), 'int32'),  # noqa: E501
            inputs,
        )


@tf.__internal__.dispatch.add_dispatch_support
def hard_activation(z: Any, threshold: float = 0.5) -> tf.Tensor:
    return K.cast(K.greater_equal(z, threshold), tf.dtypes.int32)


get_custom_objects().update({
    'hard_activation': Activation(hard_activation, name="hard_activaiton"),
})


def test_models() -> pd.DataFrame:
    """
    Run all models and apply values in the dataframe.

    Returns:
        ImageData dataframe with yhat(s)
    """
    df: pd.DataFrame = IMAGE_DF.copy().set_index('Code')
    cols = ["Name_EN", "Element", "Type_EN", "Cost", "Power", "Ex_Burst"]
    mdf: pd.DataFrame = make_database().set_index('Code')[cols]
    df = df.merge(mdf, on="Code", how='left', sort=False)
    df['Ex_Burst'] = df['Ex_Burst'].astype('uint8')

    for category in CATEGORIES:
        model_path = os.path.join(DATA_DIR, "model", f"{category}_model")

        with open(os.path.join(model_path, "category.json")) as fp:
            labels = json.load(fp)

        models = [
            load_model(model_path) for model_path in iglob(model_path + "*")
        ]
        ensemble_path = os.path.join(DATA_DIR, "model", f"{category}_ensemble")
        if category in ("Ex_Burst", "Cost"):
            # if os.path.exists(ensemble_path):
            #   model = tf.keras.models.load_model(ensemble_path)
            #   x = model(IMAGES, training=False)
            #   if category != "Ex_Burst":
            #       x = [labels[y] for y in x]
            # else:
            if category == "Ex_Burst":
                voting = MyEnsembleVoteClassifier(models,
                                                  weights=[1, 1, 3, 1, 1],
                                                  activation=hard_activation,
                                                  activation_kwargs={
                                                      'threshold': 0.1
                                                  })
                x = voting.predict(IMAGES, 'uint8')
            else:
                voting = MyEnsembleVoteClassifier(models)
                x = [labels[y] for y in voting.predict(IMAGES)]
            label_indexes = df[category].apply(lambda x: labels.index(x))
            scores = voting.scores(IMAGES, label_indexes)
            print(scores)
            voting.save_model(ensemble_path)
        else:
            x = [
                labels[np.argmax(y)] for y in models[0](IMAGES, training=False)
            ]

        df[f"{category}_yhat"] = x
        if len(labels) <= 2:
            df[f"{category}_yhat"] = df[f"{category}_yhat"].astype('UInt8')

    return df


def main() -> None:
    df = test_models().reset_index()

    # Remove the ones we know wont work without:
    # - object detection
    # - full art enablement
    df.drop([2, 3, 4, 5, 17, 19, 26, 27, 32], inplace=True)

    for category in CATEGORIES:
        comp = df[category] == df[f"{category}_yhat"]
        comp = comp.value_counts(normalize=True)

        print(f"{category} accuracy: {comp[True] * 100}%%")
        # print(xf)

    df.sort_index(axis=1, inplace=True)
    print(df)


if __name__ == '__main__':
    main()

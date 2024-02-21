# -*- coding: utf-8 -*-
"""
Element-based Card Hypermodels.

Todo:
    * N/A

"""
from typing import List

from pandas import DataFrame
from keras import layers, models, optimizers, callbacks, backend
from keras_tuner import HyperParameters
from sklearn.preprocessing import MultiLabelBinarizer

try:
    from . import CardModel
    from .mixins.compiles import (
        BinaryMixin,
        CategoricalMixin,
        OneHotMeanIoUMixin,
    )
    from .mixins.tuners import (
        BayesianOptimizationTunerMixin,
        RandomSearchTunerMixin,
    )
    from .ext.metrics import MyOneHotMeanIoU
    from .ext.callbacks import StopOnValue
except ImportError:
    from crystalvision.models import CardModel
    from crystalvision.models.mixins.compiles import (
        BinaryMixin,
        CategoricalMixin,
        OneHotMeanIoUMixin,
    )
    from crystalvision.models.mixins.tuners import (
        BayesianOptimizationTunerMixin,
        RandomSearchTunerMixin,
    )
    from crystalvision.models.ext.metrics import MyOneHotMeanIoU
    from crystalvision.models.ext.callbacks import StopOnValue


class Mono(BinaryMixin, RandomSearchTunerMixin, CardModel):
    def __init__(self, df: DataFrame, vdf: DataFrame) -> None:
        super().__init__(df, vdf, "mono", name="mono")

        self.stratify_cols.extend(["type_en"])

    def build(self, hp: HyperParameters, seed: int | None = None) -> models.Sequential:
        """
        Build a model.

        Args:
            hp (HyperParameters): A `HyperParameters` instance.

        Returns:
            A model instance.
        """
        m = models.Sequential(
            layers=[
                layers.Input(shape=self.IMAGE_SHAPE),
                layers.Conv2D(
                    32,
                    kernel_size=(3, 3),
                    activation="relu",
                ),
                layers.MaxPooling2D(),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(),
                layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(),
                layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dense(
                    hp.Int("dense_units", min_value=32, max_value=512, step=32),
                    activation="relu",
                ),
                layers.Dense(1, activation="sigmoid"),
            ],
            name=self.name,
        )

        learning_rate = hp.Choice("learning_rate", values=[0.01, 0.001, 0.0001])
        optimizer = hp.Choice("optimizer", values=["adam", "rmsprop", "sgd"])
        # learning_rate = hp.Float('learning_rate',
        #                          min_value=1e-4,
        #                          max_value=1e-2,
        #                          sampling='LOG')

        if optimizer == "adam":
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == "rmsprop":
            optimizer = optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = optimizers.SGD(learning_rate=learning_rate)

        m.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
        return m


class Element(CategoricalMixin, BayesianOptimizationTunerMixin, CardModel):
    def __init__(self, df: DataFrame, vdf: DataFrame) -> None:
        super().__init__(df, vdf, "element", name="element")

        self.stratify_cols.extend(["type_en"])

    def build(self, hp: HyperParameters, seed: int | None = None) -> models.Sequential:
        """
        Build a model.

        Args:
            hp (HyperParameters): A `HyperParameters` instance.

        Returns:
            A model instance.
        """

        m = models.Sequential(
            layers=[
                layers.Input(shape=self.IMAGE_SHAPE),
                layers.Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                ),
                layers.AveragePooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.AveragePooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.AveragePooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.Flatten(),
                layers.Dropout(0.2, seed=seed),
                # layers.Dense(hp.Int('dense_units', min_value=128, max_value=512, step=32), activation='relu'),
                layers.Dense(
                    hp.Int("dense_units", min_value=64, max_value=1024, step=32),
                    activation="relu",
                ),
                # layers.Dense(480, activation='relu'),
                layers.Dense(len(self.labels), activation="softmax"),
            ],
            name=self.name,
        )

        learning_rate = hp.Float(
            "learning_rate",
            # min_value=3.1e-4,
            min_value=1.1e-4,
            # max_value=3.5e-4,
            max_value=5.5e-4,
            sampling="LOG",
        )

        m.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss=self.loss,
            metrics=self.metrics,
        )
        return m


class ElementV2(OneHotMeanIoUMixin, BayesianOptimizationTunerMixin, CardModel):
    def __init__(self, df: DataFrame, vdf: DataFrame) -> None:
        self.name = "element_v2"
        self.tunable = True

        self.df: DataFrame = self.filter_dataframe(df.copy())
        self.vdf: DataFrame = self.filter_dataframe(vdf.copy())

        self.feature_key: str = "element_v2"
        self.stratify_cols: List[str] = ["element", "type_en"]

        self.labels: List[str] = [
            elem[0] for elem in self.df["element_v2"].unique() if len(elem) == 1
        ]

        self.mlb = MultiLabelBinarizer(classes=self.labels)

        self.df_codes = self.mlb.fit_transform(self.df["element_v2"])
        self.vdf_codes = self.mlb.transform(self.vdf["element_v2"])

        self.callbacks = [
            callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0.005,
                patience=5,
                restore_best_weights=True,
            ),
            StopOnValue(),
        ]

    def build(self, hp: HyperParameters, seed: int | None = None) -> models.Sequential:
        """
        Build a model.

        Args:
            hp (HyperParameters): A `HyperParameters` instance.

        Returns:
            A model instance.
        """
        batch_size = hp.Choice("batch_size", values=[32, 64, 128, 256, 512])  # noqa

        m = models.Sequential(
            layers=[
                layers.Input(shape=self.IMAGE_SHAPE, batch_size=batch_size),
                layers.Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                ),
                layers.AveragePooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.AveragePooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.AveragePooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.Flatten(),
                # layers.Dropout(0.2, seed=seed),
                layers.Dense(
                    hp.Int("dense_units", min_value=64, max_value=1024, step=32),
                    activation="relu",
                ),
                layers.Dense(len(self.labels), activation="sigmoid", name="result"),
            ],
            name=self.name,
        )

        learning_rate = hp.Float(
            "learning_rate",
            # min_value=3.1e-4,
            min_value=1.1e-4,
            # max_value=3.5e-4,
            max_value=5.5e-4,
            sampling="LOG",
        )

        m.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss=self.loss,
            metrics=[
                MyOneHotMeanIoU(
                    num_classes=len(self.labels),
                    threshold=0.95,
                    name="accuracy",
                    sparse_y_pred=backend.backend() != "torch",
                )
            ],
        )
        return m


if __name__ == "__main__":
    from crystalvision.models import tune_model

    # tune_model(Element)
    # tune_model(Mono)
    tune_model(ElementV2)

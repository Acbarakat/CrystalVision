# -*- coding: utf-8 -*-
"""
Element-based Card Hypermodels.

Todo:
    * N/A

"""
from pandas import DataFrame
from keras import layers, models, optimizers, backend
from keras_tuner import HyperParameters

try:
    from .base import CardModel, MultiLabelCardModel
    from .mixins.compiles import (
        BinaryMixin,
        CategoricalMixin,
    )
    from .mixins.tuners import (
        BayesianOptimizationTunerMixin,
        RandomSearchTunerMixin,
    )
    from .ext.metrics import MyOneHotMeanIoU
except ImportError:
    from crystalvision.models.base import CardModel, MultiLabelCardModel
    from crystalvision.models.mixins.compiles import (
        BinaryMixin,
        CategoricalMixin,
    )
    from crystalvision.models.mixins.tuners import (
        BayesianOptimizationTunerMixin,
        RandomSearchTunerMixin,
    )
    from crystalvision.models.ext.metrics import MyOneHotMeanIoU


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


class ElementV2(BayesianOptimizationTunerMixin, MultiLabelCardModel):
    """Multilabel protoype for Card's Element."""

    def __init__(self, df: DataFrame, vdf: DataFrame) -> None:
        super().__init__("element_v2", df, vdf, "element_v2", ["element", "type_en"])

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

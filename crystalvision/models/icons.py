# -*- coding: utf-8 -*-
"""
Element-based Card Hypermodels.

Todo:
    * N/A

"""
from pandas import DataFrame
from keras import layers, models, optimizers
from keras_tuner import HyperParameters

try:
    from .base import CardModel, MultiLabelCardModel
    from .mixins.compiles import (
        BinaryMixin,
    )
    from .mixins.tuners import RandomSearchTunerMixin, BayesianOptimizationTunerMixin
    from .ext.metrics import MyOneHotMeanIoU
except ImportError:
    from crystalvision.models.base import CardModel, MultiLabelCardModel
    from crystalvision.models.mixins.compiles import (
        BinaryMixin,
    )
    from crystalvision.models.mixins.tuners import (
        RandomSearchTunerMixin,
        BayesianOptimizationTunerMixin,
    )
    from crystalvision.models.ext.metrics import MyOneHotMeanIoU


class Exburst(BinaryMixin, RandomSearchTunerMixin, CardModel):
    def __init__(self, df: DataFrame, vdf: DataFrame) -> None:
        super().__init__(df, vdf, "ex_burst", name="exburst")

        self.stratify_cols.extend(["element", "type_en"])

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


class Multicard(BinaryMixin, RandomSearchTunerMixin, CardModel):
    def __init__(self, df: DataFrame, vdf: DataFrame) -> None:
        super().__init__(df, vdf, "multicard", name="multicard")

        self.stratify_cols.extend(["element", "type_en"])

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


class Icons(BayesianOptimizationTunerMixin, MultiLabelCardModel):
    """Multilabel protoype for Card's Icons."""

    MAX_TRIALS: int = 50

    def __init__(self, df: DataFrame, vdf: DataFrame) -> None:
        super().__init__(
            "icons",
            df,
            vdf,
            "icons",
            ["ex_burst", "multicard", "element", "type_en"],
            generate_metrics=False,
        )

        # TODO: Resolve the error when there is only one target_class_ids

    def build(self, hp: HyperParameters, seed: int | None = None) -> models.Sequential:
        """
        Build a model.

        Args:
            hp (HyperParameters): A `HyperParameters` instance.

        Returns:
            A model instance.
        """
        batch_size = hp.Choice("batch_size", values=[16, 32, 64, 128, 256, 512])  # noqa

        pl1 = self._pooling2d_choice("pooling1", hp)[1]
        pl2 = self._pooling2d_choice("pooling2", hp)[1]
        pl3 = self._pooling2d_choice("pooling3", hp)[1]
        pl4 = self._pooling2d_choice("pooling4", hp)[1]

        m = models.Sequential(
            layers=[
                layers.Input(shape=self.IMAGE_SHAPE, batch_size=batch_size),
                layers.Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                ),
                pl1((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                pl2((2, 2)),
                layers.Conv2D(128, (3, 3), activation="relu"),
                pl3((2, 2)),
                layers.Conv2D(128, (3, 3), activation="relu"),
                pl4((2, 2)),
                layers.Flatten(),
                # layers.Dropout(0.2, seed=seed),
                layers.Dense(
                    hp.Int("dense_units", min_value=32, max_value=1024, step=32),
                    activation="relu",
                ),
                layers.Dense(len(self.labels), activation="sigmoid"),
            ],
            name=self.name,
        )

        optimizer = self._optimizer_choice("optimizer", hp)[1]

        m.compile(
            optimizer=optimizer(),
            loss=self.loss,
            metrics=self._metrics
            + [
                MyOneHotMeanIoU(
                    num_classes=len(self.labels), threshold=0.95, name="accuracy"
                ),
            ],
        )
        return m


if __name__ == "__main__":
    from crystalvision.models import tune_model

    # tune_model(Exburst)
    # tune_model(Multicard)
    tune_model(Icons, num=Icons.MAX_TRIALS, save_models=False)

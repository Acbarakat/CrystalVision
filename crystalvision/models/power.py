# -*- coding: utf-8 -*-
"""
Type-based Card Hypermodels.

Todo:
    * N/A

"""
from pandas import DataFrame
from keras import layers, models, optimizers
from keras_tuner import HyperParameters

try:
    from . import CardModel
    from .mixins.compiles import CategoricalMixin
    from .mixins.tuners import BayesianOptimizationTunerMixin
except ImportError:
    from __init__ import CardModel
    from crystalvision.models.mixins.compiles import CategoricalMixin
    from crystalvision.models.mixins.tuners import BayesianOptimizationTunerMixin


class Power(CategoricalMixin, BayesianOptimizationTunerMixin, CardModel):
    def __init__(self, df: DataFrame, vdf: DataFrame) -> None:
        super().__init__(df, vdf, "power", name="power")

        self.stratify_cols.extend(["element", "type_en"])

    def build(self, hp: HyperParameters, seed: int | None = None) -> models.Sequential:
        """
        Build a model.

        Args:
            hp (HyperParameters): A `HyperParameters` instance.

        Returns:
            A model instance.
        """
        batch_size = hp.Choice("batch_size", values=[64, 128, 256])  # noqa

        m = models.Sequential(
            layers=[
                layers.Input(shape=self.IMAGE_SHAPE, batch_size=batch_size),
                layers.Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                ),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dropout(0.2, seed=seed),
                layers.Dense(
                    hp.Int("dense_units", min_value=128, max_value=512, step=128),
                    activation="relu",
                ),
                layers.Dense(len(self.labels), activation="softmax"),
            ],
            name=self.name,
        )

        learning_rate = hp.Float(
            "learning_rate", min_value=1.0e-5, max_value=1.0e-2, sampling="LOG"
        )

        optimizer = optimizers.Adam(learning_rate=learning_rate, amsgrad=True)

        m.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
        return m


if __name__ == "__main__":
    from crystalvision.models import tune_model

    tune_model(Power)

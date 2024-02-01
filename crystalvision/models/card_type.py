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


class CardTyping(CategoricalMixin, BayesianOptimizationTunerMixin, CardModel):
    def __init__(self, df: DataFrame, vdf: DataFrame) -> None:
        super().__init__(df, vdf, "type_en", name="type_en")

        self.stratify_cols.extend(["element"])

    @staticmethod
    def filter_dataframe(df: DataFrame) -> DataFrame:
        """
        Filter out data from test/train/validation dataframe.

        Args:
            df (DataFrame): Data to be filtered

        Returns:
            The DataFrame
        """
        # Ignore by language
        # df.query("~filename.str.contains('_eg')", inplace=True)  # English
        df.query("~filename.str.contains('_fr')", inplace=True)  # French
        df.query("~filename.str.contains('_es')", inplace=True)  # Spanish
        df.query("~filename.str.contains('_it')", inplace=True)  # Italian
        df.query("~filename.str.contains('_de')", inplace=True)  # German
        # df.query("~filename.str.contains('_jp')", inplace=True)  # Japanese

        return df

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

        m = models.Sequential(
            layers=[
                layers.Conv2D(
                    32,
                    (3, 3),
                    padding="same",
                    activation="relu",
                    input_shape=self.IMAGE_SHAPE,
                ),
                pl1(padding="same"),
                layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
                pl2(padding="same"),
                layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                layers.Dropout(0.2, seed=seed),
                layers.Flatten(),
                # layers.Dense(hp.Int('dense_units', min_value=128, max_value=512, step=128), activation='relu'),
                layers.Dense(128, activation="relu"),
                layers.Dense(len(self.labels), activation="softmax"),
            ],
            name=self.name,
        )

        optimizer = hp.Choice("optimizer", values=["adam", "rmsprop"])

        learning_rate = hp.Float(
            "learning_rate", min_value=1.0e-4, max_value=1.0e-2, sampling="LOG"
        )

        if optimizer == "adam":
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == "rmsprop":
            optimizer = optimizers.RMSprop(learning_rate=learning_rate)

        m.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
        return m


if __name__ == "__main__":
    from crystalvision.models import tune_model

    tune_model(CardTyping)

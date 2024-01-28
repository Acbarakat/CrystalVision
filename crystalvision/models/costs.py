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
    from . import CardModel, Identity, MinPooling2D
    from .mixins.compiles import CategoricalMixin
    from .mixins.tuners import BayesianOptimizationTunerMixin
except ImportError:
    from __init__ import CardModel, Identity, MinPooling2D
    from crystalvision.models.mixins.compiles import CategoricalMixin
    from crystalvision.models.mixins.tuners import BayesianOptimizationTunerMixin


class Cost(CategoricalMixin, BayesianOptimizationTunerMixin, CardModel):
    def __init__(self,
                 df: DataFrame,
                 vdf: DataFrame) -> None:
        super().__init__(df, vdf, "cost", name="cost")

        self.stratify_cols.extend(["element", "type_en"])

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

    def build(self,
              hp: HyperParameters,
              seed: int | None = None) -> models.Sequential:
        """
        Build a model.

        Args:
            hp (HyperParameters): A `HyperParameters` instance.

        Returns:
            A model instance.
        """

        m = models.Sequential(layers=[
            layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=self.IMAGE_SHAPE),
            layers.AveragePooling2D(padding='same'),
            layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
            layers.AveragePooling2D(padding='same'),
            layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
            layers.AveragePooling2D(padding='same'),
            layers.Dropout(0.2, seed=seed),
            layers.Flatten(),
            layers.Dense(hp.Int('dense_units', min_value=200, max_value=312, step=8), activation='relu'),
            layers.Dense(len(self.labels), activation="softmax")
        ], name=self.name)

        learning_rate = hp.Float('learning_rate',
                                 min_value=1.0e-6,
                                 max_value=1.0e-3,
                                 sampling='LOG')

        optimizer = optimizers.Adam(learning_rate=learning_rate,
                                    amsgrad=True)

        m.compile(optimizer=optimizer,
                  loss=self.loss,
                  metrics=self.metrics)
        return m


if __name__ == "__main__":
    from crystalvision.models import tune_model

    tune_model(Cost)

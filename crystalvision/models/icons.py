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
    from . import CardModel, BinaryMixin, CategoricalMixin, \
        BayesianOptimizationTunerMixin, RandomSearchTunerMixin
except ImportError:
    from crystalvision.models import CardModel, BinaryMixin, \
        CategoricalMixin, BayesianOptimizationTunerMixin, \
        RandomSearchTunerMixin

class Exburst(BinaryMixin, RandomSearchTunerMixin, CardModel):
    def __init__(self,
                 df: DataFrame,
                 vdf: DataFrame) -> None:
        super().__init__(df, vdf, "ex_burst", name="exburst")

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
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.IMAGE_SHAPE),
            layers.MaxPooling2D(),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(hp.Int('dense_units', min_value=32, max_value=512, step=32), activation='relu'),
            layers.Dense(1, activation="sigmoid")
        ], name=self.name)

        learning_rate = hp.Choice('learning_rate',
                                  values=[0.01, 0.001, 0.0001])
        optimizer = hp.Choice('optimizer',
                              values=['adam', 'rmsprop', 'sgd'])
        # learning_rate = hp.Float('learning_rate',
        #                          min_value=1e-4,
        #                          max_value=1e-2,
        #                          sampling='LOG')

        if optimizer == 'adam':
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            optimizer = optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = optimizers.SGD(learning_rate=learning_rate)

        m.compile(optimizer=optimizer,
                  loss=self.loss,
                  metrics=self.metrics)
        return m


class Multicard(BinaryMixin, RandomSearchTunerMixin, CardModel):
    def __init__(self,
                 df: DataFrame,
                 vdf: DataFrame) -> None:
        super().__init__(df, vdf, "multicard", name="multicard")

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
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.IMAGE_SHAPE),
            layers.MaxPooling2D(),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(hp.Int('dense_units', min_value=32, max_value=512, step=32), activation='relu'),
            layers.Dense(1, activation="sigmoid")
        ], name=self.name)

        learning_rate = hp.Choice('learning_rate',
                                  values=[0.01, 0.001, 0.0001])
        optimizer = hp.Choice('optimizer',
                              values=['adam', 'rmsprop', 'sgd'])
        # learning_rate = hp.Float('learning_rate',
        #                          min_value=1e-4,
        #                          max_value=1e-2,
        #                          sampling='LOG')

        if optimizer == 'adam':
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            optimizer = optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = optimizers.SGD(learning_rate=learning_rate)

        m.compile(optimizer=optimizer,
                  loss=self.loss,
                  metrics=self.metrics)
        return m


if __name__ == "__main__":
    from crystalvision.models import tune_model

    tune_model(Exburst)
    tune_model(Multicard)

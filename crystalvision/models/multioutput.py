# -*- coding: utf-8 -*-
"""
Type-based Card Hypermodels.

Todo:
    * N/A

"""
from typing import List

from pandas import DataFrame
from keras import layers, models, callbacks
from keras_tuner import HyperParameters
from sklearn.preprocessing import MultiLabelBinarizer

try:
    from . import CardModel
    from .callbacks import StopOnValue
    from .mixins.compiles import BinaryCrossMixin
    from .mixins.tuners import BayesianOptimizationTunerMixin
except ImportError:
    from __init__ import CardModel
    from crystalvision.models.callbacks import StopOnValue
    from crystalvision.models.mixins.compiles import BinaryCrossMixin
    from crystalvision.models.mixins.tuners import BayesianOptimizationTunerMixin


class MultiLabel(BinaryCrossMixin, BayesianOptimizationTunerMixin, CardModel):
    MAX_TRIALS = 50
    DEFAULT_EPOCHS = 50

    def __init__(self, df: DataFrame, vdf: DataFrame) -> None:
        self.name = "multilabel"
        self.tunable = True

        self._build = self.build
        self.build = self._build_wrapper

        self.df: DataFrame = self.filter_dataframe(df.copy())
        self.vdf: DataFrame = self.filter_dataframe(vdf.copy())

        self.feature_key: str = ["type_en", "cost", "element"]
        self.stratify_cols: List[str] = ["type_en", "cost", "element"]

        self.labels = []
        for fkey in self.feature_key:
            self.labels.extend(self.df[fkey].unique())

        self.mlb = MultiLabelBinarizer(classes=self.labels)

        self.df_codes = self.mlb.fit_transform(
            self.df[self.feature_key].to_records(index=False)
        )
        self.vdf_codes = self.mlb.transform(
            self.vdf[self.feature_key].to_records(index=False)
        )

        self.callbacks = [
            callbacks.EarlyStopping(
                monitor="val_accuracy",
                min_delta=0.005,
                patience=2,
                restore_best_weights=True,
            ),
            StopOnValue(),
        ]

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

        m = models.Sequential(
            layers=[
                layers.Conv2D(
                    32,
                    (3, 3),
                    padding="same",
                    activation="relu",
                    input_shape=self.IMAGE_SHAPE,
                ),
                layers.MaxPooling2D(padding="same"),
                layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
                layers.MaxPool2D(padding="same"),
                layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                layers.AveragePooling2D(padding="same"),
                layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                layers.AveragePooling2D(padding="same"),
                # layers.Dropout(0.2, seed=seed),
                layers.Flatten(),
                # layers.Dense(
                #     hp.Int("dense_units", min_value=2**13, max_value=2**15, step=128),
                #     activation="relu",
                # ),
                layers.Dense(
                    hp.Int(
                        "dense_units1",
                        min_value=2**11,
                        max_value=2**13,
                        sampling="LOG",
                    ),
                    activation="relu",
                ),
                layers.Dense(
                    2944,
                    activation="relu",
                ),
                layers.Dense(len(self.labels), activation="sigmoid"),
            ],
            name=self.name,
        )

        optimizer = self._optimizer_choice(
            "optimizer",
            hp,
            exclude={
                "Adadelta",
                "Adagrad",
                "Adam",
                "Amsgrad",
                "Adamax",
                "Ftrl",
                "Nadam",
            },
        )[1](
            momentum=hp.Float("mometum", min_value=0.0, max_value=0.999),
            learning_rate=hp.Float(
                "learning_rate", min_value=1.0e-7, max_value=1.0, sampling="LOG"
            ),
        )

        m.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
        return m


if __name__ == "__main__":
    from crystalvision.models import tune_model

    # from crystalvision.testmodels import IMAGE_DF

    tune_model(MultiLabel, num=10, save_models=False)

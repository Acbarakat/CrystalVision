# -*- coding: utf-8 -*-
"""
Type-based Card Hypermodels.

Todo:
    * N/A

"""
from typing import List

from pandas import DataFrame
from keras import layers, models, callbacks, metrics
from keras_tuner import HyperParameters
from sklearn.preprocessing import MultiLabelBinarizer

try:
    from . import CardModel
    from .ext.metrics import MyOneHotMeanIoU, MyOneHotIoU
    from .ext.callbacks import StopOnValue
    from .mixins.compiles import OneHotMeanIoUMixin
    from .mixins.tuners import BayesianOptimizationTunerMixin
except ImportError:
    from __init__ import CardModel
    from crystalvision.models.ext.metrics import MyOneHotMeanIoU, MyOneHotIoU
    from crystalvision.models.ext.callbacks import StopOnValue
    from crystalvision.models.mixins.compiles import OneHotMeanIoUMixin
    from crystalvision.models.mixins.tuners import BayesianOptimizationTunerMixin


class MultiLabel(OneHotMeanIoUMixin, BayesianOptimizationTunerMixin, CardModel):
    MAX_TRIALS: int = 100
    DEFAULT_EPOCHS: int = 50
    MAX_CONSECUTIVE_FAILED_TRIALS: int = 5

    def __init__(self, df: DataFrame, vdf: DataFrame) -> None:
        self.name: str = "multilabel"
        self.tunable: bool = True

        self._build = self.build
        self.build = self._build_wrapper

        self.df: DataFrame = self.filter_dataframe(df.copy())
        self.vdf: DataFrame = self.filter_dataframe(vdf.copy())

        self.feature_key: List[str] = ["type_en", "cost", "element"]
        self.stratify_cols: List[str] = ["type_en", "cost", "element"]

        self.labels: List[str] = []
        for fkey in self.feature_key:
            self.labels.extend(self.df[fkey].unique())

        idx: int = 0
        self._metrics: List[metrics.Metric] = []
        for fkey in self.feature_key:
            labels = self.df[fkey].unique()
            self._metrics.append(
                MyOneHotIoU(
                    target_class_ids=list(range(idx, len(labels) + idx)),
                    threshold=0.95,
                    name=f"{fkey}_accuracy",
                ),
                # ignore_class
            )
            idx += len(labels)

        self.mlb = MultiLabelBinarizer(classes=self.labels)

        self.df_codes = self.mlb.fit_transform(
            self.df[self.feature_key].to_records(index=False)
        )
        self.vdf_codes = self.mlb.transform(
            self.vdf[self.feature_key].to_records(index=False)
        )

        self.callbacks = [
            callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0.005,
                patience=5,
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

        pl1 = self._pooling2d_choice("pooling1", hp)[1]
        pl2 = self._pooling2d_choice("pooling2", hp)[1]
        pl3 = self._pooling2d_choice("pooling3", hp)[1]

        m = models.Sequential(
            layers=[
                layers.Input(shape=self.IMAGE_SHAPE, batch_size=batch_size),
                layers.Conv2D(
                    32,
                    (3, 3),
                    padding="same",
                    activation="relu",
                ),
                pl1(pool_size=(2, 2), padding="same"),
                layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
                pl2(pool_size=(2, 2), padding="same"),
                layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                pl3(pool_size=(2, 2), padding="same"),
                layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                # layers.Dropout(0.2, seed=seed),
                layers.Flatten(),
                layers.Dense(
                    hp.Int(
                        "dense_units1",
                        min_value=2**7,
                        max_value=2**13,
                        sampling="LOG",
                    ),
                    activation="relu",
                ),
                layers.Dense(len(self.labels), activation="sigmoid"),
            ],
            name=self.name,
        )

        optimizer = self._optimizer_choice("optimizer", hp)[1]

        # optimizer = self._optimizer_choice(
        #     "optimizer",
        #     hp,
        #     exclude={
        #         "Adadelta",
        #         "Adagrad",
        #         "Adam",
        #         "Amsgrad",
        #         "Adamax",
        #         "Ftrl",
        #         "Nadam",
        #     },
        # )[1](
        #     momentum=hp.Float("mometum", min_value=0.0, max_value=0.9),
        #     learning_rate=hp.Float(
        #         "learning_rate", min_value=1.0e-7, max_value=0.9, sampling="LOG"
        #     ),
        # )

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

    # from crystalvision.testmodels import IMAGE_DF

    tune_model(MultiLabel, num=MultiLabel.MAX_TRIALS, save_models=False)

# -*- coding: utf-8 -*-
"""
Element-based Card Hypermodels.

Todo:
    * N/A

"""
from functools import cached_property
from typing import List

from pandas import DataFrame
from keras import layers, models, optimizers, callbacks, metrics
from keras_tuner import HyperParameters
from sklearn.preprocessing import MultiLabelBinarizer

try:
    from . import CardModel
    from .mixins.compiles import (
        BinaryMixin,
        OneHotMeanIoUMixin,
    )
    from .mixins.tuners import (
        BayesianOptimizationTunerMixin,
        RandomSearchTunerMixin,
    )
    from .mixins.objective import WeightedMeanMultiObjective, Objective
    from .ext.metrics import MyOneHotMeanIoU
    from .ext.callbacks import StopOnValue
except ImportError:
    from crystalvision.models import CardModel
    from crystalvision.models.mixins.compiles import (
        BinaryMixin,
        OneHotMeanIoUMixin,
    )
    from crystalvision.models.mixins.tuners import (
        BayesianOptimizationTunerMixin,
        RandomSearchTunerMixin,
    )
    from crystalvision.models.mixins.objective import (
        WeightedMeanMultiObjective,
        Objective,
    )
    from crystalvision.models.ext.metrics import MyOneHotMeanIoU
    from crystalvision.models.ext.callbacks import StopOnValue


class Exburst(BinaryMixin, RandomSearchTunerMixin, CardModel):
    def __init__(self, df: DataFrame, vdf: DataFrame) -> None:
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


class Icons(OneHotMeanIoUMixin, BayesianOptimizationTunerMixin, CardModel):
    MAX_TRIALS: int = 50

    def __init__(self, df: DataFrame, vdf: DataFrame) -> None:
        self.name: str = "icons"
        self.tunable: bool = True

        self._build = self.build
        self.build = self._build_wrapper

        self.df: DataFrame = self.filter_dataframe(df.copy())
        self.vdf: DataFrame = self.filter_dataframe(vdf.copy())

        self.feature_key: List[str] = "icons"  # ["ex_burst", "multicard"]
        self.stratify_cols: List[str] = ["ex_burst", "multicard", "element", "type_en"]

        self.labels: List[str] = [
            elem[0] for elem in self.df["icons"].unique() if len(elem) == 1
        ]

        self._metrics: List[metrics.Metric] = []
        # TODO: Resolve the error when there is only one target_class_ids
        # for idx, fkey in enumerate(self.labels):
        #     self._metrics.append(
        #         MyOneHotIoU(
        #             target_class_ids=[idx],
        #             threshold=0.95,
        #             name=f"{fkey}_accuracy",
        #         )
        #     )

        self.mlb = MultiLabelBinarizer(classes=self.labels)

        self.df_codes = self.mlb.fit_transform(self.df[self.feature_key])
        self.vdf_codes = self.mlb.transform(self.vdf[self.feature_key])

        self.callbacks = [
            callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0.005,
                patience=5,
                restore_best_weights=True,
            ),
            StopOnValue(),
        ]

    @cached_property
    def objective(self) -> WeightedMeanMultiObjective:
        return WeightedMeanMultiObjective(
            [
                Objective("accuracy", "max"),
                Objective("val_accuracy", "max"),
                Objective("test_accuracy", "max"),
            ],
            weights=[0.9, 2.0, 0.1],
        )

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

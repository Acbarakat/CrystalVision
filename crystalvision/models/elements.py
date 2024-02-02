# -*- coding: utf-8 -*-
"""
Element-based Card Hypermodels.

Todo:
    * N/A

"""
from typing import List

from pandas import DataFrame
from keras import layers, models, optimizers, callbacks
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
    from .metrics import MyOneHotMeanIoU
    from .callbacks import StopOnValue
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
    from crystalvision.models.metrics import MyOneHotMeanIoU
    from crystalvision.models.callbacks import StopOnValue


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
                layers.Conv2D(
                    32,
                    kernel_size=(3, 3),
                    activation="relu",
                    input_shape=self.IMAGE_SHAPE,
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
        # pooling_type = hp.Choice('pooling', values=['max', 'avg'])
        # if pooling_type == 'max':
        #     pooling_layer = layers.MaxPooling2D
        # else:
        #     pooling_layer = layers.AveragePooling2D

        m = models.Sequential(
            layers=[
                layers.Conv2D(
                    32,
                    (3, 3),
                    padding="same",
                    activation="relu",
                    input_shape=self.IMAGE_SHAPE,
                ),
                layers.AveragePooling2D(padding="same"),
                layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
                layers.AveragePooling2D(padding="same"),
                layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                layers.AveragePooling2D(padding="same"),
                layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                layers.Dropout(0.2, seed=seed),
                layers.Flatten(),
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
        super().__init__(df, vdf, "element", name="element_v2")

        self.stratify_cols.extend(["type_en"])

        self.df["element_v2"] = self.df["element"].apply(lambda x: tuple(x.split("_")))
        self.vdf["element_v2"] = self.vdf["element"].apply(
            lambda x: tuple(x.split("_"))
        )

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
                layers.AveragePooling2D(padding="same"),
                layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
                layers.AveragePooling2D(padding="same"),
                layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                layers.AveragePooling2D(padding="same"),
                layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                # layers.Dropout(0.2, seed=seed),
                layers.Flatten(),
                # layers.Dense(hp.Int('dense_units', min_value=128, max_value=512, step=32), activation='relu'),
                layers.Dense(
                    hp.Int("dense_units", min_value=64, max_value=1024, step=32),
                    activation="relu",
                ),
                # layers.Dense(480, activation='relu'),
                layers.Dense(len(self.labels), activation="sigmoid"),
                # Threshold(0.95),
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
            metrics=MyOneHotMeanIoU(
                num_classes=len(self.labels), threshold=0.95, name="accuracy"
            ),
        )
        return m


if __name__ == "__main__":
    from crystalvision.models import tune_model

    # tune_model(Element)
    # tune_model(Mono)
    tune_model(ElementV2)

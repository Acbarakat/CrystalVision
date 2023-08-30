# -*- coding: utf-8 -*-
"""
Type-based Card Hypermodels.

Todo:
    * N/A

"""
from pandas import DataFrame
from keras import layers, models, optimizers
from keras_tuner import HyperParameters

from __init__ import CardModel, CategoricalMixin, \
    BayesianOptimizationTunerMixin


class CardTyping(CategoricalMixin, BayesianOptimizationTunerMixin, CardModel):
    def __init__(self,
                 df: DataFrame,
                 vdf: DataFrame) -> None:
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
        pooling_type = hp.Choice('pooling', values=['maxavg', 'avgmax'])
        if pooling_type == 'max':
            pl1, pl2 = layers.MaxPooling2D, layers.AveragePooling2D
        else:
            pl1, pl2 = layers.AveragePooling2D, layers.MaxPooling2D

        m = models.Sequential(layers=[
            layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=self.IMAGE_SHAPE),
            pl1(padding='same'),
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            pl2(padding='same'),
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            # layers.Dropout(0.2, seed=seed),
            layers.Flatten(),
            # layers.Dense(hp.Int('dense_units', min_value=128, max_value=512, step=128), activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(self.labels), activation="softmax")
        ], name=self.name)

        optimizer = hp.Choice('optimizer',
                              values=['adam', 'rmsprop', 'sgd', 'nesterov'])

        if optimizer == 'adam':
            optimizer = optimizers.Adam()
        elif optimizer == 'rmsprop':
            optimizer = optimizers.RMSprop()
        elif optimizer == 'nesterov':
            optimizer = optimizers.SGD(momentum=0.9, nesterov=True)
        else:
            optimizer = optimizers.SGD()

        m.compile(optimizer=optimizer,
                  loss=self.loss,
                  metrics=self.metrics)
        return m


if __name__ == "__main__":
    import os

    import pandas as pd
    from keras.models import load_model

    from data import SRC_DIR
    from data.dataset import imagine_database, make_database
    from models import MODEL_DIR

    df = imagine_database()
    for lang in ("_es", "_de", "_fr", "_it", "_ja"):
        df = df.loc[:, ~df.columns.str.endswith(lang)]

    TEST_IMG_DIR = os.path.abspath(os.path.join(MODEL_DIR,
                                                "..",
                                                "test")) + os.sep

    vdf = pd.read_json(os.path.join(SRC_DIR, "testmodels.json"))
    vdf["filename"] = TEST_IMG_DIR + vdf["uri"].index.astype(str) + ".jpg"
    vdf = vdf.merge(make_database(), on="code", how='left', sort=False)
    vdf.drop(["thumbs", "images", "uri", "id"], axis=1, inplace=True)
    for lang in ("_es", "_de", "_fr", "_it", "_ja"):
        vdf = vdf.loc[:, ~vdf.columns.str.endswith(lang)]
    vdf.query("full_art != 1 and focal == 1", inplace=True)
    # print(vdf)

    m = CardTyping(df, vdf)
    training_dataset, testing_dataset = m.split_data(test_size=0.1,
                                                     random_state=23,
                                                     shuffle=True)
    validation_dataset = m.split_validation()[0]
    m.clear_cache()
    m.tune_and_save(training_dataset,
                    testing_dataset,
                    validation_dataset,
                    3)

    best_model = load_model(os.path.join(MODEL_DIR, f"{m.name}_1.h5"))
    # print(best_model.summary())

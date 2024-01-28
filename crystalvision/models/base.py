# -*- coding: utf-8 -*-
"""
Base Card HyperModels and Mixins.

Attributes:
    SRC_DIR (str): Where the src folder is
    MODEL_DIR (str): Where the models are stored

Todo:
    * N/A

"""
import os
import json
import math
from typing import Tuple, List, Any

from pandas import DataFrame
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import callbacks, models
from keras.utils import image_utils
from keras.utils.image_dataset import paths_and_labels_to_dataset
from keras_tuner import HyperModel, HyperParameters

from crystalvision.models import MODEL_DIR
from crystalvision.models.callbacks import StopOnValue


from ..data.dataset import extendDataset


class CardModel(HyperModel):
    """Base hypermodel for Cards."""

    IMAGE_SHAPE: Tuple[int, int, int] = (250, 179, 3)
    DEFAULT_EPOCHS = 30

    def __init__(self,
                 df: DataFrame,
                 vdf: DataFrame,
                 feature_key: str,
                 name: str = "unknown",
                 tunable: bool = True) -> None:
        """
        Init the base hypermodel, CardModel.

        Args:
            df (DataFrame): DataFrame of test/train data
            vdf (DataFrame): DataFrame of validation data
            feature_key (str): The binary/categorical class key
            name (str, optional): Name of the hypermodel.
                (defaults is "unknown")
            tunable (bool, optional): If the hypermodel is tunable.
                (defaults is True)
        """
        super().__init__(name=name, tunable=tunable)

        self.df: DataFrame = self.filter_dataframe(df.copy())
        self.vdf: DataFrame = self.filter_dataframe(vdf.copy())

        self.feature_key: str = feature_key
        self.stratify_cols: List[str] = [feature_key]

        self.df_codes, uq1 = self.df[feature_key].factorize(sort=True)
        self.vdf_codes, uq2 = self.vdf[feature_key].factorize(sort=True)

        diff = set(uq1).difference(set(uq2))
        if len(diff) > 0:
            raise ValueError(f"missing/additional datapoints: {diff}")

        self.labels = uq1

        self.callbacks = [
            callbacks.EarlyStopping(monitor='val_accuracy',
                                    min_delta=0.005,
                                    patience=2,
                                    restore_best_weights=True),
            StopOnValue(),
        ]

    def split_data(self,
                   interpolation: str = "bilinear",
                   batch_size: int = 64,
                   **kwargs) -> List[tf.data.Dataset]:
        stratify = self.df[self.stratify_cols]
        # print(self.df.groupby(self.stratify_cols).count()['id'])
        img_paths = self.df["filename"]
        interpolation = image_utils.get_interpolation("bilinear")

        X_train, X_test, y_train, y_test = train_test_split(img_paths,
                                                            self.df_codes,
                                                            **kwargs,
                                                            stratify=stratify)

        training_dataset = extendDataset(paths_and_labels_to_dataset(
            image_paths=X_train.tolist(),
            image_size=self.IMAGE_SHAPE[:2],
            num_channels=self.IMAGE_SHAPE[2],
            labels=y_train.tolist(),
            label_mode=self.LABEL_MODE,
            num_classes=len(self.labels),
            interpolation=interpolation,
        ), batch_size=batch_size)

        testing_dataset = extendDataset(paths_and_labels_to_dataset(
            image_paths=X_test.tolist(),
            image_size=self.IMAGE_SHAPE[:2],
            num_channels=self.IMAGE_SHAPE[2],
            labels=y_test.tolist(),
            label_mode=self.LABEL_MODE,
            num_classes=len(self.labels),
            interpolation=interpolation,
        ), batch_size=batch_size)

        return training_dataset, testing_dataset

    def split_validation(self,
                         interpolation: str = "bilinear",
                         batch_size: int = 64) -> List[tf.data.Dataset]:
        interpolation = image_utils.get_interpolation("bilinear")

        validation_dataset = paths_and_labels_to_dataset(
            image_paths=self.vdf['filename'].tolist(),
            image_size=self.IMAGE_SHAPE[:2],
            num_channels=self.IMAGE_SHAPE[2],
            labels=self.vdf_codes,
            label_mode=self.LABEL_MODE,
            num_classes=len(self.labels),
            interpolation=interpolation,
        ).batch(batch_size).cache()

        return [validation_dataset]

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
        df.query("~filename.str.contains('_jp')", inplace=True)  # Japanese

        return df

    def build(self,
              hp: HyperParameters,
              seed: int | None = None) -> models.Model:
        """
        Build a model.

        Args:
            hp (HyperParameters): A `HyperParameters` instance.
            seed (int | None, optional): _description_.
                (defaults is None)

        Returns:
            A model instance.

        """
        raise NotImplementedError()

    def tune_and_save(self,
                      train_ds: tf.data.Dataset,
                      test_ds: tf.data.Dataset,
                      validation_ds: tf.data.Dataset,
                      num_models: int = 1) -> None:
        """
        Tune the hypermodel and save the top num_models as .h5 files.

        Args:
            train_ds (tf.data.Dataset): training data
            test_ds (tf.data.Dataset): testing data
            validation_ds (tf.data.Dataset): validation data
            num_models (int, optional): Top number of models to save.
                (defaults is 1)
        """
        self.search(train_ds, test_ds, validation_ds)
        # print(self.tuner.results_summary())
        best_models, best_hparams = (
            self.tuner.get_best_models(num_models=num_models),
            self.tuner.get_best_hyperparameters(num_trials=num_models)
        )
        
        best_trials = {}
        for idx, (bm, bhp) in enumerate(zip(best_models, best_hparams)):
            # print(bm.summary())
            # print(bm.optimizer.get_config())
            trial = self.tuner.oracle.trials[bm.name[-2:]]
            print(bm.name, bhp.values, trial.score)
            best_trials[bm.name] = bhp.values
            best_trials[bm.name]["score"] = trial.score
            bm.save(os.path.join(MODEL_DIR,
                                 f"{self.name}_{idx + 1}.h5"))

        # Save the labels
        with open(os.path.join(MODEL_DIR, f"{self.name}.json"), "w+") as fp:
            json.dump(self.labels.to_list(), fp)

        # Save the top X
        with open(os.path.join(MODEL_DIR, f"{self.name}_best.json"), "w+") as fp:
            json.dump(best_trials, fp, indent=4)

    def fit(self,
            hp: HyperParameters,
            model: models.Model,
            train_ds: tf.data.Dataset,
            *args,
            testing_data: tf.data.Dataset | None = None,
            epochs: int | None = None,
            **kwargs) -> Any:
        """
        Train the hypermodel.

        Args:
            hp (HyperParameters): The hyperparameters
            model: `keras.Model` built in the `build()` function
            train_ds (tf.data.Dataset): Training data
            *args: All arguments passed to `Tuner.search`
            epochs (int | None): epochs (iterations on a dataset).
                (default is DEFAULT_EPOCHS)
            testing_data (tf.data.Dataset): Testing data.
                (default is None)
            **kwargs: All arguments passed to `Tuner.search()` are in the
                `kwargs` here. It always contains a `callbacks` argument, which
                is a list of default Keras callback functions for model
                checkpointing, tensorboard configuration, and other tuning
                utilities. If `callbacks` is passed by the user from
                `Tuner.search()`, these default callbacks will be appended to
                the user provided list.

        Returns:
            A `History` object, which is the return value of `model.fit()`, a
            dictionary, or a float.

            If return a dictionary, it should be a dictionary of the metrics to
            track. The keys are the metric names, which contains the
            `objective` name. The values should be the metric values.

            If return a float, it should be the `objective` value.
        """
        if epochs is None:
            epochs = self.DEFAULT_EPOCHS
        try:
            history = model.fit(train_ds,
                                epochs=epochs,
                                steps_per_epoch=len(train_ds),
                                **kwargs)
            test_loss, test_acc = model.evaluate(testing_data)
        except tf.errors.ResourceExhaustedError:
            return -math.inf

        return {
            "loss": history.history['loss'][-1],
            "accuracy": history.history['accuracy'][-1],
            "val_loss": history.history['val_loss'][-1],
            "val_accuracy": history.history['val_accuracy'][-1],
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        }

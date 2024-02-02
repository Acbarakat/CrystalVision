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
from typing import Tuple, List, Any, Set
from functools import partial

import numpy as np
from pandas import DataFrame
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import callbacks, models, layers, optimizers
from keras.utils import image_utils
from keras.utils.image_dataset import paths_and_labels_to_dataset
from keras_tuner import HyperModel, HyperParameters

from crystalvision.models import MODEL_DIR
from crystalvision.models.callbacks import StopOnValue
from crystalvision.models.layers import MinPooling2D


from ..data.dataset import extendDataset


layers.MinPooling2D = MinPooling2D
optimizers.Amsgrad = partial(optimizers.Adam, amsgrad=True)
optimizers.Nesterov = partial(optimizers.SGD, nesterov=True)
optimizers.RMSpropCentered = partial(optimizers.RMSprop, centered=True)


class CardModel(HyperModel):
    """Base hypermodel for Cards."""

    IMAGE_SHAPE: Tuple[int, int, int] = (250, 179, 3)
    DEFAULT_EPOCHS = 30

    def __init__(
        self,
        df: DataFrame,
        vdf: DataFrame,
        feature_key: str,
        name: str = "unknown",
        tunable: bool = True,
    ) -> None:
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
            callbacks.EarlyStopping(
                monitor="val_accuracy",
                min_delta=0.005,
                patience=2,
                restore_best_weights=True,
            ),
            StopOnValue(),
        ]

    DEFAULT_BATCH_SIZE = 256

    def split_data(
        self,
        interpolation: str = "bilinear",
        batch_size: int | None = None,
        random_state: int | None = None,
        **kwargs,
    ) -> List[tf.data.Dataset]:
        if batch_size is None:
            batch_size = self.DEFAULT_BATCH_SIZE
        stratify = self.df[self.stratify_cols]
        img_paths = self.df["filename"]
        interpolation = image_utils.get_interpolation(interpolation)

        X_train, X_test, y_train, y_test = train_test_split(
            img_paths,
            self.df_codes,
            **kwargs,
            stratify=stratify,
            random_state=random_state,
        )

        training_dataset = extendDataset(
            paths_and_labels_to_dataset(
                image_paths=X_train.tolist(),
                image_size=self.IMAGE_SHAPE[:2],
                num_channels=self.IMAGE_SHAPE[2],
                labels=y_train.tolist(),
                label_mode=self.LABEL_MODE,
                num_classes=len(self.labels),
                interpolation=interpolation,
            ),
            batch_size=batch_size,
        )

        testing_dataset = extendDataset(
            paths_and_labels_to_dataset(
                image_paths=X_test.tolist(),
                image_size=self.IMAGE_SHAPE[:2],
                num_channels=self.IMAGE_SHAPE[2],
                labels=y_test.tolist(),
                label_mode=self.LABEL_MODE,
                num_classes=len(self.labels),
                interpolation=interpolation,
            ),
            batch_size=batch_size,
        )

        return training_dataset, testing_dataset

    def split_validation(
        self, interpolation: str = "bilinear", batch_size: int | None = None
    ) -> List[tf.data.Dataset]:
        interpolation = image_utils.get_interpolation(interpolation)
        if batch_size is None:
            batch_size = self.DEFAULT_BATCH_SIZE

        validation_dataset = paths_and_labels_to_dataset(
            image_paths=self.vdf["filename"].tolist(),
            image_size=self.IMAGE_SHAPE[:2],
            num_channels=self.IMAGE_SHAPE[2],
            labels=self.vdf_codes,
            label_mode=self.LABEL_MODE,
            num_classes=len(self.labels),
            interpolation=interpolation,
        )

        if batch_size:
            validation_dataset = validation_dataset.batch(batch_size)

        validation_dataset = validation_dataset.cache()

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

    def build(self, hp: HyperParameters, seed: int | None = None) -> models.Model:
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

    @staticmethod
    def _pooling2d_choice(
        name: str, hp: HyperParameters, exclude: Set[str] | None = None
    ) -> Tuple[str, layers.Layer]:
        choices: Set[str] = {"Min", "Max", "Average"}
        if exclude is not None:
            choices -= exclude

        pooling_type = hp.Choice(name, values=list(choices))

        return pooling_type, getattr(layers, f"{pooling_type}Pooling2D")

    @staticmethod
    def _optimizer_choice(
        name: str, hp: HyperParameters, exclude: Set[str] | None = None
    ) -> Tuple[str, optimizers.Optimizer]:
        choices: Set[str] = {
            "Adadelta",
            "Adagrad",
            "Adam",
            "Amsgrad",
            # "NonFusedAdam", "NonFusedAmsgrad"
            "Adamax",
            "Ftrl",
            "SGD",
            "Nesterov",
            "Nadam",
            "RMSprop",
            "RMSpropCentered",
        }

        if exclude is not None:
            choices -= exclude

        optimizer = hp.Choice(name, values=list(choices))

        return optimizer, getattr(optimizers, optimizer)

    def tune_and_save(
        self,
        num_models: int = 1,
        random_state: int | None = None,
        save_models: bool = True,
    ) -> None:
        """
        Tune the hypermodel and save the top num_models as .h5 files.

        Args:
            train_ds (tf.data.Dataset): training data
            test_ds (tf.data.Dataset): testing data
            validation_ds (tf.data.Dataset): validation data
            num_models (int, optional): Top number of models to save.
                (defaults is 1)
            save_models (bool, optional): Save the top models.
                (default is True)
        """
        self.search(random_state=random_state)
        # print(self.tuner.results_summary())
        df = []

        best_trials = self.tuner.oracle.get_best_trials(num_trials=num_models)
        if save_models:
            best_hparams = self.tuner.get_best_hyperparameters(num_trials=num_models)
            best_models = self.tuner.get_best_models(num_models=num_models)

            for idx, (bm, bhp, bt) in enumerate(
                zip(best_models, best_hparams, best_trials)
            ):
                df.append(
                    dict(
                        name=bm.name,
                        score=bt.score,
                        loss=bt.metrics.get_last_value("loss"),
                        accuracy=bt.metrics.get_last_value("accuracy"),
                        val_loss=bt.metrics.get_last_value("val_loss"),
                        val_accuracy=bt.metrics.get_last_value("val_accuracy"),
                        test_loss=bt.metrics.get_last_value("test_loss"),
                        test_accuracy=bt.metrics.get_last_value("test_accuracy"),
                        **bhp.values,
                    )
                )
                bm.save(os.path.join(MODEL_DIR, f"{self.name}_{idx + 1}.h5"))
        else:
            for trial in best_trials:
                df.append(
                    dict(
                        name=f"trial_{trial.trial_id}",
                        score=trial.score,
                        loss=trial.metrics.get_last_value("loss"),
                        accuracy=trial.metrics.get_last_value("accuracy"),
                        val_loss=trial.metrics.get_last_value("val_loss"),
                        val_accuracy=trial.metrics.get_last_value("val_accuracy"),
                        test_loss=trial.metrics.get_last_value("test_loss"),
                        test_accuracy=trial.metrics.get_last_value("test_accuracy"),
                        **trial.hyperparameters.values,
                    )
                )

        df = DataFrame(df).set_index("name")
        print(df)

        # Save the labels
        with open(os.path.join(MODEL_DIR, f"{self.name}.json"), "w+") as fp:
            json.dump(self.labels.to_list(), fp)

        # Save the top X
        with open(os.path.join(MODEL_DIR, f"{self.name}_best.json"), "w+") as fp:
            json.dump(best_trials, fp, indent=4)

    def fit(
        self,
        hp: HyperParameters,
        model: models.Model,
        *args,
        epochs: int | None = None,
        batch_size: int | None = None,
        random_state: int | None = None,
        **kwargs,
    ) -> Any:
        """
        Train the hypermodel.

        Args:
            hp (HyperParameters): The hyperparameters
            model: `keras.Model` built in the `build()` function
            train_ds (tf.data.Dataset): Training data
            *args: All arguments passed to `Tuner.search`
            epochs (int | None): epochs (iterations on a dataset).
                (default is DEFAULT_EPOCHS)
            batch_size (int | None): batch_size of dataset.
                (default is DEFAULT_BATCH_SIZE)
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
            epochs = hp.values.get("epochs", self.DEFAULT_EPOCHS)
        if batch_size is None:
            batch_size = hp.values.get("batch_size", self.DEFAULT_BATCH_SIZE)

        train_ds, testing_ds = self.split_data(
            test_size=0.1,
            random_state=random_state,
            shuffle=True,
            batch_size=hp.values.get("batch_size"),
        )
        validation_ds = self.split_validation(batch_size=hp.values.get("batch_size"))[0]

        try:
            history = model.fit(
                train_ds,
                batch_size=batch_size,
                epochs=epochs,
                steps_per_epoch=len(train_ds),
                validation_data=validation_ds,
                validation_steps=len(validation_ds),
                **kwargs,
            )
            test_loss, test_acc = model.evaluate(testing_ds, batch_size=batch_size)
        except tf.errors.ResourceExhaustedError:
            return (np.nan, np.nan, np.nan, np.nan)

        print(f"test_lost: {test_loss}")
        print(f"test_acc: {test_acc}")

        val_accuracy = history.history["val_accuracy"][-1]
        # if val_accuracy > 1.0:
        #      val_accuracy = np.nan

        val_loss = history.history["val_loss"][-1]
        if val_loss < 0.0:
            val_loss = np.nan

        if test_loss < 0.0:
            test_loss = np.nan

        return {
            "loss": history.history["loss"][-1],
            "accuracy": history.history["accuracy"][-1],
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "multi_objective": (val_accuracy, test_acc, val_loss, test_loss),
        }

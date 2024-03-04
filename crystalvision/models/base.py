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
import array
import pickle
from pathlib import Path
from typing import Tuple, List, Any, Set
from functools import partial, cached_property

import numpy as np
import scipy.sparse as sp
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from keras import callbacks, models, layers, optimizers, backend, metrics
from keras_tuner import HyperModel, HyperParameters

try:
    from . import MODEL_DIR
    from .ext.callbacks import StopOnValue
    from .ext.layers import MinPooling2D
    from .ext.metrics import MyOneHotIoU
    from ..data.dataset import (
        extend_dataset,
        paths_and_labels_to_dataset,
    )
    from .mixins.objective import WeightedMeanMultiObjective, Objective
    from .mixins.compiles import OneHotMeanIoUMixin
except ImportError:
    from crystalvision.models import MODEL_DIR
    from crystalvision.models.ext.callbacks import StopOnValue
    from crystalvision.models.ext.layers import MinPooling2D
    from crystalvision.data.dataset import (
        extend_dataset,
        paths_and_labels_to_dataset,
    )
    from crystalvision.models.ext.metrics import MyOneHotIoU
    from crystalvision.models.mixins.objective import (
        WeightedMeanMultiObjective,
        Objective,
    )
    from crystalvision.models.mixins.compiles import OneHotMeanIoUMixin


layers.MinPooling2D = MinPooling2D
optimizers.Amsgrad = partial(optimizers.Adam, amsgrad=True, name="amsgrad")
optimizers.Nesterov = partial(optimizers.SGD, nesterov=True, name="nesterov")
optimizers.RMSpropCentered = partial(
    optimizers.RMSprop, centered=True, name="rmspropcentered"
)

if backend.backend() == "tensorflow":
    from keras.src.utils.module_utils import tensorflow as tf
    from tensorflow.python.framework.errors_impl import (
        InvalidArgumentError,
    )  # pylint: disable=E611
    import tf2onnx
    import onnx

    ResourceExhaustedError = tf.errors.ResourceExhaustedError
else:
    # TODO: Find the real equivalents
    ResourceExhaustedError = ResourceWarning
    InvalidArgumentError = ArithmeticError


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
        test_size: float = 0.2,
        image_shape: Tuple[int, int, int] | None = None,
        **kwargs,
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
        self.test_size: float = test_size
        if image_shape:
            self.IMAGE_SHAPE = image_shape

    DEFAULT_BATCH_SIZE = 256

    @cached_property
    def objective(self) -> WeightedMeanMultiObjective:
        return WeightedMeanMultiObjective(
            [
                Objective("accuracy", "max"),
                Objective("val_accuracy", "max"),
                Objective("test_accuracy", "max"),
            ],
            weights=[1.0 - self.test_size, 3.0, self.test_size],
        )

    def split_data(
        self,
        interpolation: str = "bilinear",
        batch_size: int | None = None,
        random_state: int | None = None,
        **kwargs,
    ) -> List[Any]:
        if batch_size is None:
            batch_size = self.DEFAULT_BATCH_SIZE
        stratify = self.df[self.stratify_cols]
        img_paths = self.df["filename"]

        X_train, X_test, y_train, y_test = train_test_split(
            img_paths,
            self.df_codes,
            **kwargs,
            stratify=stratify,
            random_state=random_state,
        )

        # image_size: Tuple of `(height, width)` integer. Target size.
        training_dataset = extend_dataset(
            paths_and_labels_to_dataset(
                image_paths=X_train.tolist(),
                image_size=self.IMAGE_SHAPE[:2],
                num_channels=self.IMAGE_SHAPE[2],
                labels=y_train.tolist(),
                label_mode=self.LABEL_MODE,  # pylint: disable=E1101
                num_classes=len(self.labels),
                interpolation=interpolation,
                data_format="channels_last",
            ),
            batch_size=batch_size,
        )

        testing_dataset = extend_dataset(
            paths_and_labels_to_dataset(
                image_paths=X_test.tolist(),
                image_size=self.IMAGE_SHAPE[:2],
                num_channels=self.IMAGE_SHAPE[2],
                labels=y_test.tolist(),
                label_mode=self.LABEL_MODE,  # pylint: disable=E1101
                num_classes=len(self.labels),
                interpolation=interpolation,
                data_format="channels_last",
            ),
            batch_size=batch_size,
        )

        return training_dataset, testing_dataset

    def split_validation(
        self, interpolation: str = "bilinear", batch_size: int | None = None
    ) -> List[Any]:
        if batch_size is None:
            batch_size = self.DEFAULT_BATCH_SIZE

        validation_dataset = paths_and_labels_to_dataset(
            image_paths=self.vdf["filename"].tolist(),
            image_size=self.IMAGE_SHAPE[:2],
            num_channels=self.IMAGE_SHAPE[2],
            labels=self.vdf_codes,
            label_mode=self.LABEL_MODE,  # pylint: disable=E1101
            num_classes=len(self.labels),
            interpolation=interpolation,
            data_format="channels_last",
        )

        if batch_size:
            validation_dataset = validation_dataset.batch(batch_size)

        if hasattr(validation_dataset, "cache"):
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
        # df.query("~filename.str.contains('_jp')", inplace=True)  # Japanese

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
        Tune the hypermodel and save the top num_models as .keras files.

        Args:
            train_ds (tf.data.Dataset): training data
            test_ds (tf.data.Dataset): testing data
            validation_ds (tf.data.Dataset): validation data
            num_models (int, optional): Top number of models to save.
                (defaults is 1)
            save_models (bool, optional): Save the top models.
                (default is True)
        """
        # Save the labels
        with open(MODEL_DIR / f"{self.name}.json", "w+") as fp:
            json.dump(self.labels, fp)

        self.search(random_state=random_state)  # pylint: disable=E1101

        if not save_models:
            return

        best_models = self.tuner.get_best_models(  # pylint: disable=E1101
            num_models=num_models
        )
        best_trials = self.tuner.oracle.get_best_trials(  # pylint: disable=E1101
            num_trials=num_models
        )

        for idx, (bm, bt) in enumerate(zip(best_models, best_trials)):
            print(bm.name, f"trial_{bt.trial_id}", bt.score)
            print(bm.summary())
            if backend.backend() == "tensorflow":
                bm.output_names = None
                spec = (
                    tf.TensorSpec((None, *self.IMAGE_SHAPE), tf.float32, name="input"),
                )
                onnx_model, _ = tf2onnx.convert.from_keras(bm, spec)
                onnx.save(
                    onnx_model, os.path.join(MODEL_DIR, f"{self.name}_{idx + 1}.onnx")
                )
            else:
                bm.save(os.path.join(MODEL_DIR, f"{self.name}_{idx + 1}.keras"))

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
            test_size=self.test_size,
            random_state=random_state,
            shuffle=True,
            batch_size=batch_size,
        )
        validation_ds = self.split_validation(batch_size=batch_size)[0]

        try:
            history = model.fit(
                train_ds,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=validation_ds,
                **kwargs,
            )
            test_metrics = model.evaluate(
                testing_ds, batch_size=batch_size, return_dict=True
            )
        except (ResourceExhaustedError, InvalidArgumentError, RuntimeError) as err:
            if isinstance(err, RuntimeError):
                if "out of memory" not in str(err):
                    raise err
                print("Caught runtime error: ", str(err))
            elif isinstance(err, InvalidArgumentError):
                if "Graph execution error" not in str(err):
                    raise err
                print("Caught invalidargument error: ", str(err))
            result = {
                "loss": np.inf,
                "accuracy": 0,
                "val_loss": np.inf,
                "val_accuracy": 0,
                "test_loss": np.inf,
                "test_accuracy": 0,
                "multi_objective": None,
            }
            result["multi_objective"] = self.objective.get_value(result)

            return result

        result = {
            "loss": history.history["loss"][-1],
            "accuracy": history.history["accuracy"][-1],
            "val_loss": history.history["val_loss"][-1],
            "val_accuracy": history.history["val_accuracy"][-1],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
        }
        result["multi_objective"] = self.objective.get_value(result)

        return result

    def save_multilabels(self) -> None:
        pass


class MyMultiLabelBinarizer(MultiLabelBinarizer):
    def _transform(self, y, class_mapping):
        """Transforms the label sets with a given mapping.

        Parameters
        ----------
        y : iterable of iterables
            A set of labels (any orderable and hashable object) for each
            sample. If the `classes` parameter is set, `y` will not be
            iterated.

        class_mapping : Mapping
            Maps from label to column index in label indicator matrix.

        Returns
        -------
        y_indicator : sparse matrix of shape (n_samples, n_classes)
            Label indicator matrix. Will be of CSR format.
        """
        indices = array.array("i")
        indptr = array.array("i", [0])
        for labels in y:
            index = set()
            for label in labels:
                if isinstance(label, (set, tuple, list)):
                    for sub_label in label:
                        index.add(class_mapping[sub_label])
                else:
                    index.add(class_mapping[label])

            indices.extend(index)
            indptr.append(len(indices))

        data = np.ones(len(indices), dtype=int)

        return sp.csr_matrix(
            (data, indices, indptr), shape=(len(indptr) - 1, len(class_mapping))
        )


class MultiLabelCardModel(  # pylint: disable=W0223
    OneHotMeanIoUMixin, CardModel  # pylint: disable=W231
):
    """Base multilabel hypermodel for Cards."""

    def __init__(
        self,
        name: str,
        df: DataFrame,
        vdf: DataFrame,
        feature_key: List[str] | str,
        stratify_cols: List[str],
        labels: List[str] | None = None,
        generate_metrics: bool = True,
        test_size: float = 0.2,
        one_hot_threshold: float = 0.95,
        image_shape: Tuple[int, int, int] | None = None,
        **kwargs,
    ) -> None:
        self.name: str = name
        self.tunable: bool = True

        self._build = self.build
        self.build = self._build_wrapper

        self.df: DataFrame = self.filter_dataframe(df.copy())
        self.vdf: DataFrame = self.filter_dataframe(vdf.copy())

        self.feature_key: List[str] = feature_key
        self.stratify_cols: List[str] = stratify_cols

        if labels:
            self.labels = labels
        else:
            self.labels: List[str] = []
            for fkey in self.feature_key:
                labels = self.df[fkey].explode().unique()
                assert False not in [
                    isinstance(sub_label, str) for sub_label in labels
                ], f"{labels} {[isinstance(sub_label, str) for sub_label in labels]}"
                self.labels.extend(labels)

        self._metrics: List[metrics.Metric] = []

        if generate_metrics:
            idx: int = 0
            for fkey in self.feature_key:
                labels = self.df[fkey].unique()
                self._metrics.append(
                    MyOneHotIoU(
                        target_class_ids=list(range(idx, len(labels) + idx)),
                        threshold=one_hot_threshold,
                        name=f"{fkey}_accuracy",
                        sparse_y_pred=backend.backend() != "torch",
                    )
                )
                idx += len(labels)

        self.save_multilabels()

        self.callbacks = [
            callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0.005,
                patience=5,
                restore_best_weights=True,
            ),
            StopOnValue(),
        ]
        self.test_size: float = test_size
        self.one_hot_threshold: float = one_hot_threshold
        if image_shape:
            self.IMAGE_SHAPE = image_shape

    def save_multilabels(self) -> None:
        mlb_file = Path(MODEL_DIR / self.name / f"{self.name}_mlb.pkl")
        if not mlb_file.exists():
            mlb = MyMultiLabelBinarizer(classes=self.labels)

            self.df_codes = mlb.fit_transform(
                self.df[self.feature_key].to_records(index=False)
            )
            self.vdf_codes = mlb.transform(
                self.vdf[self.feature_key].to_records(index=False)
            )

            if not mlb_file.parent.exists():
                mlb_file.parent.mkdir()

            with mlb_file.open("wb+") as f:
                pickle.dump((mlb, self.df_codes, self.vdf_codes), f)
        else:
            with mlb_file.open("rb") as f:
                _, self.df_codes, self.vdf_codes = pickle.load(f)

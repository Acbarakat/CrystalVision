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
import shutil
from typing import Tuple, List, Any
from functools import cached_property

import tensorflow as tf
from keras import callbacks, models, losses, metrics
from keras_tuner import HyperModel, HyperParameter
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.tuner import maybe_distribute


SRC_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(SRC_DIR, "..", "data", "model")


class BinaryMixin:
    """Binary compile mixins."""

    @cached_property
    def loss(self) -> losses.BinaryCrossentropy:
        """The Binary loss function."""
        return losses.BinaryCrossentropy()

    @cached_property
    def metrics(self) -> List[metrics.Metric]:
        """A list of Binary Metrics."""
        return [tf.keras.metrics.BinaryAccuracy(name='accuracy')]


class CategoricalMixin:
    """Categorical compile mixins."""

    @cached_property
    def loss(self) -> losses.CategoricalCrossentropy:
        """The Categorical loss function."""
        return losses.CategoricalCrossentropy()

    @cached_property
    def metrics(self) -> List[metrics.Metric]:
        """A list of Categorical Metrics."""
        return [tf.keras.metrics.CategoricalAccuracy(name='accuracy')]


class MyRandomSearch(RandomSearch):
    """Random search tuner."""

    def load_model(self, trial):
        """
        Load a Model from a given trial.

        For models that report intermediate results to the `Oracle`, generally
        `load_model` should load the best reported `step` by relying of
        `trial.best_step`.

        Args:
            trial: A `Trial` instance, the `Trial` corresponding to the model
                to load.
        """
        model = self._try_build(trial.hyperparameters)
        # Reload best checkpoint.
        # Only load weights to avoid loading `custom_objects`.
        with maybe_distribute(self.distribution_strategy):
            fname = self._get_checkpoint_fname(trial.trial_id)
            model.load_weights(fname).expect_partial()
            model._name += f"_trial{trial.trial_id}"
        return model


class RandomSearchTunerMixin:
    """RandomSearch Tuner Mixin."""

    MAX_TRIALS = 20
    MAX_EXECUTIONS = 1

    @cached_property
    def tuner(self) -> RandomSearch:
        """Random search tuner."""
        return MyRandomSearch(self,
                              objective="val_accuracy",
                              max_trials=self.MAX_TRIALS,
                              executions_per_trial=self.MAX_EXECUTIONS,
                              directory=MODEL_DIR,
                              project_name=self.name)

    def clear_cache(self) -> None:
        """Delete the hypermodel cache."""
        shutil.rmtree(os.path.join(MODEL_DIR, self.name))

    def search(self, train_ds, test_ds, validation_ds) -> None:
        """
        Perform a search for best hyperparameter configuations.

        Args:
            train_ds (tf.data.Dataset): training data
            test_ds (tf.data.Dataset): testing data
            validation_ds (tf.data.Dataset): validation data
        """
        tf.keras.backend.clear_session()

        self.tuner.search(train_ds,
                          testing_data=test_ds,
                          validation_data=validation_ds,
                          validation_steps=len(validation_ds),
                          callbacks=self.callbacks)


class CardModel(RandomSearchTunerMixin, HyperModel):
    """Base hypermodel for Cards."""

    IMAGE_SHAPE: Tuple[int, int, int] = (250, 179, 3)

    def __init__(self, name="unknown", tunable=True) -> None:
        """
        Init the base hypermodel, CardModel.

        Args:
            name (str, optional): Name of the hypermodel.
                (defaults is "unknown")
            tunable (bool, optional): If the hypermodel is tunable.
                (defaults is True)
        """
        super().__init__(name=name, tunable=tunable)
        self.callbacks = [
            callbacks.EarlyStopping(monitor='val_accuracy',
                                    min_delta=0.005,
                                    patience=2,
                                    restore_best_weights=True),
            # callbacks.EarlyStopping(monitor='val_loss',
            #                         patience=3,
            #                         mode='min',
            #                         restore_best_weights=True),
        ]

    def build(self,
              hp: HyperParameter,
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
        best_models, best_hparams = (
            self.tuner.get_best_models(num_models=num_models),
            self.tuner.get_best_hyperparameters(num_trials=num_models)
        )
        print(self.tuner.results_summary())
        for idx, (bm, bhp) in enumerate(zip(best_models, best_hparams)):
            # print(bm.summary())
            # print(bm.optimizer.get_config())
            print(bm.name, bhp.values)
            test_loss, test_acc = bm.evaluate(test_ds)
            print('test_loss:', test_loss, "test_accuracy:", test_acc)
            bm.save(os.path.join(MODEL_DIR,
                                 f"{self.name}_{idx + 1}.h5"))

    def fit(self,
            hp: HyperParameter,
            model: models.Model,
            train_ds: tf.data.Dataset,
            *args,
            testing_data: tf.data.Dataset | None = None,
            epochs: int = 30,
            **kwargs) -> Any:
        """
        Train the hypermodel.

        Args:
            hp (HyperParameter): The hyperparameters
            model: `keras.Model` built in the `build()` function
            train_ds (tf.data.Dataset): Training data
            *args: All arguments passed to `Tuner.search`
            epochs (int): epochs (iterations on a dataset).
                (default is 30)
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
        return model.fit(train_ds,
                         epochs=epochs,
                         steps_per_epoch=len(train_ds),
                         **kwargs)

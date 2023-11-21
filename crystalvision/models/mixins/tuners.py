from functools import cached_property
import os
import shutil
from keras_tuner.engine.tuner import maybe_distribute
from keras_tuner.tuners import BayesianOptimization, Hyperband, RandomSearch

from crystalvision.models import MODEL_DIR
import tensorflow as tf


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


class TunerMixin:
    """Base Tuner Mixin."""

    MAX_EXECUTIONS = 1

    def clear_cache(self) -> None:
        """Delete the hypermodel cache."""
        project_dir = os.path.join(MODEL_DIR, self.name)
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)

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


class RandomSearchTunerMixin(TunerMixin):
    """RandomSearch Tuner Mixin."""

    MAX_TRIALS = 20

    @cached_property
    def tuner(self) -> RandomSearch:
        """Random search tuner."""
        return MyRandomSearch(self,
                              objective="val_accuracy",
                              max_trials=self.MAX_TRIALS,
                              executions_per_trial=self.MAX_EXECUTIONS,
                              directory=MODEL_DIR,
                              project_name=self.name)


class HyperbandTunerMixin(TunerMixin):
    """Hyperband Tuner Mixin."""

    @cached_property
    def tuner(self) -> RandomSearch:
        """Hyperband search tuner."""
        return Hyperband(self,
                         objective="val_accuracy",
                         executions_per_trial=self.MAX_EXECUTIONS,
                         directory=MODEL_DIR,
                         project_name=self.name)


class MyBayesianOptimization(BayesianOptimization):
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


class BayesianOptimizationTunerMixin(RandomSearchTunerMixin):
    """Bayesian Optimization Tuner Mixin."""

    @cached_property
    def tuner(self) -> RandomSearch:
        """Bayesian Optimization tuner."""
        return MyBayesianOptimization(self,
                                      objective="val_accuracy",
                                      max_trials=self.MAX_TRIALS,
                                      executions_per_trial=self.MAX_EXECUTIONS,
                                      directory=MODEL_DIR,
                                      project_name=self.name)

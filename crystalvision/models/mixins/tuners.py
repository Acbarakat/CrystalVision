from functools import cached_property
import os
import shutil

from keras_tuner import Objective
from keras_tuner.src.engine.objective import MultiObjective
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

    @cached_property
    def objective(self) -> MultiObjective:
        return MultiObjective(
            [
                Objective("val_accuracy", "max"),
                Objective("test_accuracy", "max"),
                Objective("accuracy", "max"),
                # Objective("val_loss", "min"),
                # Objective("test_loss", "min"),
                # Objective("loss", "min"),
            ]
        )

    def clear_cache(self) -> None:
        """Delete the hypermodel cache."""
        project_dir = os.path.join(MODEL_DIR, self.name)
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)

    def search(self, random_state: int | None = None) -> None:
        """
        Perform a search for best hyperparameter configuations.

        Args:
            train_ds (tf.data.Dataset): training data
            test_ds (tf.data.Dataset): testing data
            validation_ds (tf.data.Dataset): validation data
        """
        tf.keras.backend.clear_session()

        self.tuner.search(
            random_state=random_state,
            callbacks=self.callbacks,
        )


class RandomSearchTunerMixin(TunerMixin):
    """RandomSearch Tuner Mixin."""

    MAX_TRIALS = 20

    @cached_property
    def tuner(self) -> RandomSearch:
        """Random search tuner."""
        return MyRandomSearch(
            self,
            objective=self.objective,
            max_trials=self.MAX_TRIALS,
            executions_per_trial=self.MAX_EXECUTIONS,
            directory=MODEL_DIR,
            project_name=self.name,
        )


class HyperbandTunerMixin(TunerMixin):
    """Hyperband Tuner Mixin."""

    @cached_property
    def tuner(self) -> RandomSearch:
        """Hyperband search tuner."""
        return Hyperband(
            self,
            objective=self.objective,
            executions_per_trial=self.MAX_EXECUTIONS,
            directory=MODEL_DIR,
            project_name=self.name,
        )


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

    def get_best_models(self, num_models=1):
        """Returns the best model(s), as determined by the objective.

        This method is for querying the models trained during the search.
        For best performance, it is recommended to retrain your Model on the
        full dataset using the best hyperparameters found during `search`,
        which can be obtained using `tuner.get_best_hyperparameters()`.

        Args:
            num_models: Optional number of best models to return.
                Defaults to 1.

        Returns:
            List of trained models sorted from the best to the worst.
        """
        best_trials = self.oracle.get_best_trials(num_models)
        models = []

        for trial in best_trials:
            try:
                models.append(self.load_model(trial))
            except Exception as err:
                print(err)
                print(trial)
                tf.keras.backend.clear_session()

        return models

    def run_trial(self, trial, *args, **kwargs):
        result = super().run_trial(trial, *args, **kwargs)

        if isinstance(result, list) and len(result) == 1:
            result = result[0]

        objective = self.oracle.objective
        if isinstance(result, dict) and isinstance(objective, MultiObjective):
            result[objective.name] = objective.get_value(result)

        return result


class BayesianOptimizationTunerMixin(RandomSearchTunerMixin):
    """Bayesian Optimization Tuner Mixin."""

    MAX_CONSECUTIVE_FAILED_TRIALS = 3

    @cached_property
    def tuner(self) -> RandomSearch:
        """Bayesian Optimization tuner."""
        return MyBayesianOptimization(
            self,
            objective=self.objective,
            max_trials=self.MAX_TRIALS,
            executions_per_trial=self.MAX_EXECUTIONS,
            directory=MODEL_DIR,
            project_name=self.name,
            max_consecutive_failed_trials=self.MAX_CONSECUTIVE_FAILED_TRIALS,
        )

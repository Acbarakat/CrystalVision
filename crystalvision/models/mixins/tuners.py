from functools import cached_property
import os
import gc
import shutil

import pandas as pd

from keras import backend
from keras_tuner.src.engine.objective import MultiObjective
from keras_tuner.tuners import BayesianOptimization, Hyperband, RandomSearch

from crystalvision.models import MODEL_DIR


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
        model = super().load_model(trial)
        model.name += f"_trial{trial.trial_id}"
        return model


class TunerMixin:
    """Base Tuner Mixin."""

    MAX_EXECUTIONS = 1
    MAX_CONSECUTIVE_FAILED_TRIALS = 3

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
        backend.clear_session()

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


class MyHyperband(Hyperband):
    """Variation of HyperBand algorithm."""

    def run_trial(self, trial, *args, **kwargs):
        result = super().run_trial(trial, *args, **kwargs)

        if isinstance(result, list) and len(result) == 1:
            result = result[0]

        objective = self.oracle.objective
        if isinstance(result, dict) and isinstance(objective, MultiObjective):
            result[objective.name] = objective.get_value(result)

        return result

    def on_trial_end(self, trial):
        """Called at the end of a trial.

        Args:
            trial: A `Trial` instance.
        """
        super().on_trial_end(trial)

        # Write trial status to trial directory
        trial_id = trial.trial_id
        trial_dir = self.oracle._get_trial_dir(trial_id)

        trials_csv = os.path.join(trial_dir, "..", "trials.csv")
        if not os.path.exists(trials_csv):
            kwargs = {
                "mode": "w+",
                "header": True,
            }
        else:
            kwargs = {
                "mode": "a",
                "header": False,
            }

        data = dict(trial.hyperparameters.values)
        if "tuner/trial_id" not in data:
            data["tuner/trial_id"] = ""
        data.update(
            {
                key: trial.metrics.get_last_value(key)
                for key in trial.metrics.metrics.keys()
            }
        )
        df = pd.DataFrame(data, index=[f"trial_{trial_id}"])
        df.index.name = "name"

        df.to_csv(trials_csv, index=True, **kwargs)


class HyperbandTunerMixin(TunerMixin):
    """Hyperband Tuner Mixin."""

    MAX_TRIALS = 100
    MAX_CONSECUTIVE_FAILED_TRIALS = 5
    FACTOR = 3

    @cached_property
    def tuner(self) -> RandomSearch:
        """Hyperband search tuner."""
        return MyHyperband(
            self,
            objective=self.objective,
            max_epochs=self.MAX_TRIALS,
            factor=self.FACTOR,
            executions_per_trial=self.MAX_EXECUTIONS,
            directory=MODEL_DIR,
            project_name=self.name,
            max_consecutive_failed_trials=self.MAX_CONSECUTIVE_FAILED_TRIALS,
        )


class MyBayesianOptimization(BayesianOptimization):
    """BayesianOptimization tuning with Gaussian process."""

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
        model = super().load_model(trial)
        model.name += f"_trial{trial.trial_id}"
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
                raise err
            finally:
                backend.clear_session()
                gc.collect()

        return models

    def run_trial(self, trial, *args, **kwargs):
        result = super().run_trial(trial, *args, **kwargs)

        if isinstance(result, list) and len(result) == 1:
            result = result[0]

        objective = self.oracle.objective
        if isinstance(result, dict) and isinstance(objective, MultiObjective):
            result[objective.name] = objective.get_value(result)

        return result

    def on_trial_end(self, trial):
        """Called at the end of a trial.

        Args:
            trial: A `Trial` instance.
        """
        super().on_trial_end(trial)

        # Write trial status to trial directory
        trial_id = trial.trial_id
        trial_dir = self.oracle._get_trial_dir(trial_id)

        trials_csv = os.path.join(trial_dir, "..", "trials.csv")
        if not os.path.exists(trials_csv):
            kwargs = {
                "mode": "w+",
                "header": True,
            }
        else:
            kwargs = {
                "mode": "a",
                "header": False,
            }

        data = dict(trial.hyperparameters.values)
        data.update(
            {
                key: trial.metrics.get_last_value(key)
                for key in trial.metrics.metrics.keys()
            }
        )
        df = pd.DataFrame(data, index=[f"trial_{trial_id}"])
        df.index.name = "name"

        df.to_csv(trials_csv, index=True, **kwargs)


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

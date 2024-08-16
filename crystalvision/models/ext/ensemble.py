import warnings
import logging
from functools import cache
from typing import Any, Callable, List, Iterable, Tuple

import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds
from keras import ops, KerasTensor, activations, initializers
from keras.src.layers import Activation, Flatten, Input, Layer
from keras.src.models import Model
from keras.src.saving import get_custom_objects
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.externals.name_estimators import _name_estimators
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


log = logging.getLogger("ensemble")


class MyEnsembleVoteClassifier(EnsembleVoteClassifier):
    """Custom ensemble voting classifier class."""

    def __init__(
        self,
        clfs,
        voting: str = "hard",
        weights: Any | None = None,
        verbose: int = 0,
        use_clones: bool = True,
        fit_base_estimators: bool = False,
        activation: Callable | None = None,
        activation_kwargs: dict | None = None,
        labels: List[str] | None = None,
        threshold: float = 0.95,
    ):
        """
        _summary_

        Args:
            clfs (_type_): _description_
            voting (str, optional): _description_.
                (defaults is "hard")
            weights (Any | None, optional): _description_.
                (defaults is None)
            verbose (int, optional): _description_.
                (defaults is 0)
            use_clones (bool, optional): _description_.
                (defaults is True)
            fit_base_estimators (bool, optional): _description_.
                (defaults is False)
            activation (Callable | None, optional): _description_.
                (defaults is None)
            activation_kwargs (dict | None, optional): _description_.
                (defaults is None)
            labels (List[str] | None, optional): _description_.
                (defaults is None)
        """
        super().__init__(
            clfs, voting, weights, verbose, use_clones, fit_base_estimators
        )
        assert self.voting in (
            "soft",
            "multisoft",
            "hard",
            "multihard",
        ), f"Unknown voting: {self.voting}"
        assert labels is not None, "A list of labels is required"
        self.clfs_ = clfs
        self.activation = activations.linear if activation is None else activation
        self.activation_kwargs = {} if activation_kwargs is None else activation_kwargs
        self.labels = labels
        self.threshold = threshold
        self.set_weights(weights)

    def set_weights(self, weights):
        self.weights = (
            ops.ones(
                len(self.clfs_)
                if self.voting in ("hard", "soft")
                else (len(self.clfs_), len(self.labels))
            )
            if weights is None
            else ops.convert_to_tensor(weights)
        )
        self.weights /= ops.sum(
            self.weights, axis=None if len(self.weights.shape) == 1 else 0
        )

    def transform(self, X):
        """Return class labels or probabilities for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        If `voting='soft'` : array-like = [n_classifiers, n_samples, n_classes]
            Class probabilties calculated by each classifier.
        If `voting='hard'` : array-like = [n_classifiers, n_samples]
            Class labels predicted by each classifier.

        """
        if self.voting in ("soft", "multisoft"):
            return self._predict_probas(X)

        return self._predict(X)

    def _predict(self, X: KerasTensor) -> KerasTensor:
        """Collect results from clf.predict calls."""
        if not self.fit_base_estimators:
            log.debug("Collection predictions from %s", self.clfs_)
            predictions = self._predict_probas(X)

            if predictions.shape[2] == 1:
                raise NotImplementedError(predictions, predictions.shape)
                return self.activation(predictions[0], **self.activation_kwargs)

            if self.voting in ("soft", "hard"):
                predictions = ops.transpose(
                    ops.vectorized_map(lambda x: ops.argmax(x, axis=1), predictions),
                )
            else:
                raise NotImplementedError(predictions, self.voting)

            return self.activation(predictions, **self.activation_kwargs)

        log.debug("Collection predictions from %s w/ fit_base_estimators", self.clfs_)
        return ops.transpose([self.le_.transform(self._predict_probas(X))])

    @cache
    def _predict_probas(self, X, apply_weights: bool = False) -> KerasTensor:
        """Collect results from clf.predict_proba calls."""
        result = ops.stack(clf.predict(X) for clf in self.clfs_)
        if apply_weights:
            result *= (
                ops.reshape(self.weights, (-1, 1, 1))
                if len(self.weights.shape)
                else self.weights
            )

        return result

    def predict(
        self, X: KerasTensor, dtype: str = "", with_Y=False
    ) -> KerasTensor | Tuple[KerasTensor, KerasTensor]:
        """
        Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.

        """
        predictions = self.transform(X)
        if dtype:
            predictions = ops.cast(predictions, dtype=dtype)

        if self.voting == "hard":
            maj = ops.vectorized_map(
                lambda x: ops.argmax(
                    ops.bincount(
                        x, minlength=predictions.shape[1], weights=self.weights
                    )
                ),
                predictions,
            )
        elif self.voting == "soft":
            maj = (
                ops.reshape(self.weights, (-1, 1, 1))
                if len(self.weights.shape) == 1
                else self.weights
            )
            maj = ops.argmax(ops.sum(ops.multiply(predictions, maj), axis=0), axis=1)
            predictions = ops.argmax(ops.transpose(predictions), axis=0)
        elif self.voting == "multisoft":
            maj = (
                ops.reshape(self.weights, (-1, 1, 1))
                if len(self.weights.shape) == 1
                else ops.reshape(self.weights, (-1, 1, len(self.labels)))
            )
            maj = ops.sum(ops.multiply(predictions, maj), axis=0)

            maj = self.activation(maj, **self.activation_kwargs)
            predictions = self.activation(predictions, **self.activation_kwargs)
        else:
            raise NotImplementedError(self.voting)

        if self.fit_base_estimators:
            maj = self.le_.inverse_transform(maj)

        if self.voting in ("soft", "hard"):
            maj = pd.Series(maj.cpu())
            maj.replace(dict(enumerate(self.labels)), inplace=True)
        elif self.voting in ("multisoft",):
            maj = pd.DataFrame(maj.cpu(), columns=self.labels)
        else:
            raise NotImplementedError(self.voting)

        if with_Y:
            return maj, predictions

        return maj

    def score(self, X, y, sample_weight=None, Y=None):
        """
        Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of ``self.predict(X)`` w.r.t. `y`.
        """
        if self.voting in ("soft", "hard"):
            return self._single_score(X, y, sample_weight=sample_weight, Y=Y)
        elif self.voting in ("multisoft", "multihard"):
            return self._multi_score(X, y, sample_weight=sample_weight, Y=Y)

        raise NotImplementedError(self.voting)

    def _multi_score(self, X, y, sample_weight=None, Y=None):
        if Y is None:
            Y = self.predict(X)

        y = ops.convert_to_numpy(y.tolist())
        if y.shape < Y.shape:
            columns_to_keep = Y.columns[Y.shape[1] - y.shape[1] :]
            Y = Y[columns_to_keep]

        assert y.shape == Y.shape, f"{y.shape} does not match {Y.shape}"

        result = {
            "all": ops.divide(
                ops.sum(ops.logical_and(ops.ravel(y), ops.ravel(Y.to_numpy()))),
                ops.sum(ops.ravel(y)),
            )
            .cpu()
            .numpy()
        }

        y = ops.transpose(y)
        for i, label in enumerate(self.labels):
            result[label] = pd.NA
            if (score_denom := ops.sum(y[i])) > 0:
                result[label] = ops.sum(ops.logical_and(y[i], Y[label].to_numpy()))
                result[label] = ops.divide(result[label], score_denom).cpu().numpy()

        return pd.Series(result)

    def _single_score(self, X, y, sample_weight=None, Y=None):
        if Y is not None:
            try:
                return accuracy_score(y, Y, sample_weight=sample_weight)
            except ValueError:
                y = ops.convert_to_numpy(y.tolist())
                return accuracy_score(y, Y, sample_weight=sample_weight)

        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def scores(
        self, X, y, sample_weight=None, with_dataframe=False, with_Y=False
    ) -> pd.Series | Tuple[pd.Series, pd.DataFrame]:
        """
        Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of ``self.predict(X)`` w.r.t. `y`.
        """
        log.info("Predicting result")
        Y, result = self.predict(X, with_Y=True)

        if self.voting in ("hard", "soft"):
            result_df = pd.DataFrame(result.cpu(), columns=[c.name for c in self.clfs_])
            result_df.replace(dict(enumerate(self.labels)), inplace=True)

            result = result_df.apply(
                lambda p: accuracy_score(y, p, sample_weight=sample_weight),
                axis="rows",
                raw=True,
                result_type="reduce",
            )
            result["ensemble"] = self.score(X, y, Y=Y)
        elif self.voting in ("multisoft",):
            indexes = pd.MultiIndex.from_product(
                [
                    [c.name + f"_{i}" for i, c in enumerate(self.clfs_)],
                    range(result.shape[1]),
                ],
                names=["model", "uuid"],
            )
            flattened_result = result.cpu().numpy().reshape(-1, result.shape[2])
            result_df = pd.DataFrame(
                flattened_result,
                index=indexes,
                columns=self.labels,
            )

            result = result_df.groupby("model").apply(
                lambda p: self._multi_score(
                    X, y, sample_weight=sample_weight, Y=p
                ),  # , threshold=threshold[p.name]),
            )
            ensemble = self._multi_score(X, y, sample_weight=sample_weight, Y=Y)
            ensemble.name = "ensemble"
            result = result._append(ensemble)
        else:
            raise NotImplementedError(self.voting)

        result_df.name = "predict"
        result.name = "accuracy"

        if with_Y and with_dataframe:
            return result, result_df, Y
        elif with_Y:
            return result, Y
        elif with_dataframe:
            return result, result_df
        return result

    def generate_model(self) -> Model:
        """Converts/creates our ensemble as a single Model."""
        if self.voting != "hard":
            raise NotImplementedError(f"Cannot generate a '{self.voting}' voting model")

        name = self.clfs_[0].name
        name = name.replace(name.split("_")[-1], "ensemble")
        model_input = Input(shape=self.clfs_[0].input_shape[1:])
        y_models = [model(model_input, training=False) for model in self.clfs_]

        active = self.activation(y_models, **self.activation_kwargs)

        if self.voting in ("hard", "soft"):
            if self.clfs_[0].output_shape[1] == 1:
                flatten = Flatten()(active)
                outputs = HardBinaryVote(vote_weights=self.weights)(flatten)
            else:
                outputs = HardClassVote(vote_weights=self.weights)(active)
        else:
            raise NotImplementedError(self.voting)

        return Model(inputs=model_input, outputs=outputs, name=name)

    def save_model(self, file_path: str) -> None:
        """Save the model to Keras SavedModel."""
        return self.generate_model().save(file_path)

    def fit(self, X, y, sample_weight=None):
        """Learn weight coefficients from training data for each classifier.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights passed as sample_weights to each regressor
            in the regressors list as well as the meta_regressor.
            Raises error if some regressor does not support
            sample_weight in the fit() method.

        Returns
        -------
        self : object

        """
        if isinstance(y, KerasTensor) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError(
                "Multilabel and multi-output classification is not supported."
            )

        if len(self.weights) != len(self.clfs):
            raise ValueError(
                "Number of classifiers and weights must be equal"
                "; got %d weights, %d clfs" % (len(self.weights), len(self.clfs))
            )

        self.le_ = LabelEncoder()
        self.le_.fit(y)
        self.classes_ = self.le_.classes_

        if not self.fit_base_estimators and self.use_clones:
            warnings.warn(
                "fit_base_estimators=False " "enforces use_clones to be `False`"
            )
            self.use_clones = False

        if self.use_clones:
            self.clfs_ = clone(self.clfs)
        else:
            self.clfs_ = self.clfs

        if self.fit_base_estimators:
            log.debug("Fitting %d classifiers...", (len(self.clfs)))

            for clf in self.clfs_:
                if self.verbose > 0:
                    i = self.clfs_.index(clf) + 1
                    log.debug(
                        "Fitting clf%d: %s (%d/%d)",
                        i,
                        _name_estimators((clf,))[0][0],
                        i,
                        len(self.clfs_),
                    )

                if self.verbose > 2:
                    if hasattr(clf, "verbose"):
                        clf.set_params(verbose=self.verbose - 2)

                if self.verbose > 1:
                    print(_name_estimators((clf,))[0][1])

                if sample_weight is None:
                    clf.fit(X, self.le_.transform(y))
                else:
                    clf.fit(X, self.le_.transform(y), sample_weight=sample_weight)
        return self

    def find_activation(self, X, y, method="nelder-mead") -> Any:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)

        def function_to_minimize(threshold):
            newclf = MyEnsembleVoteClassifier(
                voting=self.voting,
                use_clones=False,
                fit_base_estimators=False,
                clfs=self.clfs_,
                weights=self.weights,  # use the new weights
                labels=self.labels,
                activation=self.activation,
                activation_kwargs={"threshold": threshold},
            )

            newclf.fit(X_train, y_train)

            # this is the mean accuracy
            score = newclf.score(X_val, y_val)

            # change accuracy to error so that smaller is better
            score_to_minimize = 1.0 - score

            return score_to_minimize

        threshold = self.activation_kwargs.get("threshold", 0.5)

        results = minimize(
            function_to_minimize,
            threshold,
            bounds=[(0.0, 1.0)],
            method=method,
        )

        self.activation_kwargs["threshold"] = results["x"]

        return results

    @classmethod
    def create_initial_simplex_from_guess(cls, initial_guess, max_val=0.5):
        """
        Create an initial simplex for optimization based on the given initial guess.

        Args:
            initial_guess (array-like): The starting point for the simplex.
            epsilon (float): The perturbation amount used to create the simplex.

        Returns:
            np.ndarray: A simplex where each vertex is perturbed from the initial guess.
        """
        # n = len(initial_guess)  # Number of dimensions
        # simplex = [initial_guess] #, np.diag(np.ones(n)), np.full((n, n), 1 / (n-1))]
        # np.fill_diagonal(simplex[2], 0)
        d = len(initial_guess)  # Number of dimensions

        # Initialize simplex vertices
        simplex = np.zeros((d + 1, d))

        # First vertex is the initial guess
        simplex[0] = initial_guess

        # Create the other d vertices by perturbing each component
        for i in range(d):
            vertex = np.copy(initial_guess)
            perturbation = 1 / (d - 1)

            # Clamp the perturbation to max_val and adjust the remaining elements
            if perturbation > max_val:
                perturbation = max_val

            vertex[i] += perturbation
            remaining_value = 1 - vertex[i]
            remaining_indices = [j for j in range(d) if j != i]

            for j in remaining_indices:
                vertex[j] = min(remaining_value / len(remaining_indices), max_val)
                remaining_value -= vertex[j]

            # Ensure the sum of the row is exactly 1
            simplex[i + 1] = vertex / vertex.sum()

        return simplex

    def find_weights(self, X_train, X_val, y_train, y_val, method="nelder-mead") -> Any:
        newclf = MyEnsembleVoteClassifier(
            voting=self.voting,
            use_clones=False,
            fit_base_estimators=False,
            clfs=self.clfs_,
            weights=None,  # use the new weights
            labels=self.labels,
            activation=self.activation,
            activation_kwargs=self.activation_kwargs,
        )

        def function_to_minimize(weights, prime_label="all"):
            newclf.set_weights(weights)
            newclf.fit(X_train, y_train)

            # this is the mean accuracy
            score = newclf.score(X_val, y_val)
            if self.voting in ("multisoft", "multihard"):
                score = score[prime_label]

            # change accuracy to error so that smaller is better
            score_to_minimize = 1.0 - score

            return score_to_minimize

        contraints = None
        if method not in ("nelder-mead"):
            contraints = {"type": "eq", "fun": lambda w: ops.sum(w) - 1}
        options = {
            "xatol": 1e-6,
            "fatol": 1e-6,
        }

        if len(self.weights.shape) == 1:
            weights = self.weights.cpu().numpy()
            options["initial_simplex"] = self.create_initial_simplex_from_guess(weights)

            results = minimize(
                function_to_minimize,
                weights,
                bounds=Bounds(0.0, 1.0),
                method=method,
                constraints=contraints,
                options=options,
            )

            self.weights = results.x / sum(results.x)

            return results

        results = []
        new_weights = []
        weights = ops.transpose(self.weights).cpu()
        for i, prime_label in enumerate(self.labels):
            new_weight = weights[i].numpy()
            options["initial_simplex"] = self.create_initial_simplex_from_guess(
                new_weight
            )
            try:
                result = minimize(
                    function_to_minimize,
                    new_weight,
                    prime_label,
                    bounds=Bounds(0.0, 1.0),
                    method=method,
                    constraints=contraints,
                    options=options,
                )
                new_weight = result.x / sum(result.x)
            except TypeError as err:
                log.warning("%s is NA (%s)", prime_label, err)
                result = None

            log.info("%s new weight is %s", prime_label, new_weight)
            results.append(result)
            new_weights.append(new_weight)

        self.weights = ops.transpose(ops.convert_to_tensor(new_weights))

        return results


class HardBinaryVote(Layer):
    """Hard Binary Voting Layer."""

    def __init__(
        self,
        *,
        activity_regularizer=None,
        trainable=True,
        dtype: str | None = None,
        autocast=True,
        name: str = "hard_vote",
        vote_weights: Iterable[int] | Iterable[float] | None = None,
        labelcount: int | None = 0,
        **kwargs,
    ):
        super().__init__(
            activity_regularizer=activity_regularizer,
            trainable=trainable,
            dtype=dtype,
            autocast=autocast,
            name=name,
        )

        self.vote_weights = None
        if vote_weights is not None:
            self.vote_weights = ops.convert_to_numpy(vote_weights)
        self.labelcount = labelcount

    def build(self, input_shape):
        super().build(input_shape)

        if isinstance(input_shape, list):
            if isinstance(input_shape[1], tuple):
                input_shape = (len(input_shape), *input_shape[1])
            else:
                input_shape = (len(input_shape), *input_shape)

        vote_weights_init = self.vote_weights
        if vote_weights_init is None:
            vote_weights_init = ops.divide(ops.ones(input_shape[0]), input_shape[0])

        self.vote_weights = self.add_weight(
            shape=(input_shape[0],),
            initializer=initializers.Zeros,
            trainable=self.trainable,
            name="vote_weights",
        )
        self.vote_weights.assign_add(vote_weights_init)

        labelcount_init = self.labelcount
        self.labelcount = self.add_variable(
            shape=(1,),
            initializer=initializers.Zeros,
            dtype="int64",
            trainable=False,
            name="labelcount",
        )
        if labelcount_init:
            self.labelcount.assign_add(labelcount_init)
        else:
            self.labelcount.assign_add(input_shape[2])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vote_weights": ops.convert_to_numpy(self.vote_weights.value).tolist(),
                "labelcount": int(self.labelcount.value[0]),
            }
        )

        return config

    def call(self, inputs: Any) -> KerasTensor | Iterable[KerasTensor]:
        """
        Hard Binary Voting logic.

        The `call()` method may not create state (except in its first
        invocation, wrapping the creation of variables or other resources in
        `tf.init_scope()`).  It is recommended to create state in `__init__()`,
        or the `build()` method that is called automatically before `call()`
        executes the first time.

        Args:
          inputs: Input tensor, or dict/list/tuple of input tensors.
            The first positional `inputs` argument is subject to special rules:
            - `inputs` must be explicitly passed. A layer cannot have zero
              arguments, and `inputs` cannot be provided via the default value
              of a keyword argument.
            - NumPy array or Python scalar values in `inputs` get cast as
              tensors.
            - Keras mask metadata is only collected from `inputs`.
            - Layers are built (`build(input_shape)` method)
              using shape info from `inputs` only.
            - `input_spec` compatibility is only checked against `inputs`.
            - Mixed precision input casting is only applied to `inputs`.
              If a layer has tensor arguments in `*args` or `**kwargs`, their
              casting behavior in mixed precision should be handled manually.
            - The SavedModel input specification is generated using `inputs`
              only.
            - Integration with various ecosystem packages like TFMOT, TFLite,
              TF.js, etc is only supported for `inputs` and not for tensors in
              positional and keyword arguments.

        Returns:
            A tensor or list/tuple of tensors.
        """
        if isinstance(inputs, list):
            inputs = ops.stack(inputs)

        inputs = ops.vectorized_map(
            lambda z: ops.cast(
                ops.bincount(z, minlength=2, weights=self.vote_weights), "uint8"
            ),  # noqa: E501
            ops.transpose(inputs),
        )

        return ops.argmax(inputs, axis=1)


class HardClassVote(HardBinaryVote):
    """Hard Classifer Voting Layer."""

    def call(self, inputs: Any) -> KerasTensor | Iterable[KerasTensor]:
        """
        Hard Classification Voting logic.

        The `call()` method may not create state (except in its first
        invocation, wrapping the creation of variables or other resources in
        `tf.init_scope()`).  It is recommended to create state in `__init__()`,
        or the `build()` method that is called automatically before `call()`
        executes the first time.

        Args:
          inputs: Input tensor, or dict/list/tuple of input tensors.
            The first positional `inputs` argument is subject to special rules:
            - `inputs` must be explicitly passed. A layer cannot have zero
              arguments, and `inputs` cannot be provided via the default value
              of a keyword argument.
            - NumPy array or Python scalar values in `inputs` get cast as
              tensors.
            - Keras mask metadata is only collected from `inputs`.
            - Layers are built (`build(input_shape)` method)
              using shape info from `inputs` only.
            - `input_spec` compatibility is only checked against `inputs`.
            - Mixed precision input casting is only applied to `inputs`.
              If a layer has tensor arguments in `*args` or `**kwargs`, their
              casting behavior in mixed precision should be handled manually.
            - The SavedModel input specification is generated using `inputs`
              only.
            - Integration with various ecosystem packages like TFMOT, TFLite,
              TF.js, etc is only supported for `inputs` and not for tensors in
              positional and keyword arguments.

        Returns:
            A tensor or list/tuple of tensors.
        """
        if isinstance(inputs, list):
            inputs = ops.stack(inputs)

        inputs = ops.vectorized_map(lambda z: ops.argmax(z, axis=1), inputs)

        inputs = ops.vectorized_map(
            lambda z: ops.bincount(
                z, minlength=int(self.labelcount.value), weights=self.vote_weights
            ),
            ops.transpose(inputs),
        )
        return ops.argmax(inputs, axis=1)


# TODO: Find a keras means for dispatch support
# @tf.__internal__.dispatch.add_dispatch_support
def hard_activation(z: Any, threshold: float = 0.5) -> KerasTensor:
    """
    Hard activation function based on a threshold.

    For example:

    >>> a = tf.constant([-.5, -.1, 0.0, .1, .5], dtype = tf.float32)
    >>> b = hard_activation(a)
    >>> b.numpy()
    array([0., 0.,  0.,  0.,  1.], dtype=float32)

    Args:
        z (Any): Input tensor
        threshold (float, optional): z >= threshold
            (default is 0.5)

    Returns:
        A Tensor where all values either 0 or 1 based on the threshold
    """
    return ops.cast(ops.greater_equal(z, threshold), "int32")


get_custom_objects().update(
    {
        "hard_activation": Activation(hard_activation, name="hard_activaiton"),
    }
)

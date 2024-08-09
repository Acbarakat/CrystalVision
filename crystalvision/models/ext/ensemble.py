import warnings
import logging
from typing import Any, Callable, List, Iterable, Tuple

import pandas as pd
from scipy.optimize import minimize
from keras import ops, KerasTensor, activations
from keras.layers import Activation, Flatten, Input, Layer
from keras.models import Model
from keras.saving import get_custom_objects
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.externals.name_estimators import _name_estimators
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
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
        assert self.voting in ("soft", "hard"), f"Unknown voting: {self.voting}"
        self.clfs_ = clfs
        self.activation = activations.linear
        if activation is not None:
            self.activation = activation
        self.activation_kwargs = {}
        if activation_kwargs is not None:
            self.activation_kwargs = activation_kwargs
        self.weights = (
            ops.ones(len(clfs))
            if self.weights is None
            else ops.convert_to_tensor(self.weights)
        )
        self.weights /= ops.sum(self.weights)
        self.labels = labels

    def _predict(self, X: KerasTensor) -> KerasTensor:
        """Collect results from clf.predict calls."""
        if not self.fit_base_estimators:
            log.info("Collection predictions from %s", self.clfs_)
            predictions = ops.stack(clf.predict(X) for clf in self.clfs_)

            if predictions.shape[2] == 1:
                raise NotImplementedError(predictions, predictions.shape)
                return self.activation(predictions[0], **self.activation_kwargs)

            predictions = ops.transpose(
                ops.vectorized_map(lambda x: ops.argmax(x, axis=1), predictions),
            )

            return self.activation(predictions, **self.activation_kwargs)

        log.info("Collection predictions from %s w/ fit_base_estimators", self.clfs_)
        return ops.transpose([self.le_.transform(clf.predict(X)) for clf in self.clfs_])

    def _predict_probas(self, X) -> KerasTensor:
        """Collect results from clf.predict_proba calls."""
        probas = [clf(X, training=False) for clf in self.clfs_]
        probas = [ops.dot(c, w) for c, w in zip(probas, self.weights)]
        probas = ops.divide(ops.sum(probas, axis=0), len(self.clfs_))
        return normalize(probas, axis=1, norm="l1")

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

        if self.voting == "soft":
            raise NotImplementedError("soft")
            print(predictions)
            print(ops.dot(predictions, self.weights))
            maj = ops.argmax(predictions, axis=1)
        elif self.voting == "hard":
            maj = ops.vectorized_map(
                lambda x: ops.argmax(ops.bincount(x, minlength=predictions.shape[1])),
                predictions,
            )

        if self.fit_base_estimators:
            maj = self.le_.inverse_transform(maj)

        maj = pd.Series(maj.cpu())
        if self.labels:
            maj.replace(dict(enumerate(self.labels)), inplace=True)

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
        if Y is not None:
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
        result_df = pd.DataFrame(result.cpu(), columns=[c.name for c in self.clfs_])
        result_df.name = "predict"

        if self.labels:
            result_df.replace(dict(enumerate(self.labels)), inplace=True)

        result = result_df.apply(
            lambda p: accuracy_score(y, p, sample_weight=sample_weight),
            axis="rows",
            raw=True,
            result_type="reduce",
        )
        result["ensemble"] = self.score(X, y, Y=Y)
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
            raise NotImplementedError("Cannot generate a 'soft' voting model")

        name = self.clfs_[0].name.replace("_1", "_ensemble")
        model_input = Input(shape=self.clfs_[0].input_shape[1:])
        y_models = [model(model_input, training=False) for model in self.clfs_]

        active = self.activation(y_models, **self.activation_kwargs)
        if self.clfs_[0].output_shape[1] == 1:
            flatten = Flatten()(active)
            outputs = HardBinaryVote(vote_weights=self.weights)(flatten)
        else:
            outputs = HardClassVote(vote_weights=self.weights)(active)

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

        if self.voting not in ("soft", "hard"):
            raise ValueError(
                "Voting must be 'soft' or 'hard'; got (voting=%r)" % self.voting
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
                    print(
                        "Fitting clf%d: %s (%d/%d)"
                        % (i, _name_estimators((clf,))[0][0], i, len(self.clfs_))
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
            # print(newclf.weights)

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

    def find_weights(self, X_train, X_val, y_train, y_val, method="nelder-mead") -> Any:
        def function_to_minimize(weights):
            newclf = MyEnsembleVoteClassifier(
                voting=self.voting,
                use_clones=False,
                fit_base_estimators=False,
                clfs=self.clfs_,
                weights=weights,  # use the new weights
                labels=self.labels,
                activation=self.activation,
                activation_kwargs=self.activation_kwargs,
            )
            # print(newclf.weights)

            newclf.fit(X_train, y_train)

            # this is the mean accuracy
            score = newclf.score(X_val, y_val)

            # change accuracy to error so that smaller is better
            score_to_minimize = 1.0 - score

            return score_to_minimize

        results = minimize(
            function_to_minimize,
            self.weights.cpu(),
            bounds=[(0, 1)] * len(self.clfs_),
            method=method,
            constraints=({"type": "eq", "fun": lambda w: ops.sum(w) - 1}),
        )

        self.weights = results.x / sum(results.x)

        return results


class HardBinaryVote(Layer):
    """Hard Binary Voting Layer."""

    def __init__(
        self,
        trainable: bool = False,
        name: str = "hard_vote",
        dtype: str | None = None,
        dynamic: bool = False,
        vote_weights: Iterable[int] | Iterable[float] | None = None,
        **kwargs,
    ):
        """
        Initialize the voting layer.

        Args:
            trainable (bool, optional): Whether the layer should be trained,
                i.e. whether its potentially-trainable weights should be
                returned as part of`layer.trainable_weights`.
                (default is False)
            name (str | None, optional): The name of the layer.
                (defaults is "hard_vote")
            dtype (_type_, optional): The dtype of the layer's computations
                and weights. Can also be a `tf.keras.mixed_precision.Policy`,
                which allows the computation and weight dtype to differ.
                Default of `None` means to use
                `tf.keras.mixed_precision.global_policy()`, which is a float32
                policy unless set to different value.
                (defaults is None)
            dynamic (bool, optional): Set this to `True` if your layer should
                only be run eagerly, and should not be used to generate a
                static computation graph. This would be the case for a Tree-RNN
                or a recursive network, for example, or generally for any layer
                that manipulates tensors using Python control flow. If `False`,
                we assume that the layer can safely be used to generate a
                static computation graph.
                (defaults is False)
            vote_weights (Any | None, optional): The inital voting weights.
                If `None`, the default tensor will be filled with 1s.
                (defaults is None)
        """
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.vote_weights = None
        if vote_weights is not None:
            self.vote_weights = ops.convert_to_tensor(vote_weights, name="votes")

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
        inputs = ops.transpose(inputs)

        return ops.vectorized_map(
            lambda z: ops.cast(
                ops.argmax(ops.bincount(z, weights=self.vote_weights)), "int32"
            ),  # noqa: E501
            inputs,
        )


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
        inputs = ops.transpose(ops.cast(ops.argmax(inputs), "int32"))

        return ops.vectorized_map(
            lambda z: ops.cast(
                ops.argmax(ops.bincount(z, weights=self.vote_weights)), "int32"
            ),  # noqa: E501
            inputs,
        )


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

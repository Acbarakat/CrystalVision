import tensorflow as tf
from keras import losses, metrics


from functools import cached_property
from typing import List


class BinaryMixin:
    """Binary compile mixins."""

    LABEL_MODE = "binary"

    @cached_property
    def loss(self) -> losses.BinaryCrossentropy:
        """The Binary loss function."""
        return losses.BinaryCrossentropy()

    @cached_property
    def metrics(self) -> List[metrics.Metric]:
        """A list of Binary Metrics."""
        return [tf.keras.metrics.BinaryAccuracy(name="accuracy")]


class BinaryCrossMixin:
    """Binary compile mixins."""

    LABEL_MODE = "binary"

    @cached_property
    def loss(self) -> losses.BinaryCrossentropy:
        """The Binary Cross loss function."""
        return losses.BinaryCrossentropy()

    @cached_property
    def metrics(self) -> List[metrics.Metric]:
        """A list of Binary Metrics."""
        return [tf.keras.metrics.BinaryCrossentropy(name="accuracy")]


class CategoricalMixin:
    """Categorical compile mixins."""

    LABEL_MODE = "categorical"

    @cached_property
    def loss(self) -> losses.CategoricalCrossentropy:
        """The Categorical loss function."""
        return losses.CategoricalCrossentropy()

    @cached_property
    def metrics(self) -> List[metrics.Metric]:
        """A list of Categorical Metrics."""
        return [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]


class SparseCategoricalMixin:
    """Sparse Categorical compile mixins."""

    LABEL_MODE = "categorical"

    @cached_property
    def loss(self) -> losses.SparseCategoricalCrossentropy:
        """The Sparse Categorical loss function."""
        return losses.SparseCategoricalCrossentropy()

    @cached_property
    def metrics(self) -> List[metrics.Metric]:
        """A list of Sparse Categorical Metrics."""
        return [tf.keras.metrics.SparseCategoricalCrossentropy(name="accuracy")]

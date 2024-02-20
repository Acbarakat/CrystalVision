from functools import cached_property
from typing import List

from keras import losses, metrics, backend


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
        return [metrics.BinaryAccuracy(name="accuracy")]


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
        return [metrics.BinaryCrossentropy(name="accuracy")]


class CategoricalMixin:
    """Categorical compile mixins."""

    LABEL_MODE = "categorical"

    @cached_property
    def loss(self) -> losses.Loss:
        """The Categorical loss function."""
        if backend.backend() == "torch":
            return losses.SparseCategoricalCrossentropy()
        return losses.CategoricalCrossentropy()

    @cached_property
    def metrics(self) -> List[metrics.Metric]:
        """A list of Categorical Metrics."""
        if backend.backend() == "torch":
            return [metrics.SparseCategoricalAccuracy(name="accuracy")]
        return [metrics.CategoricalAccuracy(name="accuracy")]


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
        return [metrics.SparseCategoricalCrossentropy(name="accuracy")]


class OneHotMeanIoUMixin:
    """OneHotMeanIoU compile mixins."""

    LABEL_MODE = "binary"

    @cached_property
    def loss(self) -> losses.BinaryCrossentropy:
        """The Binary Cross loss function."""
        return losses.BinaryCrossentropy()

    @cached_property
    def metrics(self) -> List[metrics.Metric]:
        """A list of OneHotMeanIoUl Metrics."""
        return [metrics.OneHotMeanIoU(num_classes=len(self.labels), name="accuracy")]

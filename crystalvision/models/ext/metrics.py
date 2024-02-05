from typing import Optional, Union, List, Tuple

import tensorflow as tf
from keras import metrics
from keras.dtensor import utils as dtensor_utils


class MyOneHotMeanIoU(metrics.OneHotMeanIoU):
    @dtensor_utils.inject_mesh
    def __init__(
        self,
        num_classes: int,
        threshold: int | float,
        name: str = None,
        dtype: Optional[Union[str, tf.dtypes.DType]] = None,
        ignore_class: Optional[int] = None,
        sparse_y_pred: bool = True,
        axis: int = -1,
    ):
        super().__init__(
            num_classes=num_classes,
            axis=axis,
            name=name,
            dtype=dtype,
            ignore_class=ignore_class,
            # sparse_y_true=False,
            sparse_y_pred=sparse_y_pred,
        )
        self.threshold = threshold

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "threshold": self.threshold,
            "name": self.name,
            "dtype": self._dtype,
            "ignore_class": self.ignore_class,
            "sparse_y_pred": self.sparse_y_pred,
            "axis": self.axis,
        }

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the confusion matrix statistics.

        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can
            be a `Tensor` whose rank is either 0, or the same rank as `y_true`,
            and must be broadcastable to `y_true`.

        Returns:
          Update op.
        """

        # if not self.sparse_y_true:
        #     y_true = tf.argmax(y_true, axis=self.axis)
        # if not self.sparse_y_pred:
        #     y_pred = tf.argmax(y_pred, axis=self.axis)
        y_pred = tf.where(y_pred >= self.threshold, 1.0, y_pred)

        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)

        # Flatten the input if its rank > 1.
        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1])

        if y_true.shape.ndims > 1:
            y_true = tf.reshape(y_true, [-1])

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self._dtype)
            if sample_weight.shape.ndims > 1:
                sample_weight = tf.reshape(sample_weight, [-1])

        if self.ignore_class is not None:
            ignore_class = tf.cast(self.ignore_class, y_true.dtype)
            valid_mask = tf.not_equal(y_true, ignore_class)
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]
            if sample_weight is not None:
                sample_weight = sample_weight[valid_mask]

        # Accumulate the prediction to current confusion matrix.
        current_cm = tf.math.confusion_matrix(
            y_true,
            y_pred,
            self.num_classes,
            weights=sample_weight,
            dtype=self._dtype,
        )
        return self.total_cm.assign_add(current_cm)


class MyOneHotIoU(metrics._IoUBase):
    @dtensor_utils.inject_mesh
    def __init__(
        self,
        target_class_ids: Union[List[int], Tuple[int, ...]],
        threshold: int | float,
        name=None,
        dtype=None,
        ignore_class: Optional[int] = None,
        sparse_y_pred: bool = True,
        axis: int = -1,
    ):
        super().__init__(
            num_classes=len(target_class_ids),
            name=name,
            dtype=dtype,
            ignore_class=ignore_class,
            sparse_y_pred=sparse_y_pred,
            axis=axis,
        )
        self.threshold = threshold
        self.target_class_ids = target_class_ids

    def result(self):
        """Compute the intersection-over-union via the confusion matrix."""
        sum_over_row = tf.cast(tf.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
        sum_over_col = tf.cast(tf.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
        true_positives = tf.cast(
            tf.linalg.tensor_diag_part(self.total_cm), dtype=self._dtype
        )

        # sum_over_row + sum_over_col =
        #     2 * true_positives + false_positives + false_negatives.
        denominator = sum_over_row + sum_over_col - true_positives

        # Only keep the target classes
        # true_positives = tf.gather(true_positives, self.target_class_ids)
        # denominator = tf.gather(denominator, self.target_class_ids)

        # If the denominator is 0, we need to ignore the class.
        num_valid_entries = tf.reduce_sum(
            tf.cast(tf.not_equal(denominator, 0), dtype=self._dtype)
        )

        iou = tf.math.divide_no_nan(true_positives, denominator)

        return tf.math.divide_no_nan(
            tf.reduce_sum(iou, name="mean_iou"), num_valid_entries
        )

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "target_class_ids": self.target_class_ids,
            "threshold": self.threshold,
            "name": self.name,
            "dtype": self._dtype,
            "ignore_class": self.ignore_class,
            "sparse_y_pred": self.sparse_y_pred,
            "axis": self.axis,
        }

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the confusion matrix statistics.

        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can
            be a `Tensor` whose rank is either 0, or the same rank as `y_true`,
            and must be broadcastable to `y_true`.

        Returns:
          Update op.
        """

        # if not self.sparse_y_true:
        #     y_true = tf.argmax(y_true, axis=self.axis)
        # if not self.sparse_y_pred:
        #     y_pred = tf.argmax(y_pred, axis=self.axis)
        y_pred = tf.where(y_pred >= self.threshold, 1.0, y_pred)

        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)

        y_true = tf.gather(y_true, self.target_class_ids, axis=1)
        y_pred = tf.gather(y_pred, self.target_class_ids, axis=1)

        # Flatten the input if its rank > 1.
        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1])

        if y_true.shape.ndims > 1:
            y_true = tf.reshape(y_true, [-1])

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self._dtype)
            if sample_weight.shape.ndims > 1:
                sample_weight = tf.reshape(sample_weight, [-1])

        if self.ignore_class is not None:
            ignore_class = tf.cast(self.ignore_class, y_true.dtype)
            valid_mask = tf.not_equal(y_true, ignore_class)
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]
            if sample_weight is not None:
                sample_weight = sample_weight[valid_mask]

        # Accumulate the prediction to current confusion matrix.
        current_cm = tf.math.confusion_matrix(
            y_true,
            y_pred,
            self.num_classes,
            weights=sample_weight,
            dtype=self._dtype,
        )
        return self.total_cm.assign_add(current_cm)

"""Contains the Identity layer."""

import tensorflow.compat.v2 as tf

from keras.utils.generic_utils import get_custom_objects
from keras.engine.base_layer import Layer
from keras.layers.pooling.max_pooling2d import MaxPooling2D

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.Identity")
class Identity(Layer):
    """Identity layer.

    This layer should be used as a placeholder when no operation is to be
    performed. The layer is argument insensitive, and returns its `inputs`
    argument as output.

    Args:
        name: Optional name for the layer instance.
    """

    def call(self, inputs):
        return tf.nest.map_structure(tf.identity, inputs)


@keras_export("keras.layers.MinPooling2D", "keras.layers.MinPool2D")
class MinPooling2D(MaxPooling2D):
    def call(self, inputs):
        return tf.negative(MaxPooling2D.call(self, tf.negative(inputs)))


@keras_export("keras.layers.Threshold")
class Threshold(Layer):
    def __init__(self, threshold, below_zero=True, **kwargs):
        super(Threshold, self).__init__(name="threshold", **kwargs)
        self.threshold = threshold
        self.below_zero = below_zero

    def call(self, inputs):
        result = tf.keras.activations.relu(
            inputs, threshold=self.threshold, max_value=1.0
        )

        if not self.below_zero:
            # return tf.where(inputs >= self.threshold, 1.0, 0.0)
            return result

        return tf.where(inputs > self.threshold, 1.0, input)


get_custom_objects().update(
    {"Identity": Identity, "MinPooling2D": MinPooling2D, "Threshold": Threshold}
)
"""Contains the Identity layer."""

import tensorflow.compat.v2 as tf

from keras.utils.generic_utils import get_custom_objects
from keras.engine.base_layer import Layer
from keras.layers.pooling.max_pooling2d import MaxPooling2D
from keras import backend as K

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
        return -MaxPooling2D.call(self, -inputs)


@keras_export("keras.layers.Threshold")
class Threshold(Layer):
    def __init__(self, threshold, **kwargs):
        super(Threshold, self).__init__(**kwargs)
        self.threshold = threshold

    def call(self, inputs):
        return K.cast(K.greater_equal(inputs, self.threshold), K.floatx())


get_custom_objects().update(
    {"Identity": Identity, "MinPooling2D": MinPooling2D, "Threshold": Threshold}
)

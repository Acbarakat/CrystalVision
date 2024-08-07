"""Contains the Identity layer."""

# from keras.src.api_export import keras_export
from keras import ops, saving

# from keras.src.utils.generic_utils import get_custom_objects
from keras.src.layers.layer import Layer
from keras.src.layers.pooling.max_pooling2d import MaxPooling2D


# @keras_export("keras.layers.MinPooling2D", "keras.layers.MinPool2D")
@saving.register_keras_serializable(package="MinPooling2D")
class MinPooling2D(MaxPooling2D):
    def call(self, inputs):
        return ops.negative(MaxPooling2D.call(self, ops.negative(inputs)))


# @keras_export("keras.layers.Threshold")
@saving.register_keras_serializable(package="Threshold")
class Threshold(Layer):
    def __init__(self, threshold, below_zero=True, **kwargs):
        super(Threshold, self).__init__(name="threshold", trainable=False, **kwargs)
        self.threshold: float = threshold
        self.below_zero: bool = below_zero

    def call(self, inputs):
        if self.below_zero:
            return ops.where(ops.greater_equal(inputs, self.threshold), 1.0, 0.0)

        return ops.where(ops.greater(inputs, self.threshold), 1.0, inputs)

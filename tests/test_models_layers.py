"""Test custom layers."""

import numpy as np
import tensorflow.compat.v2 as tf

from crystalvision.models.layers import MinPooling2D, Identity


def test_minpooling2d() -> None:
    x = tf.constant([[7.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 8.0, 9.0]])
    x = tf.reshape(x, [1, 3, 3, 1])
    layer = MinPooling2D(pool_size=(2, 2), strides=(1, 1), padding="valid")
    result = layer(x)
    assert result.shape == (1, 2, 2, 1), "TensorShape mismatch"
    assert np.equal(
        result, [[[2.0], [2.0]], [[1.0], [5.0]]]
    ).all(), "Failed to create MinPooling2D"


def test_identity() -> None:
    x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    x = tf.reshape(x, [1, 3, 3, 1])
    layer = Identity()
    result = layer(x)
    assert result.shape == (1, 3, 3, 1), "TensorShape mismatch"
    assert np.equal(result, x).all(), "Failed to create Identity"

"""
Test custom layers.
"""

import pytest
import numpy as np

try:
    from . import backend  # noqa
except ImportError:
    from tests import backend  # noqa


@pytest.mark.parametrize("backend", ["tensorflow", "torch"], indirect=True)
def test_minpooling2d(backend) -> None:
    from keras import ops
    from crystalvision.models.ext.layers import MinPooling2D

    x = ops.convert_to_tensor([[7.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 8.0, 9.0]])
    x = ops.reshape(x, [1, 3, 3, 1])
    layer = MinPooling2D(pool_size=(2, 2), strides=(1, 1), padding="valid")
    result = layer(x).cpu()
    assert result.shape == (1, 2, 2, 1), "TensorShape mismatch"
    assert np.equal(
        result, [[[2.0], [2.0]], [[1.0], [5.0]]]
    ).all(), "Failed to create MinPooling2D"


@pytest.mark.parametrize("backend", ["tensorflow", "torch"], indirect=True)
def test_threshold(backend):
    from keras import ops
    from crystalvision.models.ext.layers import Threshold

    x = ops.convert_to_tensor([[1.1, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    layer = Threshold(4.9)
    result = layer(x).cpu()
    assert result.shape == (3, 3), "TensorShape mismatch"
    assert np.equal(
        result, [[0, 0, 0], [0, 1, 1], [1, 1, 1]]
    ).all(), "Failed to create Threshold"


@pytest.mark.parametrize("backend", ["tensorflow", "torch"], indirect=True)
def test_threshold2(backend):
    from keras import ops
    from crystalvision.models.ext.layers import Threshold

    x = ops.convert_to_tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    layer = Threshold(0.49)
    result = layer(x).cpu()
    assert result.shape == (3, 3), "TensorShape mismatch"
    assert np.equal(
        result, [[0, 0, 0], [0, 1, 1], [1, 1, 1]]
    ).all(), "Failed to create Threshold"


@pytest.mark.parametrize("backend", ["tensorflow", "torch"], indirect=True)
def test_threshold_nonzero(backend):
    from keras import ops
    from crystalvision.models.ext.layers import Threshold

    x = ops.convert_to_tensor([[1.1, 2.0, 3.0], [4.9, 5.0, 6.0], [7.0, 8.0, 9.0]])
    layer = Threshold(4.9, below_zero=False)
    result = layer(x).cpu().numpy()
    assert result.shape == (3, 3), "TensorShape mismatch"
    assert np.equal(
        result,
        np.array(
            [[1.1, 2.0, 3.0], [4.9, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=result.dtype
        ),
    ).all(), "Failed to create Threshold"


@pytest.mark.parametrize("backend", ["tensorflow", "torch"], indirect=True)
def test_threshold_nonzero2(backend):
    from keras import ops
    from crystalvision.models.ext.layers import Threshold

    x = ops.convert_to_tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    layer = Threshold(0.49, below_zero=False)
    result = layer(x).cpu().numpy()
    assert result.shape == (3, 3), "TensorShape mismatch"
    assert np.equal(
        result,
        np.array(
            [[0.1, 0.2, 0.3], [0.4, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=result.dtype
        ),
    ).all(), "Failed to create Threshold"

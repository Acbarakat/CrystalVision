import os

import numpy as np
import pandas as pd
import pytest

try:
    import tensorflow  # noqa: F401
except ModuleNotFoundError:
    os.environ["KERAS_BACKEND"] = "torch"


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


@pytest.mark.parametrize("vote_weights", [[1.0, 0, 0, 0, 0], None])
def test_hard_binary_vote(vote_weights, batch_size=5, samples=83):
    from keras import ops
    from keras.src.random import uniform
    from crystalvision.models.ext.ensemble import HardBinaryVote

    x = [
        ops.cast(
            ops.where(
                ops.greater_equal(uniform((samples,), seed=23 + i), 0.5), 1.0, 0.0
            ),
            "uint8",
        )
        for i in range(5)
    ]

    layer = HardBinaryVote(vote_weights=vote_weights)
    result = layer(x)
    print(result)

    assert result.shape == (samples,), "TensorShape mismatch"
    if vote_weights is not None:
        assert ops.equal(result, x[0]).all(), "Failed to match"


@pytest.mark.parametrize("vote_weights", [[1.0, 0, 0, 0, 0], None])
def test_hard_class_vote(vote_weights, num_classes=4, batch_size=5, samples=83):
    from keras import ops
    from keras.src.random import categorical
    from crystalvision.models.ext.ensemble import HardClassVote

    # Create logits with the same value for all classes to ensure even distribution
    logits = np.full((samples, num_classes), 0.4 / (num_classes - 1))
    for idx in range(logits.shape[0]):
        logits[idx, idx % num_classes] = 0.6

    x = ops.stack(
        ops.bincount(categorical(logits, 1, seed=23 + i), minlength=num_classes)
        for i in range(batch_size)
    )

    layer = HardClassVote(vote_weights=vote_weights)
    result = layer(x)

    assert result.shape == (samples,), "TensorShape mismatch"
    if vote_weights is not None:
        assert ops.equal(
            result, ops.squeeze(ops.transpose(categorical(logits, 1, seed=23)))
        ).all(), "Failed to match"

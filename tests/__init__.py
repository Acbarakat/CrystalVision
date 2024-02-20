import os
import pytest
import importlib


@pytest.fixture(scope="session", params=["tensorflow", "torch"])
def backend(request):
    os.environ["KERAS_BACKEND"] = request.param
    import keras

    importlib.reload(keras)
    assert keras.backend.backend() == request.param, keras.backend.backend()
    yield request.param

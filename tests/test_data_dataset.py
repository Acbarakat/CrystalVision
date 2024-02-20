"""
Test DataSets.
"""
import pytest
import numpy as np

try:
    from . import backend  # noqa
except ImportError:
    from tests import backend  # noqa


@pytest.fixture(scope="module")
def dataset():
    from crystalvision.data.dataset import (
        paths_and_labels_to_dataset,
        imagine_database,
    )

    df = imagine_database("images").head(5)

    elements, uniques = df["element"].factorize(sort=True)

    ds = paths_and_labels_to_dataset(
        image_paths=df["filename"].tolist(),
        image_size=(250, 179),
        labels=elements,
        num_classes=len(uniques),
        num_channels=3,
        label_mode="categorical",
        interpolation="bilinear",
        data_format="channels_last",
    )

    yield ds, df, uniques


@pytest.mark.parametrize("backend", ["tensorflow", "torch"], indirect=True)
def test_paths_and_labels_to_dataset(dataset, backend):
    ds, _, uniques = dataset

    if backend == "torch":
        image, label = ds[4]
    elif backend == "tensorflow":
        image, label = next(iter(ds.take(1)))
        label = np.argmax(label)
    else:
        raise NotImplementedError(backend)

    assert uniques[label] == "火"
    assert image.shape == (250, 179, 3)
    assert len(ds) == 5


@pytest.mark.parametrize("backend", ["tensorflow", "torch"], indirect=True)
@pytest.mark.parametrize("mode", ["additive", "random"])
def test_extend_dataset(mode, dataset, backend):
    from crystalvision.data.dataset import (
        extend_dataset,
    )

    ds, df, _ = dataset

    ds = extend_dataset(
        ds, batch_size=None, name="element", shuffle=False, additive=mode == "additive"
    )

    assert len(ds) == df.shape[0] * (2 if mode != "additive" else 10)

    if backend == "torch":
        image1, label1 = (
            ds.dataset.datasets[0].datasets[0][0]
            if mode == "additive"
            else ds.dataset.datasets[0][0]
        )
        image1 = image1.cpu().numpy()
        image2, label2 = (
            ds.dataset.datasets[1].datasets[0][0]
            if mode == "additive"
            else ds.dataset.datasets[1][0]
        )
        image2 = image2.cpu().numpy()
    elif backend == "tensorflow":
        image1, label1 = next(iter(ds.take(1)))
        label1 = np.argmax(label1)
        image2, label2 = next(iter(ds.skip(4).take(1)))
        label2 = np.argmax(label2)
    else:
        raise NotImplementedError(backend)

    assert np.all((0.0 <= image1) & (image1 <= 1.0))
    assert label1 == label2
    assert not np.array_equal(image1, image2)

import numpy as np

from crystalvision.data.dataset import (
    paths_and_labels_to_dataset,
    extend_dataset,
    imagine_database,
)


def test_paths_and_labels_to_dataset():
    df = imagine_database("images").head(5)

    ds = paths_and_labels_to_dataset(
        image_paths=df["filename"].tolist(), labels=df["element"].tolist()
    )

    image, label = ds[4]
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    assert label == "火"
    assert image.shape == (448, 224, 3)
    assert np.all((0.0 <= image[0][0]) & (image[0][0] <= 1.0))
    assert len(ds) == 5


def test_extend_dataset():
    df = imagine_database("images").head(5)

    ds = paths_and_labels_to_dataset(
        image_paths=df["filename"].tolist(), labels=df["element"].tolist()
    )

    ds = extend_dataset(ds, batch_size=None)

    assert len(ds) == 100

    # print(ds.dataset.datasets[0].datasets[0].datasets[0])
    # print(ds.dataset.datasets[1].datasets[0].datasets[0])

    image1, label1 = ds.dataset.datasets[0].datasets[0].datasets[0][0]
    image2, label2 = ds.dataset.datasets[1].datasets[0].datasets[0][0]

    assert label1 == label2
    assert not np.array_equal(image1, image2)

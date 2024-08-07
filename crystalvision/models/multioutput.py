# -*- coding: utf-8 -*-
"""
Type-based Card Hypermodels.

Todo:
    * N/A

"""
from pandas import DataFrame
from keras import layers, models, optimizers, metrics
from keras_tuner import HyperParameters

try:
    from .base import MultiLabelCardModel
    from .mixins.tuners import HyperbandTunerMixin
except ImportError:
    from crystalvision.models.base import MultiLabelCardModel
    from crystalvision.models.mixins.tuners import HyperbandTunerMixin


class MultiLabel(HyperbandTunerMixin, MultiLabelCardModel):
    """Multilabel for all of a Card's labels."""

    MAX_TRIALS: int = 100
    DEFAULT_EPOCHS: int = 50

    def __init__(self, df: DataFrame, vdf: DataFrame, **kwargs) -> None:
        super().__init__(
            "multilabel",
            df,
            vdf,
            ["type_en", "cost", "element_v2", "power", "icons"],
            ["type_en", "cost", "element", "power", "ex_burst", "multicard"],
            **kwargs
        )

        self.callbacks[0].min_delta = 5e-06
        self.callbacks[0].patience = 10

    def build(self, hp: HyperParameters, seed: int | None = None) -> models.Sequential:
        """
        Build a model.

        Args:
            hp (HyperParameters): A `HyperParameters` instance.

        Returns:
            A model instance.
        """
        batch_size = hp.Choice("batch_size", values=[32, 64])  # noqa

        pl1 = self._pooling2d_choice("pooling1", hp)[1]
        pl2 = self._pooling2d_choice("pooling2", hp, exclude={"Min"})[1]
        pl3 = self._pooling2d_choice("pooling3", hp, exclude={"Min"})[1]

        m = models.Sequential(
            layers=[
                layers.Input(shape=self.IMAGE_SHAPE, batch_size=batch_size),
                layers.Conv2D(
                    64,
                    (3, 3),
                    activation="relu",
                ),
                pl1(pool_size=(2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                pl2(pool_size=(2, 2)),
                layers.Conv2D(128, (3, 3), activation="relu"),
                pl3(pool_size=(2, 2)),
                layers.Conv2D(256, (3, 3), activation="relu"),
                layers.Flatten(),
                layers.Dense(
                    hp.Int(
                        "dense_units1",
                        min_value=650,
                        max_value=850,
                    ),
                    activation="relu",
                ),
                layers.Dense(len(self.labels), activation="sigmoid", name="result"),
            ],
            name=self.name,
        )

        # learning_rate = hp.Float(
        #     "learning_rate", min_value=1.0e-7, max_value=0.01
        # )

        m.compile(
            optimizer=optimizers.Adam(amsgrad=True, name="amsgrad"),
            loss=self.loss,
            metrics=self._metrics
            + [
                metrics.BinaryAccuracy(
                    name="accuracy", threshold=self.one_hot_threshold
                )
            ],
        )

        return m


if __name__ == "__main__":
    from crystalvision.models import tune_model

    tune_model(MultiLabel, clear_cache=True)

# -*- coding: utf-8 -*-
"""
Type-based Card Hypermodels.

Todo:
    * N/A

"""
from pandas import DataFrame
from keras import layers, models, backend
from keras_tuner import HyperParameters

try:
    from .base import MultiLabelCardModel
    from .ext.metrics import MyOneHotMeanIoU
    from .mixins.tuners import HyperbandTunerMixin
except ImportError:
    from crystalvision.models.base import MultiLabelCardModel
    from crystalvision.models.ext.metrics import MyOneHotMeanIoU
    from crystalvision.models.mixins.tuners import HyperbandTunerMixin


class MultiLabel(HyperbandTunerMixin, MultiLabelCardModel):
    """Multilabel for all of a Card's labels."""

    MAX_TRIALS: int = 30
    DEFAULT_EPOCHS: int = 50

    def __init__(self, df: DataFrame, vdf: DataFrame) -> None:
        super().__init__(
            "multilabel",
            df,
            vdf,
            ["type_en", "cost", "element_v2", "power", "icons"],
            ["type_en", "cost", "element", "power", "ex_burst", "multicard"],
        )

    def build(self, hp: HyperParameters, seed: int | None = None) -> models.Sequential:
        """
        Build a model.

        Args:
            hp (HyperParameters): A `HyperParameters` instance.

        Returns:
            A model instance.
        """
        batch_size = hp.Choice("batch_size", values=[32, 64, 128])  # noqa

        pl1 = self._pooling2d_choice("pooling1", hp)[1]
        pl2 = self._pooling2d_choice("pooling2", hp)[1]
        pl3 = self._pooling2d_choice("pooling3", hp)[1]

        m = models.Sequential(
            layers=[
                layers.Input(shape=self.IMAGE_SHAPE, batch_size=batch_size),
                layers.Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                ),
                pl1(pool_size=(2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                pl2(pool_size=(2, 2)),
                layers.Conv2D(128, (3, 3), activation="relu"),
                pl3(pool_size=(2, 2)),
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.Flatten(),
                # layers.Dropout(0.1, seed=seed),
                layers.Dense(
                    hp.Int(
                        "dense_units1",
                        min_value=2**7,
                        max_value=2**12,
                        sampling="LOG",
                    ),
                    activation="relu",
                ),
                layers.Dense(len(self.labels), activation="sigmoid"),
            ],
            name=self.name,
        )

        optimizer = self._optimizer_choice(
            "optimizer",
            hp,
            # exclude={
            #     "Adadelta",
            #     "Adagrad",
            #     "Adam",
            #     "Adamax",
            #     "Ftrl",
            #     "SGD",
            #     "Nesterov",
            #     "Nadam",
            #     "RMSprop",
            # }
        )[1]

        # momentum=hp.Float("mometum", min_value=0.0, max_value=0.9)
        # learning_rate=hp.Float(
        #     "learning_rate", min_value=1.0e-7, max_value=0.9, sampling="LOG"
        # )

        m.compile(
            optimizer=optimizer(),
            loss=self.loss,
            metrics=self._metrics
            + [
                MyOneHotMeanIoU(
                    num_classes=len(self.labels),
                    threshold=0.95,
                    name="accuracy",
                    sparse_y_pred=backend.backend() != "torch",
                ),
            ],
        )

        return m


if __name__ == "__main__":
    from crystalvision.models import tune_model

    tune_model(MultiLabel)

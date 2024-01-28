import numpy as np
from keras import callbacks
from keras.utils import io_utils


from typing import Any


class StopOnValue(callbacks.Callback):
    def __init__(
        self,
        monitor: str ="val_loss",
        monitor_op: Any = np.equal,
        value: float = 0.0,
    ):
        super().__init__()

        self.monitor = monitor
        self.monitor_op = monitor_op
        self.value = value

    def on_epoch_end(self, epoch, logs=None) -> None:
        if logs is None:
            return

        if self.monitor_op(logs[self.monitor], self.value):
            io_utils.print_msg(
                f"\nReached {self.monitor} of {logs[self.monitor]} ({self.monitor_op} {self.value}). Stopping training."
            )
            self.model.stop_training = True
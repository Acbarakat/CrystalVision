from typing import List, Any

import numpy as np
from keras_tuner.src.engine.objective import MultiObjective, Objective


class WeightedMeanMultiObjective(MultiObjective):
    def __init__(
        self,
        objectives: List[Objective],
        weights: List[float] | np.ndarray | None = None,
    ):
        super().__init__(objectives)
        self.weights = weights
        self.direction = "max"

    def get_value(self, logs: Any):
        obj_value = []

        for objective in self.objectives:
            metric_value = logs[objective.name]
            obj_value.append(metric_value)

        obj_value = np.average(obj_value, weights=self.weights)

        return obj_value

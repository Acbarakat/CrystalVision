import pandas as pd
import numpy as np
from paretoset import paretorank

from crystalvision.models import MODEL_DIR

df = pd.read_json(MODEL_DIR / "multilabel_best.json")

acc_df = df[["accuracy", "val_accuracy", "test_accuracy"]]

df["score_mean"] = acc_df.mean(axis=1)
df["score_mean_weighted"] = acc_df.apply(
    lambda x: np.average(x, weights=[0.9, 2.0, 0.1]), axis=1
)
df["rank"] = paretorank(acc_df, sense=["max", "max", "max"])

print(df)

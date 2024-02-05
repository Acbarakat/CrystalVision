import pandas as pd
import numpy as np
from paretoset import paretorank

from crystalvision.models import MODEL_DIR

pd.set_option("display.max_rows", None)
# pd.set_option('display.max_columns', None)


def main(fname):
    df = pd.read_json(MODEL_DIR / fname)

    acc_keys = ["accuracy", "val_accuracy", "test_accuracy"]
    # loss_keys = ["loss", "val_loss", "test_loss"]
    weights = [0.9, 2, 0.1]

    acc_df = df[acc_keys]

    df["score_mean"] = acc_df.mean(axis=1)
    df["score_mean_weighted"] = acc_df.apply(
        lambda x: np.average(x, weights=weights), axis=1
    )
    # df["rank"] = paretorank(df[acc_keys + loss_keys], sense=["max", "max", "max", "min", "min", "min"])
    df["rank"] = paretorank(df[acc_keys], sense=["max", "max", "max"])

    # df["weighted_rank"] = paretorank(df[acc_keys + loss_keys] * [0.9, 1.25, 0.1, 0.09, .125, 0.01], sense=["max", "max", "max", "min", "min", "min"])
    df["weighted_rank"] = paretorank(
        df[acc_keys] * weights, sense=["max", "max", "max"]
    )

    # df.sort_values("score_mean_weighted", ascending=False, inplace=True)

    print(df)


if __name__ == "__main__":
    main("multilabel_best.json")
    # main("icons_best.json")

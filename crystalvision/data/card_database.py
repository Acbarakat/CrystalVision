from crystalvision.data.base import CARD_API_FILEPATH


import pandas as pd


import json


def make_database(clear_extras: bool = False) -> pd.DataFrame:
    """
    Load card data and clean up any issue found in the API.

    Returns:
        Card API DataFrame
    """
    with open(CARD_API_FILEPATH, "r") as fp:
        data = json.load(fp)["cards"]

    df = pd.DataFrame(data)
    # Remove the extra lang columns
    if clear_extras:
        for lang in ("_es", "_de", "_fr", "_it", "_ja"):
            df = df.loc[:, ~df.columns.str.endswith(lang)]

    df["thumbs"] = df["images"].apply(lambda i: [j.split("/")[-1] for j in i["thumbs"]])
    df["images"] = df["images"].apply(lambda i: [j.split("/")[-1] for j in i["full"]])
    df["ex_burst"] = (
        df["ex_burst"].apply(lambda i: i == "\u25cb" or i == "1").astype(bool)
    )
    df["multicard"] = (
        df["multicard"].apply(lambda i: i == "\u25cb" or i == "1").astype(bool)
    )
    df["limit_break"] = df["text_en"].str.contains("Limit Break --")

    icon_labels = ["multicard", "ex_burst", "limit_break"]

    def get_true_column(row):
        true_cols = row[row].index
        return true_cols[0] if len(true_cols) > 0 else pd.NA

    df["icons"] = (
        df[icon_labels]
        .apply(get_true_column, axis=1)
        .astype(pd.CategoricalDtype(categories=icon_labels, ordered=True))
    )

    df["mono"] = df["element"].apply(lambda i: len(i) == 1 if i else True).astype(bool)
    df["element_v2"] = df["element"].apply(
        lambda x: tuple(x) if x is not None else tuple()
    )
    df["element"] = df["element"].str.join("_")

    df["power"] = (
        df["power"].str.replace(" ", "").replace("\u2015", "").replace("\uff0d", "")
    )

    df["power_v2"] = df["power"].astype(
        pd.CategoricalDtype(
            categories=(str(i * 1000) for i in range(1, 11)), ordered=True
        )
    )

    df["cost"] = df["cost"].astype(
        pd.CategoricalDtype(categories=(str(i) for i in range(1, 12)), ordered=True)
    )

    # Bugfixes:
    df.loc[df["code"] == "7-132S", "ex_burst"] = True
    df.loc[df["code"] == "7-132S", "icons"] = "ex_burst"

    return df

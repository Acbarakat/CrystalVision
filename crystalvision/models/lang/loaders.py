import json
from pathlib import Path

try:
    from ...data.dataset import make_database
except (ModuleNotFoundError, ImportError):
    from crystalvision.data.dataset import make_database


kanji_to_english = {
    "火": "Fire",
    "水": "Water",
    "土": "Earth",
    "風": "Wind",
    "雷": "Lightning",
    "氷": "Ice",
    "闇": "Dark",
    "光": "Light",
}


def explain_database():
    with Path(__file__).parent / "df_description.json" as fp:
        description = json.loads(fp.read_bytes())

    df = make_database().drop(
        ["id", "images", "thumbs", "element", "power", "multicard", "mono"], axis=1
    )
    for lang in ("de", "fr", "es", "it", "ja"):
        df.drop(
            [f"name_{lang}", f"text_{lang}", f"job_{lang}", f"type_{lang}"],
            axis=1,
            inplace=True,
        )

    df.rename({"element_v2": "element_ja", "power_v2": "power"}, inplace=True, axis=1)
    df["element"] = df["element_ja"].apply(
        lambda row: {kanji_to_english.get(kanji, kanji) for kanji in row}
    )
    df["numElements"] = df["element"].apply(lambda row: len(row))

    for col in df.columns:
        if col_attrs := description.get(col):
            df[col].attrs.update(col_attrs)

    return df

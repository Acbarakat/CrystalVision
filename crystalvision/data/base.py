# -*- coding: utf-8 -*-
"""
Basis for working with data.

Todo:
    * ???

"""
import os
from pathlib import Path

SRC_DIR = Path(os.path.dirname(__file__), "..").resolve()
DATA_DIR = Path(SRC_DIR, "..", "data").resolve()

CARD_API_FILEPATH = DATA_DIR / "cards.json"
MISSING_CARDS_FILEPATH = SRC_DIR / "missing_cards.json"

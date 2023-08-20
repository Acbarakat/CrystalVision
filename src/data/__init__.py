# -*- coding: utf-8 -*-
"""
Methods to work with test/train/validation data.

Todo:
    * ???

"""
import os

SRC_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(SRC_DIR, "..", "data")

CARD_API_FILEPATH = os.path.join(DATA_DIR, "cards.json")
MISSING_CARDS_FILEPATH = os.path.join(SRC_DIR, "missing_cards.json")

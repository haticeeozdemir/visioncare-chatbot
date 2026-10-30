"""Loads the clinical content files. All clinical decisions live in /data, not in code."""
import json
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LANGS = ("en", "tr", "el")
LEVEL_ORDER = {"INFO": 0, "APPOINTMENT": 1, "SAME_DAY": 2, "EMERGENCY": 3}


def _load(name: str) -> dict:
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


@lru_cache
def questions() -> list[dict]:
    return _load("questions.json")["questions"]


@lru_cache
def rules() -> dict:
    return _load("safety_rules_v0.1.json")


@lru_cache
def messages() -> dict:
    return _load("messages.json")


@lru_cache
def mapping() -> dict:
    return _load("differential_map.json")


@lru_cache
def links() -> dict:
    return _load("official_links.json")


@lru_cache
def cards() -> dict[str, dict]:
    return {c["id"]: c for c in _load("cards.json")["cards"]}


def msg(key: str, lang: str, group: str = "support") -> str:
    entry = messages()[group][key]
    return entry.get(lang) or entry["en"]


def level_msg(level: str, lang: str) -> str:
    return msg(level, lang, group="levels")


def max_level(*levels: str | None) -> str:
    present = [lv for lv in levels if lv]
    return max(present, key=LEVEL_ORDER.__getitem__) if present else "INFO"

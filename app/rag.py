"""Retrieval over approved cards only.

Step 1: multilingual alias match (TR/EN/EL names) - an explicit disease name is the strongest signal.
Step 2: semantic search over card chunks (embeddings.py); top-3 chunks, only if the best cosine score reaches
SCORE_THRESHOLD. Step 3: if still nothing and Claude is available, Claude picks from the card catalog.
No match -> caller returns the fixed 'no approved source' message.
"""
import logging
import re
from functools import lru_cache

from . import data, embeddings, llm
from .keywords import normalize

log = logging.getLogger(__name__)

TOP_K = 3
# multilingual-e5-small scores are compressed: unrelated questions reach ~0.79-0.845, real matches 0.845-0.91.
# The threshold only removes clearly unrelated questions; llm.answer_from_cards answers NO_SOURCE for the rest.
SCORE_THRESHOLD = 0.84
# Chunks scoring more than this below the best chunk are dropped: they only passed the threshold by noise
# ("baş ağrısının sebebi ne": astigmatism 0.874 is real, cataract 0.847 is not).
SCORE_MARGIN = 0.015


@lru_cache
def _card_chunks(cid: str) -> list[dict]:
    return [c for c in embeddings.build_chunks() if c["card_id"] == cid and c["lang"] == "en"]


def _chunks_of(card_ids: list[str]) -> list[dict]:
    return [{**c, "score": None} for cid in card_ids for c in _card_chunks(cid)]


def _alias_cards(question: str) -> list[str]:
    t = normalize(question)
    hits = []
    for cid, card in data.cards().items():
        # alias must start at a word boundary (allows Turkish suffixes: "katarakta", blocks "sharp" -> "rp")
        best = max((len(a) for a in card["aliases"]
                    if re.search(r"(?<!\w)" + re.escape(normalize(a)), t)), default=0)
        if best:
            hits.append((best, cid))
    return [cid for _, cid in sorted(hits, reverse=True)[:2]]


def retrieve_chunks(question: str) -> list[dict]:
    """Chunks (card_id, field, lang, text, score) that may answer the question; [] means no approved source.
    score is None for alias / Claude-picked cards, which contribute the card's full English text."""
    ids = _alias_cards(question)
    if ids:
        return _chunks_of(ids)
    try:
        hits = embeddings.search(question, TOP_K)
        if hits and hits[0][1] >= SCORE_THRESHOLD:
            return [{**c, "score": s} for c, s in hits if s >= hits[0][1] - SCORE_MARGIN]
    except Exception:  # model/index unavailable -> do not fail the question, use the remaining fallback
        log.exception("semantic search failed")
    catalog = {cid: c["title"]["en"] for cid, c in data.cards().items()}
    return _chunks_of(llm.pick_cards(question, catalog) or [])


def retrieve(question: str) -> list[str]:
    """Source card ids, best first."""
    return list(dict.fromkeys(c["card_id"] for c in retrieve_chunks(question)))


def chunks_for_llm(chunks: list[dict]) -> list[str]:
    return [f"[{c['card_id']}/{c['field']}] {c['text']}" for c in chunks]


def chunks_answer(chunks: list[dict], lang: str) -> str:
    """Template-mode answer: the full text of each retrieved chunk's card, in the user's language.
    A single matching chunk (e.g. only "who") is too thin to read on its own."""
    ids = dict.fromkeys(c["card_id"] for c in chunks)
    return "\n\n".join(card_text(cid, lang).split("] ", 1)[1] for cid in ids)


def card_text(cid: str, lang: str = "en") -> str:
    c = data.cards()[cid]
    body = c.get(lang) or c["en"]
    title = c["title"].get(lang) or c["title"]["en"]
    return (f"[{cid}] {title}\n- {body['what']}\n- {body['who']}\n- {body['signs']}\n- {body['help']}"
            + (f"\n- {body['ask']}" if body.get("ask") else ""))


def card_summary(cid: str, lang: str) -> dict:
    c = data.cards()[cid]
    body = c.get(lang) or c["en"]
    return {"id": cid, "title": c["title"].get(lang) or c["title"]["en"], "what": body["what"],
            "when_to_get_help": body["help"], "ask_your_doctor": body.get("ask", ""), "source": c.get("source"),
            "status": "draft - awaiting doctor approval"}

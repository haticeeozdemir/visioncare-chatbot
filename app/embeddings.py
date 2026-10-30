"""Embedding index over approved cards.

Each card is split into chunks (what, who, signs, help, ask) in EN and TR. Chunks are embedded with
multilingual-e5-small ("passage: " prefix; questions use "query: ") and stored in a FAISS inner-product
index over normalized vectors, i.e. cosine similarity. The index is built on first use and cached on disk;
the cache is keyed by a hash of cards.json so edited cards always trigger a rebuild.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np

from . import data

MODEL_NAME = "intfloat/multilingual-e5-small"
CHUNK_FIELDS = ("what", "who", "signs", "help", "ask")
CHUNK_LANGS = ("en", "tr")
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"


@lru_cache
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def _embed(texts: list[str], prefix: str) -> np.ndarray:
    vecs = _model().encode([prefix + t for t in texts], normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


def build_chunks() -> list[dict]:
    chunks = []
    for cid, card in data.cards().items():
        for lang in CHUNK_LANGS:
            body = card.get(lang)
            if not body:
                continue
            title = card["title"].get(lang) or card["title"]["en"]
            for field in CHUNK_FIELDS:
                if body.get(field):
                    # title is part of the text so a chunk like "signs" still knows which condition it is about
                    chunks.append({"card_id": cid, "field": field, "lang": lang,
                                   "text": f"{title}. {body[field]}"})
    return chunks


def _fingerprint() -> str:
    raw = (data.DATA_DIR / "cards.json").read_bytes()
    return hashlib.sha256(raw + MODEL_NAME.encode()).hexdigest()


@lru_cache
def _index() -> tuple[faiss.Index, list[dict]]:
    fp = _fingerprint()
    index_file, meta_file = CACHE_DIR / "cards.faiss", CACHE_DIR / "cards_chunks.json"
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta["fingerprint"] == fp:
            # serialize/deserialize instead of faiss.read_index: faiss cannot open non-ASCII Windows paths
            return faiss.deserialize_index(np.frombuffer(index_file.read_bytes(), dtype="uint8")), meta["chunks"]
    except Exception:  # missing or corrupt cache -> rebuild
        pass

    chunks = build_chunks()
    vecs = _embed([c["text"] for c in chunks], "passage: ")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        index_file.write_bytes(faiss.serialize_index(index).tobytes())
        meta_file.write_text(json.dumps({"fingerprint": fp, "chunks": chunks}, ensure_ascii=False), encoding="utf-8")
    except OSError:  # read-only disk: still works, just rebuilds next start
        pass
    return index, chunks


def search(question: str, k: int = 3) -> list[tuple[dict, float]]:
    """Top-k chunks by cosine similarity, best first."""
    index, chunks = _index()
    scores, ids = index.search(_embed([question], "query: "), min(k, len(chunks)))
    return [(chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]

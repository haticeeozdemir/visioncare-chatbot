"""RAG retrieval layer: chunks, FAISS index + disk cache, semantic search (no API key, template mode).

Only questions whose right card comes first with multilingual-e5-small are asserted here. The model is weak on
some symptom wording (e.g. "düz çizgiler eğri görünüyor" ranks myopia above amd), so those are left out on purpose.
Wording like "göremiyorum" (vision loss) trips the emergency rules first, and "göz tansiyonu" is an alias of
glaucoma, so neither belongs in the semantic-search cases below.
"""
import os

import pytest

os.environ.pop("ANTHROPIC_API_KEY", None)

from fastapi.testclient import TestClient  # noqa: E402

from app import data, embeddings, rag  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


def ask(lang, question):
    return client.post("/api/v1/content/ask", json={"language": lang, "question": question}).json()


def test_chunks_cover_every_card_in_en_and_tr():
    chunks = embeddings.build_chunks()
    for cid in data.cards():
        mine = [c for c in chunks if c["card_id"] == cid]
        assert {c["lang"] for c in mine} == {"en", "tr"}, cid
        assert {"what", "who", "signs", "help"} <= {c["field"] for c in mine}, cid
        assert all(c["text"].startswith(data.cards()[cid]["title"][c["lang"]]) for c in mine)


def test_index_is_cached_on_disk_and_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "CACHE_DIR", tmp_path)
    embeddings._index.cache_clear()
    try:
        index, chunks = embeddings._index()
        assert index.ntotal == len(chunks) == len(embeddings.build_chunks())
        assert (tmp_path / "cards.faiss").exists() and (tmp_path / "cards_chunks.json").exists()

        embeddings._index.cache_clear()
        monkeypatch.setattr(embeddings, "_embed", lambda *a, **k: pytest.fail("cache was not used"))
        index2, chunks2 = embeddings._index()
        assert index2.ntotal == index.ntotal and chunks2 == chunks
    finally:
        monkeypatch.undo()
        embeddings._index.cache_clear()   # later tests go back to the real cache


@pytest.mark.parametrize("lang,question,card", [
    ("tr", "gece farlar gözümü alıyor", "cataract"),
    ("en", "Night headlights glare is blinding me", "cataract"),
    ("tr", "gözlerim farklı yönlere bakıyor", "strabismus"),
])
def test_question_without_disease_name_finds_the_right_card(lang, question, card):
    chunks = rag.retrieve_chunks(question)
    assert 1 <= len(chunks) <= rag.TOP_K
    assert all(c["score"] >= rag.SCORE_THRESHOLD - 0.05 for c in chunks)   # top-3 lie close to the best score
    assert rag.retrieve(question)[0] == card

    r = ask(lang, question)
    assert r["type"] == "answer" and r["mode"] == "template"   # no API key -> retrieved chunks are the answer
    assert r["sources"][0] == card
    assert [c["id"] for c in r["cards"]] == r["sources"]
    assert data.cards()[card]["title"][lang] in r["answer"]
    body = data.cards()[card][lang]
    assert all(body[f] in r["answer"] for f in ("what", "who", "signs", "help"))   # full card, not one thin chunk


def test_weak_chunks_far_below_the_best_one_are_dropped(monkeypatch):
    chunks = embeddings.build_chunks()[:3]
    hits = [(chunks[0], 0.874), (chunks[1], 0.847), (chunks[2], 0.870)]
    monkeypatch.setattr(embeddings, "search", lambda q, k=3: hits)
    got = rag.retrieve_chunks("anything")
    assert [c["text"] for c in got] == [chunks[0]["text"], chunks[2]["text"]]


def test_headache_question_no_longer_returns_unrelated_cards():
    assert rag.retrieve("baş ağrısının sebebi ne") == ["astigmatism"]


def test_alias_match_is_still_used_first():
    chunks = rag.retrieve_chunks("katarakt nedir")
    assert {c["card_id"] for c in chunks} == {"cataract"} and chunks[0]["score"] is None
    assert ask("tr", "katarakt nedir")["sources"] == ["cataract"]


@pytest.mark.parametrize("question", ["how to bake bread", "Τι είναι ο κερατόκωνος;", "python programming tutorial"])
def test_unrelated_question_gets_no_match(question):
    assert rag.retrieve_chunks(question) == []
    assert ask("en", question)["type"] == "no_source"


def test_best_score_below_threshold_returns_no_match(monkeypatch):
    chunk = embeddings.build_chunks()[0]
    monkeypatch.setattr(embeddings, "search", lambda q, k=3: [(chunk, rag.SCORE_THRESHOLD - 0.01)])
    assert rag.retrieve_chunks("anything") == []
    monkeypatch.setattr(embeddings, "search", lambda q, k=3: [(chunk, rag.SCORE_THRESHOLD)])
    assert [c["card_id"] for c in rag.retrieve_chunks("anything")] == [chunk["card_id"]]


def test_search_failure_falls_back_instead_of_crashing(monkeypatch):
    def boom(q, k=3):
        raise RuntimeError("model unavailable")
    monkeypatch.setattr(embeddings, "search", boom)
    assert rag.retrieve_chunks("how to bake bread") == []          # no alias, no API key
    assert rag.retrieve("What is glaucoma?") == ["glaucoma"]       # alias fallback still works


def test_chunks_are_passed_to_the_llm_and_no_source_is_respected(monkeypatch):
    seen = {}

    def fake_answer(question, lang, texts):
        seen["texts"] = texts
        return "NO_SOURCE"
    monkeypatch.setattr("app.llm.answer_from_cards", fake_answer)
    r = ask("tr", "gece farlar gözümü alıyor")
    assert r["type"] == "no_source"                                  # the LLM gate wins over the retrieval hit
    assert seen["texts"] and all(t.startswith("[") for t in seen["texts"])
    assert any(t.startswith("[cataract/") for t in seen["texts"])

    monkeypatch.setattr("app.llm.answer_from_cards", lambda q, l, t: "An eye doctor can check this.")
    r = ask("tr", "gece farlar gözümü alıyor")
    assert r["type"] == "answer" and r["mode"] == "llm" and r["sources"][0] == "cataract"

"""VisionCare chatbot API (FastAPI). Endpoint names follow the project doc section 12;
/content/ask is an addition for RAG question answering (to agree with the VisionCare team)."""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import data, flow, guard, keywords, llm, rag, rules

app = FastAPI(title="VisionCare Chatbot API", version="0.1.0-demo")
STATIC = Path(__file__).resolve().parent.parent / "static"


class NewSession(BaseModel):
    language: str = Field("en", pattern="^(en|tr|el)$")
    country: str = Field("TR", pattern="^(TR|GR)$")


class Answer(BaseModel):
    question_id: str
    answer_codes: list[str] | None = None
    free_text: str | None = Field(None, max_length=1000)


class Ask(BaseModel):
    question: str = Field(..., max_length=1000)
    language: str = Field("en", pattern="^(en|tr|el)$")
    session_id: str | None = None


def _session(sid: str) -> flow.Session:
    s = flow.SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "session not found")
    return s


def _step(s: flow.Session) -> dict:
    if s.status != "in_progress":
        return {"session_id": s.id, "status": s.status, "result": flow.result(s)}
    q = flow.next_question(s)
    return {"session_id": s.id, "status": s.status, "notice": s.notice,
            "question": flow.render_question(q, s), "llm_enabled": llm.enabled()}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/v1/health")
def health():
    return {"ok": True, "llm_enabled": llm.enabled(), "model": llm.MODEL if llm.enabled() else None,
            "rule_version": rules.version()}


@app.post("/api/v1/assessment/sessions")
def create_session(body: NewSession):
    s = flow.Session(language=body.language, country=body.country)
    flow.SESSIONS[s.id] = s
    return {**_step(s), "intro": data.msg("start", s.language)}


@app.get("/api/v1/assessment/questions/{age_group}")
def questions_for(age_group: str, language: str = "en"):
    s = flow.Session(language=language)
    s.state["age_group"] = age_group
    return [flow.render_question(q, s) for q in data.questions()]


@app.post("/api/v1/assessment/sessions/{sid}/answers")
def answer(sid: str, body: Answer):
    s = _session(sid)
    if s.status != "in_progress":
        return _step(s)
    flow.apply_answer(s, body.question_id, body.answer_codes, body.free_text)
    return _step(s)


@app.post("/api/v1/assessment/sessions/{sid}/evaluate")
def evaluate(sid: str):
    s = _session(sid)
    if s.status == "in_progress":
        s.fired = rules.evaluate(s.state)
        s.status = "emergency" if s.fired and s.fired[0]["level"] == "EMERGENCY" else "completed"
    return flow.result(s)


@app.get("/api/v1/content")
def content(language: str = "en", topic: str | None = None):
    ids = [topic] if topic else list(data.cards())
    return [rag.card_summary(c, language) for c in ids if c in data.cards()]


@app.post("/api/v1/content/ask")
def ask(body: Ask):
    lang = body.language
    # 1. red flags in the question itself always win
    fired = rules.evaluate({**keywords.parse(body.question), "age_group": None})
    if fired and fired[0]["level"] == "EMERGENCY":
        return {"type": "emergency", "answer": data.level_msg("EMERGENCY", lang), "fired_rules": [r["id"] for r in fired]}
    # 2. hard limits: no diagnosis, no medicine, no false reassurance
    blocked = guard.intent(body.question)
    if blocked:
        out = {"type": "refusal", "answer": data.msg(blocked, lang)}
        s = flow.SESSIONS.get(body.session_id or "")
        if s and s.status != "in_progress":
            out["reminder"] = flow.result(s).get("message")
        return out
    # 3. RAG over approved cards only
    chunks = rag.retrieve_chunks(body.question)
    if not chunks:
        return {"type": "no_source", "answer": data.msg("no_source", lang)}
    ids = list(dict.fromkeys(c["card_id"] for c in chunks))
    generated = llm.answer_from_cards(body.question, lang, rag.chunks_for_llm(chunks))
    if generated == "NO_SOURCE":   # the model judged the retrieved chunks do not answer the question
        return {"type": "no_source", "answer": data.msg("no_source", lang)}
    if generated and guard.output_ok(generated):
        answer_text, mode = generated, "llm"
    else:
        answer_text, mode = rag.chunks_answer(chunks, lang), "template"
    return {"type": "answer", "answer": answer_text, "mode": mode, "sources": ids,
            "cards": [rag.card_summary(c, lang) for c in ids], "disclaimer": data.level_msg("INFO", lang)}


@app.get("/api/v1/official-appointments")
def official(country: str = "TR"):
    if country not in data.links():
        raise HTTPException(404, "unknown country")
    return data.links()[country]

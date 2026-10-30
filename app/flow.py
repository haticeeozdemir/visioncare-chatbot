"""Assessment session: walks the question tree, runs the safety check after every answer, builds the result."""
import uuid
from dataclasses import dataclass, field

from . import data, keywords, llm, rag, rules

FIELDS = ["age_group", "main_concern", "onset", "duration", "course", "associated", "history"]
CHILD = {"preschool", "school_age"}


@dataclass
class Session:
    language: str = "en"
    country: str = "TR"
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: dict = field(default_factory=lambda: {"associated": [], "history": []})
    answered: set = field(default_factory=set)   # question ids answered or skipped
    skipped: set = field(default_factory=set)
    status: str = "in_progress"                  # in_progress | emergency | completed
    fired: list = field(default_factory=list)
    notice: str | None = None                    # e.g. "please pick an option"


SESSIONS: dict[str, Session] = {}   # in memory only; no names or IDs collected; free text is never stored


def _question(qid: str) -> dict:
    return next(q for q in data.questions() if q["id"] == qid)


def render_question(q: dict, s: Session) -> dict:
    lang = s.language
    is_child = s.state.get("age_group") in CHILD
    text = (q.get("parent_text") if is_child and q.get("parent_text") else q["text"])
    opts = q["options"] + (q.get("child_only_options", []) if is_child else [])
    return {
        "id": q["id"], "field": q["field"], "type": q["type"], "skippable": q["skippable"],
        "text": text.get(lang) or text["en"],
        "options": [{"code": o["code"], "label": o["label"].get(lang) or o["label"]["en"]} for o in opts],
    }


def next_question(s: Session) -> dict | None:
    for q in data.questions():
        if q["id"] in s.answered:
            continue
        value = s.state.get(q["field"])
        if value not in (None, []):            # already filled from free text
            s.answered.add(q["id"])
            continue
        return q
    return None


def _merge(target: dict, part: dict) -> None:
    for k, v in part.items():
        if k in ("associated", "history"):
            target[k] = sorted(set(target.get(k, [])) | set(v or []))
        elif v and not target.get(k):
            target[k] = v


def apply_answer(s: Session, question_id: str, codes: list[str] | None, free_text: str | None) -> None:
    q = _question(question_id)
    s.notice = None
    before = {k: (list(v) if isinstance(v, list) else v) for k, v in s.state.items()}

    if codes:
        valid = {o["code"] for o in q["options"] + q.get("child_only_options", [])}
        codes = [c for c in codes if c in valid]
        if q["type"] == "multi":
            _merge(s.state, {q["field"]: [c for c in codes if c not in ("none", "prefer_not")]})
        elif codes:
            s.state[q["field"]] = codes[0]

    if free_text:
        kw = keywords.parse(free_text)
        ai = llm.extract(free_text, q["field"]) or {}
        _merge(s.state, ai)
        _merge(s.state, kw)
        # safety net: rules also run on the keyword view alone, so an LLM miss cannot hide a red flag
        kw_view = {**s.state, **{k: v for k, v in kw.items() if v}}
        s.fired = rules.evaluate(kw_view)

    filled = s.state.get(q["field"]) not in (None, []) or (codes and q["type"] == "multi")
    if filled:
        s.answered.add(q["id"])
    elif not codes and not free_text and q["skippable"]:
        s.answered.add(q["id"])
        s.skipped.add(q["id"])
    elif free_text and s.state == before:
        s.notice = data.msg("confirm_unclear", s.language)

    fired = rules.evaluate(s.state)
    s.fired = sorted({r["id"]: r for r in fired + s.fired}.values(), key=lambda r: -data.LEVEL_ORDER[r["level"]])
    if s.fired and s.fired[0]["level"] == "EMERGENCY":
        s.status = "emergency"
    elif next_question(s) is None:
        s.status = "completed"


def _match_row(row: dict, st: dict) -> bool:
    for key in ("age_group", "main_concern", "onset"):
        if key in row and st.get(key) not in row[key]:
            return False
    if "history_has" in row and not set(row["history_has"]) & set(st.get("history", [])):
        return False
    return True


def result(s: Session) -> dict:
    lang, st = s.language, s.state
    rule_level = s.fired[0]["level"] if s.fired else None
    base = {"session_id": s.id, "status": s.status,
            "safety": {"rule_version": rules.version(), "fired_rules": [r["id"] for r in s.fired]}}

    if s.status == "emergency":
        link = data.links()[s.country]
        base["safety"]["urgency_level"] = "EMERGENCY"
        base["message"] = data.level_msg("EMERGENCY", lang)
        base["next_step"] = {"type": "call_emergency", "number": link["emergency"]["number"]}
        return base   # no disease information, no extra questions in an emergency

    row = next((r for r in data.mapping()["rows"] if _match_row(r, st)), None)
    has_symptom = bool(st.get("main_concern"))
    level = data.max_level(rule_level, row["level"] if row else None, "APPOINTMENT" if has_symptom else "INFO")
    card_ids = (row["cards"] if row else [])[: data.mapping()["max_cards"]]
    cards = [rag.card_summary(c, lang) for c in card_ids]

    base["summary"] = {k: st.get(k) for k in FIELDS if st.get(k) not in (None, [])}
    base["safety"]["urgency_level"] = level
    base["message"] = data.level_msg(level, lang)
    base["information"] = {
        "heading": data.msg("topics_heading", lang) if cards else data.msg("no_match", lang),
        "cards": cards, "disclaimer": "This is not a diagnosis.",
    }
    if "Q5" in s.skipped or level in ("APPOINTMENT", "SAME_DAY"):
        base["warning_signs"] = data.msg("warning_signs", lang)
    base["next_step"] = {
        "type": "find_eye_clinic", "country": s.country,
        "appointment_links": data.links()[s.country]["appointments"],
        "emergency_number": data.links()[s.country]["emergency"]["number"],
        "questions_for_doctor": [c["ask_your_doctor"] for c in cards if c["ask_your_doctor"]],
    }
    return base

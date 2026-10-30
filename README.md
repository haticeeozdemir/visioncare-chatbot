# VisionCare Chatbot (hackathon demo)

AI chatbot that VisionCare calls over an API. It asks 6 questions, catches emergency signs with fixed rules,
shows the next step, and answers eye-health questions only from approved cards.

> DEMO. All clinical content in `data/` is a draft from the Clinical Review Pack and still needs doctor approval.
> Synthetic data only.

## Run

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # optional; without it the app runs in "keyword mode"
uvicorn app.main:app --reload
# open http://localhost:8000
python -m pytest -q                  # 20 scenarios from the Review Pack
```

## How it works

| Part | File | Role |
| --- | --- | --- |
| Question tree | `data/questions.json` | 6 questions (Q4 split into duration + course), TR/EN/EL, child options |
| Safety rules | `data/safety_rules_v0.1.json` | R01–R07, versioned. Checked after every answer. The LLM cannot override it |
| Fixed messages | `data/messages.json` | Urgency messages and refusals, never rewritten by the LLM |
| Mapping | `data/differential_map.json` | Symptoms -> max 3 cards; no list if no match |
| Cards | `data/cards.json` | 16 cards (RAG source) |
| Links | `data/official_links.json` | 112, MHRS, finddoctors |
| Claude | `app/llm.py` | Free text -> answer codes (forced tool call); card explanations in TR/EN/EL |
| Keyword net | `app/keywords.py` | Always runs next to Claude; red flags are caught even if Claude misses them |
| Guard | `app/guard.py` | Blocks diagnosis / medicine / reassurance requests and unsafe model output |

Model: `claude-sonnet-5-5` (change with `ANTHROPIC_MODEL`).

## API (project doc §12)

- `POST /api/v1/assessment/sessions` `{language, country}`
- `POST /api/v1/assessment/sessions/{id}/answers` `{question_id, answer_codes?, free_text?}`
- `POST /api/v1/assessment/sessions/{id}/evaluate`
- `GET  /api/v1/assessment/questions/{age_group}`
- `GET  /api/v1/content?language=&topic=`
- `POST /api/v1/content/ask` `{question, language, session_id?}` — **new, to agree with the VisionCare team**
- `GET  /api/v1/official-appointments?country=TR|GR`

## Privacy

No names or IDs collected. Sessions live in memory only. Free text is never stored or logged.

## Known limits / to do

- Doctor approval of everything in `data/`.
- Retrieval uses multilingual aliases + Claude picking cards; multilingual-e5 embeddings can replace it later.
- The keyword parser ignores negation ("no pain") — it over-triages on purpose.
- Greek texts are drafts for the Greek team.

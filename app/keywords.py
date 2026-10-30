"""Keyword parser for TR/EN/EL free text.

Two jobs:
1. Safety net: always runs next to the LLM, so a red flag is caught even if the LLM misses it.
2. Fallback: lets the demo run without an API key.
It may over-detect (e.g. ignores negation). For safety, over-triage is preferred to under-triage.
"""
import re
import unicodedata


def normalize(text: str) -> str:
    text = text.replace("İ", "i").replace("I", "ı").lower().replace("ı", "i")
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


# Patterns are written in normalized form (no accents, Turkish ı -> i, Greek without tonos).
P = {
    "trauma_chemical": r"injur|\bhit\b|chemical|bleach|cleaning liquid|splash|yaralan|darbe|carpt|kimyasal|camasir suyu|deterjan|sicra|χημικ|χτυπ|τραυμ",
    "sudden_loss": r"can'?t see|cannot see|lost (my )?(sight|vision)|vision loss|went dark|goremiyorum|goremiyor|gorme kayb|kor oldum|karard|δεν βλεπω|απωλεια ορασης|εχασα την ορασ",
    "curtain": r"curtain|shadow|perde|golge|κουρτιν|σκια",
    "flashes": r"flash|isik cakma|cakmalar|cakiyor|λαμψ|φλας",
    "floaters": r"floater|ucusma|sinek|μυγακ|μυγες|στιγματα",
    "double": r"double|cift gor|ikili gor|διπλ",
    "severe_pain": r"severe|terrible pain|unbearable|siddetli|cok agri|dayanilmaz|εντον|αφορητ",
    "pain": r"pain|hurt|(?<!head)ache|(?<!bas )agri|aciy|sizla|πον",
    "red": r"red eye|redness|bloodshot|kizar|κοκκιν",
    "white_pupil": r"white (glow|reflect|reflex|spot)|beyaz (yansima|parlama|isik|leke)|λευκ",
    "eye_turn": r"turn(s|ing|ed)? (in|out)|crossed|cross-eyed|kay(iyor|ma|maya|di)|sasi|στραβ|αλληθωρ",
    "behaviour_signs": r"covers? one eye|close to (the )?(tv|screen)|gozunu kapat|yakin otur|gozlerini kis|ekrana yakin|κλεινει το ενα ματι",
    "blurred": r"blur|fuzzy|hazy|can'?t read|cannot read|hard to read|bulanik|net gor|okuyam|zor oku|bulaniklik|θολ|δυσκολευομαι να διαβασ",
    "vision_change": r"blur|vision|see|gorme|gor|bulanik|ορασ|βλεπ|θολ",
    "one_eye": r"(left|right|one) eye|(sol|sag|tek) goz|αριστερ|δεξ|ενα ματι",
    "diabetes": r"diabet|seker hasta|diyabet|διαβητ|ζαχαρ",
    "high_myopia": r"very short.?sighted|high myopia|yuksek miyop|cok miyop|υψηλη μυωπια",
}
ONSET = [
    ("sudden", r"sudden|this morning|since morning|an hour|minutes ago|aniden|birden|bu sabah|saattir|ξαφνικ|σημερα το πρωι"),
    ("days", r"yesterday|few days|\d+ days|days ago|dun|gun once|gundur|χθες|μερες|ημερες"),
    ("gradual", r"slowly|gradual|month|over time|getting worse|yavas|aylar|aydir|giderek|σταδιακ|μηνες|μηνα"),
    ("longstanding", r"years|long time|yillar|yildir|uzun sure|χρονια"),
]
DURATION = [
    ("lt_24h", r"hours?\b|this morning|saat|bu sabah|ωρες|ωρα"),
    ("1_7d", r"yesterday|days?\b|dun|gun|χθες|μερες|ημερες"),
    ("1_4w", r"weeks?\b|hafta|εβδομαδ"),
    ("gt_1m", r"months?\b|years?\b|\bay\b|aydir|aylar|yil|μην|χρον"),
]
MAIN_PRIORITY = ["trauma_chemical", "sudden_loss", "curtain", "flashes_floaters", "white_pupil",
                 "eye_turn", "double", "pain", "red", "behaviour_signs", "blurred"]


def _has(key: str, t: str) -> bool:
    return re.search(P[key], t) is not None


def parse(text: str) -> dict:
    """Return a partial state: main_concern, onset, duration, associated[], history[]."""
    t = normalize(text)
    found = {k for k in P if _has(k, t)}
    out: dict = {"associated": [], "history": []}

    concerns = set(found) & set(MAIN_PRIORITY)
    if "flashes" in found or "floaters" in found:
        concerns.add("flashes_floaters")
    for c in MAIN_PRIORITY:
        if c in concerns:
            out["main_concern"] = c
            break

    for onset, pattern in ONSET:
        if re.search(pattern, t):
            out["onset"] = onset
            break
    for dur, pattern in DURATION:
        if re.search(pattern, t):
            out["duration"] = dur
            break

    assoc_map = {"flashes": "new_flashes", "floaters": "new_floaters", "severe_pain": "severe_pain",
                 "pain": "pain", "red": "redness", "trauma_chemical": "recent_injury", "curtain": "curtain",
                 "white_pupil": "white_pupil", "one_eye": "one_eye"}
    out["associated"] = sorted({v for k, v in assoc_map.items() if k in found})
    if "vision_change" in found and ("pain" in found or "severe_pain" in found):
        out["associated"].append("vision_change")
    out["history"] = [h for h in ("diabetes", "high_myopia") if h in found]
    return out

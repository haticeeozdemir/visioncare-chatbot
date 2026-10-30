"""Input intent guard and output safety filter (hard limits from the project brief)."""
import re

from .keywords import normalize

INTENTS = {
    "no_diagnosis": r"which disease|what do i have|what('s| is) wrong with me|diagnos|chance|percent|%|probabilit|likelihood|"
                    r"hangi hastalik|teshis|tani koy|ihtimal|yuzde|olasilik|ne hastaligi|ποια ασθενεια|διαγνωσ|πιθανοτητ|ποσοστ",
    "no_medicine": r"medicine|medication|\bdrug|drops|pill|treat|cure|ilac|damla|tedavi|\bhap\b|merhem|φαρμακ|κολλυρι|θεραπει",
    "no_reassurance": r"nothing to worry|should i worry|is it serious|is this serious|am i ok|is it bad|"
                      r"endiselen|ciddi mi|onemli mi|korkmali|kotu bir sey mi|ανησυχ|σοβαρ",
}

BLOCKED_OUTPUT = re.compile(
    r"\byou (probably |likely |may )?have\b|\d+\s*%|percent|don'?t worry|do not worry|nothing (serious|to worry)|"
    r"endiselenmeyin|merak etmeyin|ciddi bir sey yok|sizde .{0,30} var|μην ανησυχ|\bmg\b|eye ?drops|damla",
    re.IGNORECASE,
)


def intent(question: str) -> str | None:
    t = normalize(question)
    for key, pattern in INTENTS.items():
        if re.search(pattern, t):
            return key
    return None


def output_ok(text: str) -> bool:
    return not BLOCKED_OUTPUT.search(normalize(text)) and not BLOCKED_OUTPUT.search(text)

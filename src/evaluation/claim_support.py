"""Lexical claim-support check for the answer gate.

The NLI model (trained on MNLI/SNLI/FEVER) scores correct procurement answers as contradictions (e.g. "Manual
Credit Review is performed by the Vendor Risk team" vs a chunk saying exactly that), and on a labeled set it passes
correct and wrong answers at the same rate (reports/gate_labeled_eval.json). This check asks narrower questions
that fail in the right places:

- every NUMBER in the sentence must appear in the retrieved chunks with the same unit near it ('5 business days';
  a table header counts), not just anywhere; list markers, section and step numbers are ignored -- a changed
  number is the most common and most costly in-domain hallucination in a policy assistant;
- every KEY TERM (capitalised name/team/role mid-sentence, frequency word) must appear in the chunks -- a swapped
  team, role or cadence changes one of these;
- at least `r` of the sentence's CONTENT words (stopwords removed, light stemming) must appear in the chunks --
  an answer about something the evidence does not cover fails this.

It does not understand negation or polarity: "Yes." vs "No.", "included" vs "excluded" can pass. That limit is
measured in reports/gate_labeled_eval.json, not assumed away.
"""
from __future__ import annotations

import re

_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
          "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fourteen": 14, "fifteen": 15, "twenty": 20}
_STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been being this that these
those it its it's their there which who whom what when where why how than then so such not no yes do does did done can
could should would will must may might shall has have had any all each every other only also more most less least
very into over under up down out about after before above below between during through while within without per via
i you we they he she them our your his her my me us case cases order orders policy section since because""".split())
_FREQ = {"daily", "weekly", "monthly", "quarterly", "annual", "annually", "yearly", "biweekly"}
_UNICODE = {" ": " ", " ": " ", "‑": "-", "‐": "-", "–": "-", "’": "'"}
_LIST_MARKER = re.compile(r"(^|\n)\s*[-*]?\s*\d+[.)]\s")
_REFERENCE = re.compile(r"\b(section|step|level)\s+\d+(\.\d+)?", re.IGNORECASE)
_CLAUSE_START = re.compile(r"(^|[\n:;(]|[-*]\s|\d[.)]\s)\s*[A-Z]")


def split_sentences(text: str) -> list[str]:
    """Same sentence rule the NLI gate uses (sentences of 20+ characters)."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 20]


def _clean(text: str) -> str:
    for a, b in _UNICODE.items():
        text = text.replace(a, b)
    return re.sub(r"[*_`]", "", text)


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def content_words(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z][a-z\-]+", _clean(text).lower()) if w not in _STOP and len(w) > 2}


def _tokens(text: str) -> list[str]:
    """Lowercased tokens, number words -> digits, thousands separators removed. Markdown heading lines, list markers
    and section/step references are dropped so they never count as a claim or as evidence for one."""
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    text = _LIST_MARKER.sub("\n", text)
    text = _REFERENCE.sub(lambda m: m.group(1), text)
    toks = re.findall(r"\d[\d,]*|[a-z][a-z\-]+", _clean(text).lower())
    return [str(_WORDS[t]) if t in _WORDS else t.replace(",", "") for t in toks]


def _number_claims(tokens: list[str]) -> set[tuple[str, str]]:
    """(number, next content word) pairs: '5 business days' -> ('5', 'business')."""
    out = set()
    for i, t in enumerate(tokens):
        if t.isdigit():
            nxt = next((_stem(w) for w in tokens[i + 1:i + 4] if not w.isdigit() and w not in _STOP), "")
            out.add((t, nxt))
    return out


def _number_supported(claim: tuple[str, str], ev_tokens: list[str], window: int = 8) -> bool:
    """The same number with the same unit NEAR it (either side, so a table header 'Target (business days)' above
    '| 10 |' counts) -- not any occurrence of the number anywhere in the chunks."""
    num, unit = claim
    for i, t in enumerate(ev_tokens):
        if t == num:
            if not unit or unit in {_stem(w) for w in ev_tokens[max(0, i - window): i + window + 1]}:
                return True
    return False


def key_terms(sentence: str) -> set[str]:
    text = _CLAUSE_START.sub(lambda m: m.group(0).lower(), _clean(sentence))
    words = re.findall(r"[A-Za-z][A-Za-z\-]+", text)
    keys = {w.lower() for w in words[1:] if w[0].isupper() and len(w) > 2 and w.lower() not in _STOP}
    keys |= {w.lower() for w in words if w.lower() in _FREQ}
    return {_stem(k) for k in keys}


def sentence_support(sentence: str, chunks: list[str]) -> dict:
    ev_tokens = _tokens("\n".join(chunks))
    ev_words = {_stem(t) for t in ev_tokens if not t.isdigit()}
    missing_claims = sorted(c for c in _number_claims(_tokens(sentence)) if not _number_supported(c, ev_tokens))
    missing_keys = sorted(key_terms(sentence) - ev_words)
    s_words = content_words(sentence)
    recall = len(s_words & ev_words) / len(s_words) if s_words else 1.0
    return {"numbers_supported": not missing_claims, "key_terms_supported": not missing_keys,
            "content_recall": round(recall, 3), "missing_number_claims": missing_claims,
            "missing_key_terms": missing_keys, "missing_words": sorted(s_words - ev_words)}


def answer_supported(answer: str, chunks: list[str], min_recall: float) -> tuple[bool, list[dict]]:
    sents = split_sentences(answer)
    detail = [dict(sentence=s, **sentence_support(s, chunks)) for s in sents]
    ok = bool(sents) and all(d["numbers_supported"] and d["key_terms_supported"] and d["content_recall"] >= min_recall
                             for d in detail)
    return ok, detail

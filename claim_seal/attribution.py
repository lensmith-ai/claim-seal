"""Attribution overclaim: does the WRITING say its source states something the source only implies?

A HEURISTIC. Pure code, no model call, English only. It reads the attribution verb the writing
itself uses about its source in a checked sentence ("the report says ...", "the council argued
...") and compares it with the judge's relation for that sentence:

  strong attribution + a step-beyond relation  -> "overclaim" ("possible overclaim: ...")
        the writing says the source states it, but the source only implies (suggests, ...) it
  strong attribution + contradicts             -> "misattribution" ("possible misattribution: ...")
        the writing says the source states it, but the source contradicts it
  hedged attribution + says                    -> "understated" (informational, not a problem)
        the writing hedges ("argues", "claims") where the source states it plainly
  anything else, no attribution verb, or an undetermined relation -> no flag

Which verb: word-boundary, case-insensitive matches against the two lists below (present and
past tense). If the sentence names the cited source (its label, title, ref or id), the first
listed verb AFTER that name counts; otherwise the first listed verb in the sentence.

A listed word is SKIPPED when:
  - it follows a determiner ("the reports", "its notes"): a noun;
  - it is the first word of the sentence ("Reports show ..."): a plural noun as subject;
  - it is part of a known compound ("United States", "Member states", "TV shows") or of a
    hyphenated word ("Lost-and-found");
  - it sits inside the cited source's name ("Council Notes");
  - it sits inside quotation marks (quoted speech, not the writing's own attribution);
  - one of not / never / no / didn't / hasn't / doesn't / don't / isn't / wasn't / won't is
    among the three words before it ("never said", "has not confirmed").
A sentence ending in "?" is not checked at all.

Known misfires that remain (documented, not fixed): a verb about someone other than the source
("the mayor says ..."), including one placed after the source's name ("Harbour Press, which the
mayor says is biased, noted ..."); a plural noun after an adjective or another noun ("Recent
reports show", "Police notes"); the passive "It was found that"; negation further away ("Nothing
in Harbour Press states ..."); hedges not on the list ("estimates", "warns", "insists").

TREAT A FLAG AS A PROMPT TO REREAD THE SENTENCE, NOT AS A FINDING.

This check never changes a verdict or what a page prints, and it is NOT part of the seal: its
results live outside the claim-evidence rows.
"""
from __future__ import annotations

import re

from .relations import THROUGH_LINE, read_relation

STRONG = ("says", "said", "states", "stated", "reports", "reported", "describes", "described",
          "writes", "wrote", "notes", "noted", "announces", "announced", "confirms", "confirmed",
          "shows", "showed", "found", "finds")
HEDGED = ("argues", "argued", "claims", "claimed", "suggests", "suggested", "believes", "believed",
          "predicts", "predicted", "proposes", "proposed", "expects", "expected", "thinks",
          "thought", "speculates", "speculated")
_STRENGTH = {**{w: "strong" for w in STRONG}, **{w: "hedged" for w in HEDGED}}
_VERB = re.compile(r"\b(" + "|".join(sorted(_STRENGTH, key=len, reverse=True)) + r")\b", re.I)
_DETERMINERS = frozenset("the a an this that these those its their his her our my your some many "
                         "several any each every no".split())
_PREV_WORD = re.compile(r"([A-Za-z]+)\W*$")
_NEGATIONS = frozenset("not never no didn't hasn't doesn't don't isn't wasn't won't".split())
_COMPOUNDS = {("united", "states"), ("member", "states"), ("tv", "shows")}
_QUOTED = re.compile(r'"[^"]*"|\u201c[^\u201d]*\u201d')
_WORDS = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")
REREAD_NOTE = "treat a flag as a prompt to reread the sentence, not as a finding"


def _sentence_at(text: str, offset: int) -> tuple[int, int]:
    """(start, end) of the sentence in `text` containing `offset` (split on . ! ? + space)."""
    start = 0
    for m in re.finditer(r"[.!?]+(?:\s+|$)", text):
        if m.end() <= offset:
            start = m.end()
        elif m.start() >= offset:
            return start, m.end()
    return start, len(text)


def _name_spans(lowered: str, source_names) -> list[tuple[int, int]]:
    spans = []
    for name in source_names:
        name = str(name or "").strip().lower()
        if len(name) < 3:
            continue
        at = lowered.find(name)
        while at >= 0:
            spans.append((at, at + len(name)))
            at = lowered.find(name, at + 1)
    return spans


def find_attribution(sentence: str, source_names=()) -> tuple[str | None, str | None]:
    """(verb as written, "strong" | "hedged") or (None, None) when the sentence has none."""
    if not isinstance(sentence, str) or not sentence.strip():
        return None, None
    text = sentence.replace("\u2019", "'")
    if text.rstrip().endswith("?"):
        return None, None  # a question attributes nothing
    lowered = text.lower()
    names = _name_spans(lowered, source_names)
    quoted = [m.span() for m in _QUOTED.finditer(text)]
    first_word = _WORDS.search(text)
    candidates = []
    for m in _VERB.finditer(text):
        s, e = m.span()
        before_words = [w.lower() for w in _WORDS.findall(text[:s])]
        if before_words and before_words[-1] in _DETERMINERS:
            continue  # "the reports", "its notes": a noun, not an attribution
        if first_word is not None and first_word.start() == s:
            continue  # "Reports show ...": a sentence-initial plural noun
        if before_words and (before_words[-1], m.group(1).lower()) in _COMPOUNDS:
            continue  # "United States", "Member states", "TV shows"
        if (s > 0 and text[s - 1] == "-") or (e < len(text) and text[e] == "-"):
            continue  # "Lost-and-found": part of a hyphenated word
        if any(a <= s and e <= b for a, b in names):
            continue  # inside the cited source's own name ("Council Notes")
        if any(a < s and e <= b for a, b in quoted):
            continue  # quoted speech, not the writing's attribution
        if _NEGATIONS & set(before_words[-3:]):
            continue  # "never said", "has not confirmed"
        candidates.append(m)
    if not candidates:
        return None, None
    after = None
    for a, b in names:
        after = b if after is None else min(after, b)
    chosen = next((m for m in candidates if after is not None and m.start() >= after), candidates[0])
    verb = chosen.group(1)
    return verb, _STRENGTH[verb.lower()]


def judge_attribution(verb: str | None, strength: str | None, relation: str | None) -> dict:
    """The per-sentence result: {attribution_verb, attribution_strength, relation, flag, message}."""
    out = {"attribution_verb": verb, "attribution_strength": strength, "relation": relation,
           "flag": None, "message": None}
    if not verb or not relation:
        return out
    line = THROUGH_LINE.get(relation)
    if strength == "strong" and line == "step-beyond":
        out.update(flag="overclaim",
                   message=f'possible overclaim: the writing says the source states it ("{verb}"), '
                           f"but the source only {relation} it, as the judge read it; {REREAD_NOTE}")
    elif strength == "strong" and line == "contradicts":
        out.update(flag="misattribution",
                   message=f'possible misattribution: the writing says the source states it ("{verb}"), '
                           f"but the source contradicts it, as the judge read it; {REREAD_NOTE}")
    elif strength == "hedged" and line == "states":
        out.update(flag="understated",
                   message=f'understated (informational, not a problem): the writing hedges ("{verb}"), '
                           "but the source states it, as the judge read it")
    return out


def _names(citation: dict, labels: dict | None) -> list:
    sid = citation.get("source")
    return [n for n in ((labels or {}).get(sid), citation.get("matched_title"), citation.get("ref"),
                        sid) if n]


def check_span(item: dict, citation: dict, labels: dict | None = None) -> dict | None:
    """The attribution result for one citation's span, or None when it has no span."""
    span = citation.get("span") if isinstance(citation, dict) else None
    if not isinstance(span, dict):
        return None
    field = span.get("field")
    if field == "insight":
        text = item.get("insight") or item.get("text") or item.get("headline") or ""
    elif field == "why_it_matters":
        text = item.get("why_it_matters") or ""
    elif isinstance(field, str) and field.startswith("detail["):
        details = item.get("detail") or []
        k = int(field[7:-1])
        text = details[k] if 0 <= k < len(details) else ""
    else:
        return None
    if not isinstance(text, str):
        return None  # a hand-built brief with a non-string line: nothing to read, never a crash
    start = span.get("start") if isinstance(span.get("start"), int) else 0
    s, e = _sentence_at(text, start)
    sentence = text[s:e].strip()
    verb, strength = find_attribution(sentence, _names(citation, labels))
    rel = span.get("sentence_relation")
    relation = read_relation(rel)[0] if rel else None
    result = judge_attribution(verb, strength, relation)
    result.update({"field": field, "source": citation.get("source"), "sentence": sentence})
    return result


def attribution_checks(brief: dict, labels: dict | None = None) -> list[dict]:
    """One result per spanned citation of every item in an annotated brief. Not sealed."""
    out = []
    for index, item in enumerate(brief.get("items") or []):
        for c_index, citation in enumerate(((item.get("verification") or {}).get("citations")) or []):
            result = check_span(item, citation, labels)
            if result is not None:
                out.append({"item": index + 1, "citation_index": c_index, **result})
    return out

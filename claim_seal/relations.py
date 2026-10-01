"""The relation vocabulary, and how each word draws (or refuses) a "through line".

A relation names how one sentence, or one claim, stands to one source:

  says          the source itself states it
  implies, suggests, connotes, follows, explains, posits, illustrates
                the writing takes a step beyond the source; the verb names the step
  contradicts   the source disagrees
  undetermined  the judge did not (or could not) rate it

The judge's own definition, given to it in the prompt: "'says' means this source itself states
that sentence's content; any other verb names the step that sentence takes beyond this source."

Three views of the same closed vocabulary live here, so they cannot drift apart:

  derive_coarse_relation(verb)  the coarse axis carried in the sealed record
                                (stated / inferred / contradicted / undetermined)
  through_line(verb)            states / step-beyond / contradicts / unrated
  line_rule(verb)               what a reader-facing page may print for a line holding that verb:
                                None (no line-level state of its own), "reading" (printed as
                                "our reading"), or "left-out" (never printed)

An unknown word is REFUSED (UnknownRelation), never mapped to a default: a page that guessed
would print a state nobody chose.
"""
from __future__ import annotations

SPAN_RELATIONS = frozenset({"says", "implies", "suggests", "connotes", "follows", "explains",
                            "posits", "illustrates", "contradicts", "undetermined"})
STEP_BEYOND = frozenset({"implies", "suggests", "connotes", "follows", "explains", "posits",
                         "illustrates"})

_COARSE_RELATION_BY_VERB = {
    "says": "stated", "contradicts": "contradicted", "undetermined": "undetermined",
    **{verb: "inferred" for verb in STEP_BEYOND},
}

THROUGH_LINE = {
    "says": "states",
    "undetermined": "unrated",
    **{verb: "step-beyond" for verb in STEP_BEYOND},
    "contradicts": "contradicts",
}

# The publishing rule a reader-facing page applies per printed line (mirrors the finished-page
# packet builder the composer ships with):
#   says          no line state of its own: the line carries its claim's verdict, and is printed
#                 "supported" at line level only when a second, sentence-level check said so
#   undetermined  no line state of its own: the judge did not rate the sentence
#   step-beyond   "reading": printed as "our reading", a step the writing takes beyond the source
#   contradicts   "left-out": the source disagrees, so the line is never printed
LINE_RULE = {
    "says": None,
    "undetermined": None,
    **{verb: "reading" for verb in STEP_BEYOND},
    "contradicts": "left-out",
}

PRINTED_VERDICTS = {"supported": "supported", "partial": "partly_supported"}
LABEL = {"supported": "supported", "partly_supported": "partly supported",
         "reading": "our reading"}


class UnknownRelation(ValueError):
    """A relation word outside the closed vocabulary."""


def _closed(verb: object) -> str:
    word = str(verb if verb is not None else "").strip().lower()
    if word not in SPAN_RELATIONS:
        raise UnknownRelation(f"relation word {verb!r} is not in the vocabulary "
                              f"{sorted(SPAN_RELATIONS)}")
    return word


def derive_coarse_relation(relation: object) -> str:
    """The coarse axis of the sealed record. Anything outside the vocabulary is 'undetermined'
    here (the record must still be written); the publishing views below refuse it instead."""
    return _COARSE_RELATION_BY_VERB.get(str(relation or "").lower(), "undetermined")


def through_line(verb: object) -> str:
    """states / step-beyond / contradicts / unrated. Unknown words raise UnknownRelation."""
    return THROUGH_LINE[_closed(verb)]


def line_rule(verb: object) -> str | None:
    """None / 'reading' / 'left-out'. Unknown words raise UnknownRelation."""
    return LINE_RULE[_closed(verb)]


def _relation(citation: dict) -> str | None:
    span = citation.get("span") or {}
    rel = span.get("sentence_relation") or citation.get("relation")
    if rel is None:
        return None
    return _closed(rel)


def item_lines(item: dict) -> list:
    """(field, span field name, text) in reading order: the claim, why it matters, details."""
    lines = [("insight", "insight", item.get("insight") or item.get("text") or item.get("headline"))]
    if item.get("why_it_matters"):
        lines.append(("why_it_matters", "why_it_matters", item["why_it_matters"]))
    for k, detail in enumerate(item.get("detail") or []):
        if isinstance(detail, str) and detail.strip():
            lines.append(("detail", "detail[{}]".format(k), detail))
    return lines


def line_states(brief: dict, second_checks: dict | None = None) -> dict:
    """Decide what a reader-facing page may print for every line of a judged brief.

    `brief` is the annotated brief judge() returns. `second_checks` optionally maps
    item index -> {(span field, text): {...}} for sentences a separate sentence-level check
    found supported. Returns {"lines": [...], "omitted": [...], "left_out": [...]}:

    - an item is printed only when its claim verdict is supported or partial;
    - a line whose span relation is contradicts is left out; on the claim line itself the whole
      item is omitted;
    - a line whose span relation is a step beyond the source is "our reading", but only when that
      citation resolved; otherwise the line is left out;
    - otherwise the line carries its claim's verdict ("card" scope), or "supported" at line scope
      when a second check said so.
    Unknown relation words raise UnknownRelation.
    """
    second_checks = second_checks or {}
    lines_out, omitted, left_out = [], [], []
    for index, item in enumerate(brief.get("items") or []):
        verification = item.get("verification") or {}
        verdict = PRINTED_VERDICTS.get(verification.get("verdict"))
        if not verdict:
            omitted.append({"item": index + 1, "rule": "verdict-not-printable",
                            "verdict": verification.get("verdict")})
            continue
        citations = [c for c in (verification.get("citations") or []) if isinstance(c, dict)]
        planned = []
        heading_contradicted = False
        for field, span_field, text in item_lines(item):
            spanned = [c for c in citations if (c.get("span") or {}).get("field") == span_field]
            relations = [_relation(c) for c in spanned]
            if "contradicts" in relations:
                if field == "insight":
                    heading_contradicted = True
                else:
                    left_out.append({"item": index + 1, "field": span_field,
                                     "rule": "span-contradicts"})
                continue
            resolved = [c for c in spanned if c.get("resolution") != "unresolved"]
            readings = [c for c in resolved if LINE_RULE.get(_relation(c)) == "reading"]
            if not readings and any(LINE_RULE.get(r) == "reading" for r in relations):
                left_out.append({"item": index + 1, "field": span_field,
                                 "rule": "reading-without-resolved-citation"})
                continue
            planned.append((field, span_field, text, resolved, readings))
        if heading_contradicted:
            omitted.append({"item": index + 1, "rule": "heading-span-contradicts"})
            continue
        for field, span_field, text, resolved, readings in planned:
            second_check = second_checks.get(index, {}).get((span_field, text))
            if readings:
                anchor = readings[0]
                state, scope, relation = "reading", "line", _relation(anchor)
                source = anchor.get("source")
            elif second_check:
                state, scope, relation = "supported", "line", None
                source = second_check.get("source")
            else:
                state, scope = verdict, "card"
                relation = _relation(resolved[0]) if resolved else None
                source = resolved[0].get("source") if resolved else None
            lines_out.append({"item": index + 1, "field": span_field, "text": text,
                              "state": state, "label": LABEL[state], "state_scope": scope,
                              "relation": relation,
                              "through_line": THROUGH_LINE.get(relation) if relation else None,
                              "source": source})
    return {"lines": lines_out, "omitted": omitted, "left_out": left_out}

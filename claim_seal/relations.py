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

Reading a judge's word: read_relation() trims and lower-cases it, reads the synonyms in
RELATION_SYNONYMS as the verb they name, and reads ANY OTHER word as "undetermined", marked
unknown ("can't tell": never read as support). The three table views above are stricter: handed
a word that is neither a vocabulary verb nor a synonym directly, they REFUSE it (UnknownRelation)
instead of guessing, because nothing says where it came from.
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
         "reading": "our reading", "cant_tell": "can't tell"}


# Synonyms the reader accepts and the verb each is read as. THE ONE LIST: the claim-level guard,
# the evidence pass, the per-sentence intake and the page views all read through read_relation.
# The judge's prompt still offers only the ten vocabulary words; this only forgives a judge that
# answers with a near word. Documented in the README ("Relation words").
RELATION_SYNONYMS = {
    # states it -> says
    **{w: "says" for w in ("states", "stated", "reports", "reported", "notes", "noted",
                           "describes", "described", "writes", "wrote", "announces", "announced",
                           "confirms", "confirmed", "said")},
    # a step beyond -> a canonical step-beyond verb
    **{w: "implies" for w in ("implied", "infers", "inferred", "argues", "argued")},
    "suggested": "suggests", "connoted": "connotes", "followed": "follows",
    "explained": "explains", "posited": "posits", "illustrated": "illustrates",
    # disagrees -> contradicts
    **{w: "contradicts" for w in ("contradicted", "refutes", "refuted", "denies", "denied",
                                  "disputes", "disputed", "disagrees", "disagreed", "conflicts",
                                  "rejects", "rejected")},
}
# Kept for callers of the earlier name: the three composer aliases are synonyms now.
RELATION_ALIASES = {w: RELATION_SYNONYMS[w] for w in ("stated", "inferred", "contradicted")}


def read_relation(raw: object) -> tuple[str | None, str]:
    """The ONE reader of a judge's relation word, used by every guard so they cannot disagree.

    Trims and lower-cases, then returns (verb, how):
      (verb, "verb")             a word in the closed vocabulary
      (verb, "synonym")          a word in RELATION_SYNONYMS, read as the verb it names
      ("undetermined", "unknown")  any other word: CAN'T TELL. Never read as support.
      (None, "empty")            nothing given
    """
    word = str(raw if raw is not None else "").strip().lower()
    if word in SPAN_RELATIONS:
        return word, "verb"
    if word in RELATION_SYNONYMS:
        return RELATION_SYNONYMS[word], "synonym"
    return (None, "empty") if not word else ("undetermined", "unknown")


class UnknownRelation(ValueError):
    """A relation word outside the closed vocabulary."""


def _closed(verb: object) -> str:
    """A vocabulary verb or synonym, read; anything else is refused (UnknownRelation).

    The table views refuse an unknown word because nothing says where it came from. A judge's
    unknown word never reaches them raw: intake has already read it as "undetermined" and marked
    the span (sentence_relation_read: "unknown"), and line_states shows that sentence as can't tell.
    """
    word, how = read_relation(verb)
    if how not in ("verb", "synonym"):
        raise UnknownRelation(f"relation word {verb!r} is not in the vocabulary "
                              f"{sorted(SPAN_RELATIONS)} or its synonyms")
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


def _safe_check(check_span, item, citation, labels):
    """The attribution heuristic is informational: it must never stop a page from rendering."""
    try:
        return check_span(item, citation, labels)
    except Exception:  # noqa: BLE001 - a heuristic failure is no result, never a crash
        return None


def line_states(brief: dict, second_checks: dict | None = None, labels: dict | None = None) -> dict:
    """Decide what a reader-facing page may print for every line of a judged brief.

    `brief` is the annotated brief judge() returns. `second_checks` optionally maps
    item index -> {(span field, text): {...}} for sentences a separate sentence-level check
    found supported. Returns {"lines": [...], "omitted": [...], "left_out": [...]}:

    - an item is printed only when its claim verdict is supported or partial;
    - a line whose span relation is contradicts is left out; on the claim line itself the whole
      item is omitted;
    - a line whose span relation is a step beyond the source is "our reading", but only when that
      citation resolved; otherwise the line is left out;
    - a line whose span carries a word the judge used that the reader did not know
      (sentence_relation_read: "unknown") is "can't tell" at line scope: it never takes the
      claim's verdict and never prints as supported;
    - otherwise the line carries its claim's verdict ("card" scope), or "supported" at line scope
      when a second check said so.
    Every printed and left-out line also carries "attribution": the attribution-overclaim results
    (claim_seal.attribution) for the spans on it. They are informational: they change no state.
    `labels` ({source id: name}) helps find the source's name in a sentence.
    A relation word outside the vocabulary and its synonyms that did NOT come through intake
    raises UnknownRelation.
    """
    from .attribution import check_span  # local: attribution imports this module
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
            attribution = [r for r in (_safe_check(check_span, item, c, labels) for c in spanned) if r]
            if "contradicts" in relations:
                if field == "insight":
                    heading_contradicted = True
                else:
                    left_out.append({"item": index + 1, "field": span_field,
                                     "rule": "span-contradicts", "attribution": attribution})
                continue
            resolved = [c for c in spanned if c.get("resolution") != "unresolved"]
            readings = [c for c in resolved if LINE_RULE.get(_relation(c)) == "reading"]
            if not readings and any(LINE_RULE.get(r) == "reading" for r in relations):
                left_out.append({"item": index + 1, "field": span_field,
                                 "rule": "reading-without-resolved-citation",
                                 "attribution": attribution})
                continue
            cant_tell = [c for c in spanned
                         if (c.get("span") or {}).get("sentence_relation_read") == "unknown"]
            planned.append((field, span_field, text, resolved, readings, cant_tell, attribution))
        if heading_contradicted:
            omitted.append({"item": index + 1, "rule": "heading-span-contradicts"})
            continue
        for field, span_field, text, resolved, readings, cant_tell, attribution in planned:
            try:
                second_check = second_checks.get(index, {}).get((span_field, text))
            except TypeError:  # a hand-built line that is not text (a dict, a list) has no second check
                second_check = None
            if readings:
                anchor = readings[0]
                state, scope, relation = "reading", "line", _relation(anchor)
                source = anchor.get("source")
            elif cant_tell:
                state, scope, relation = "cant_tell", "line", "undetermined"
                source = cant_tell[0].get("source")
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
                              "source": source, "attribution": attribution})
    return {"lines": lines_out, "omitted": omitted, "left_out": left_out}

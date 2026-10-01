"""Per-sentence citation spans: WHERE in the writing each source's support lands.

The judge proposes a quote from the writing (span_quote) for each citation, plus the relation of
THAT sentence to THAT source (span_relation). Everything here is fail-closed: a quote is kept
only when it occurs exactly once, in exactly one field; otherwise it is refused with a named
reason and counted. A span records the producer that placed it, so a reader can always tell a
model quotation from a computed placement.

Offsets are Unicode code points into the brief's own text ("unicode-code-point-v1"), and the
stored quote is always sliced from OUR prose, never copied from what the model typed.
"""
from __future__ import annotations

import re
import unicodedata

from .relations import SPAN_RELATIONS
from .resolve import target_for_claim

SPAN_CONTRACT = "citation-span/v1"

_SPAN_PUNCT = {"’": "'", "‘": "'", "“": '"', "”": '"',
               "—": "-", "–": "-", "−": "-", " ": " "}


def _norm_with_map(text):
    """Normalised text plus, for each normalised char, its index in the ORIGINAL string.

    The returned offsets are the provenance record, so this mapping is the whole risk in
    normalised matching (tests/test_span_normalization.py asserts on slices, not numbers).
    """
    out, idx, prev_space = [], [], False
    for i, ch in enumerate(text):
        c = _SPAN_PUNCT.get(ch, ch)
        c = unicodedata.normalize("NFKC", c)
        if c.isspace():
            if prev_space:
                continue                      # collapse the run; its FIRST index is already recorded
            c, prev_space = " ", True
        else:
            prev_space = False
            c = c.lower()
        out.append(c)
        idx.append(i)
    return "".join(out), idx


def find_forgiving(text, quote):
    """Locate quote in text ignoring case, whitespace runs and smart punctuation.

    Fallback ONLY: callers try the exact match first, so this can add recoveries but never remove
    one the exact path makes. Returns (start, end) in ORIGINAL offsets, or None. It must NOT
    rescue paraphrase, which stays unresolved by design.
    """
    ntext, idx = _norm_with_map(text)
    nquote, _ = _norm_with_map(quote)
    nquote = nquote.strip()
    if not nquote:
        return None
    hit = ntext.find(nquote)
    if hit < 0 or ntext.find(nquote, hit + 1) >= 0:
        return None                            # absent, or ambiguous under normalisation
    return idx[hit], idx[hit + len(nquote) - 1] + 1


def text_slice(text, start, end):
    """The stored quote must be what OUR prose says, not what the model typed."""
    return text[start:end]


def validate_span(span_fields, citation: dict, proposal: object) -> tuple[dict | None, str | None]:
    """Fail closed: keep one model quote only when it occurs exactly once, in exactly one field.

    Accepts the claim text alone (str), or a mapping of field name -> text. A quote that appears
    in two different fields is AMBIGUOUS and refused: picking one would attach a source to a
    sentence nobody chose.
    """
    if isinstance(span_fields, str):
        span_fields = {"insight": span_fields}
    if not isinstance(citation, dict) or citation.get("resolution") == "unresolved":
        return None, "unresolved_citation"
    if citation.get("completion") or citation.get("evidence_completion"):
        return None, "evidence_completion_citation"
    if not citation.get("text"):
        return None, "unresolved_citation"
    idx = proposal.get("citation_index") if isinstance(proposal, dict) else None
    if not isinstance(idx, int) or isinstance(idx, bool):
        return None, "invalid_citation_index"
    quote = proposal.get("span_quote")
    if quote is None:
        return None, "not_proposed"
    if not isinstance(quote, str):
        return None, "non_string_quote"
    if not quote:
        return None, "empty_quote"
    hits = []
    for field, text in (span_fields or {}).items():
        if not isinstance(text, str):
            continue
        first = text.find(quote)
        if first < 0:
            continue
        if text.find(quote, first + 1) >= 0:
            return None, "quote_ambiguous"
        hits.append((field, first))
    if not hits:
        # Exact matching failed. A quote that differs from our prose only by case, a collapsed
        # space or a smart apostrophe is the same quote. Paraphrase still fails here.
        soft = []
        for field, text in (span_fields or {}).items():
            if not isinstance(text, str):
                continue
            got = find_forgiving(text, quote)
            if got:
                soft.append((field, got))
        if len(soft) == 1:
            field, (start, end) = soft[0]
            return {"field": field, "start": start, "end": end,
                    "quote": text_slice(span_fields[field], start, end),
                    "coordinate_system": "unicode-code-point-v1",
                    "producer": "model-quote-normalized-v1"}, None
        return None, "quote_not_found"
    if len(hits) > 1:
        return None, "quote_ambiguous_across_fields"
    field, first = hits[0]
    return {"field": field, "start": first, "end": first + len(quote), "quote": quote,
            "coordinate_system": "unicode-code-point-v1", "producer": "model-exact-quote-v1"}, None


def span_fields(brief: dict, claim: dict) -> dict:
    """The texts a span may anchor into: the claim, and the sentences that support it.

    Marks belong in the supporting paragraph rather than on the thesis, which is the sentence
    every other sentence exists to support and so the one that needs marking least. Every span
    records its field: a mark in the paragraph says WHERE a source's support lands, not that the
    sentence was separately verified.
    """
    fields = {"insight": claim.get("claim") or ""}
    items = brief.get("items") or []
    obj = None
    if claim.get("kind") == "item" and isinstance(claim.get("index"), int):
        obj = items[claim["index"]] if claim["index"] < len(items) else None
    elif claim.get("kind") == "connection":
        obj = brief.get("connection")
    if not isinstance(obj, dict):
        return fields
    why = obj.get("why_it_matters")
    if isinstance(why, str) and why.strip():
        fields["why_it_matters"] = why
    for d_i, d in enumerate(obj.get("detail") or []):
        if isinstance(d, str) and d.strip():
            fields["detail[%d]" % d_i] = d
    return fields


_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have if in into is it its of on or so than that the "
    "their there these this to was were what when which while who will with would".split())


def _content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9'-]+", (text or "").lower())
            if len(w) > 2 and w not in _STOPWORDS}


def _sentences_with_offsets(text: str):
    """(start, end) code-point offsets of each sentence, split on . ! ? followed by space/end."""
    for m in re.finditer(r"[^.!?]+(?:[.!?]+|$)", text or ""):
        s, e = m.start(), m.end()
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e - s >= 12:
            yield s, e


def relocate_to_support(fields: dict, source_text: str) -> dict | None:
    """Pick the supporting sentence the cited source most plausibly bears on. Local, no model.

    Asking the JUDGE to place marks in the supporting paragraph raised placement from 22% to 96%
    but made one whole-brief call outgrow its time cap (3 of 7 runs lost everything). Placement
    is mostly a text-overlap question and every input is in hand, so this scores each non-thesis
    sentence by content-word overlap with the source excerpt and picks one only when it clearly
    wins. Fail-closed: no winner, a tie, or a thin overlap return None.

    The floor (>=3 shared content words AND >=0.25 of the sentence's words) is deliberate: below
    it, an overlap says "shares vocabulary", not "bears on". The result records its score and the
    producer local-overlap-relocation-v1.
    """
    src = _content_words(source_text)
    if not src:
        return None
    best = None          # (score, field, start, end, overlap_n)
    tied = False
    for field in [k for k in ("why_it_matters",) if k in (fields or {})] + \
                 sorted(k for k in (fields or {}) if k.startswith("detail[")):
        text = fields.get(field)
        if not isinstance(text, str):
            continue
        for s, e in _sentences_with_offsets(text):
            words = _content_words(text[s:e])
            if not words:
                continue
            overlap = len(words & src)
            score = overlap / max(1, len(words))
            if overlap >= 3 and score >= 0.25:
                if best is None or score > best[0] + 1e-9:
                    best, tied = (score, field, s, e, overlap), False
                elif abs(score - best[0]) <= 1e-9:
                    tied = True
    if best is None:
        return None
    if tied:             # two sentences scored identically: choosing one would be a guess
        return None
    score, field, s, e, overlap = best
    return {"field": field, "start": s, "end": e, "quote": text_slice(fields[field], s, e),
            "coordinate_system": "unicode-code-point-v1",
            "producer": "local-overlap-relocation-v1",
            "overlap": {"shared_words": overlap, "sentence_score": round(score, 3)}}


def span_receipt(report: dict, verdicts: dict, brief: dict, *, local_placement: bool = True) -> dict:
    """Validate every proposed item-citation span, attach the survivors, and account for the rest.

    Invariant (asserted): eligible == produced + dropped + not_proposed.
    """
    eligible_rows = []
    for claim_n, claim in enumerate(report["claims"]):
        if claim["kind"] != "item":
            continue
        for citation_index, citation in enumerate(claim["citations"]):
            if citation["resolution"] != "unresolved" and citation.get("text"):
                eligible_rows.append((claim_n, claim, citation_index, citation))
    result = {"status": "ok", "eligible": len(eligible_rows), "produced": 0, "dropped": 0,
              "not_proposed": 0, "dropped_by_reason": {}, "contract": SPAN_CONTRACT}
    spans: dict[tuple[int, int], dict] = {}
    for claim_n, claim, citation_index, citation in eligible_rows:
        raw = verdicts.get(claim_n, {}).get("citation_spans")
        proposals = raw if isinstance(raw, list) else []
        matching = [p for p in proposals if isinstance(p, dict)
                    and isinstance(p.get("citation_index"), int)
                    and not isinstance(p.get("citation_index"), bool)
                    and p.get("citation_index") == citation_index]
        if not matching:
            result["not_proposed"] += 1
            continue
        if len(matching) != 1:
            result["dropped"] += 1
            result["dropped_by_reason"]["duplicate_citation_index"] = (
                result["dropped_by_reason"].get("duplicate_citation_index", 0) + 1)
            continue
        fields = span_fields(brief, claim)
        span, reason = validate_span(fields, citation, matching[0])
        if span is not None:
            # The judge's verb for the QUOTED sentence vs this source. Whitelisted, else dropped.
            # Attached before relocation on purpose: a RELOCATED mark sits on a sentence the judge
            # never rated, so relocation loses the field rather than carrying a wrong rating.
            rel = matching[0].get("span_relation") if isinstance(matching[0], dict) else None
            if isinstance(rel, str) and rel.strip().lower() in SPAN_RELATIONS:
                span["sentence_relation"] = rel.strip().lower()
            # Local placement: a validated span that landed on the THESIS may move to the
            # supporting sentence the source text most overlaps. Only a span that EXISTS is moved;
            # the support verdict stays the judge's.
            if span.get("field") == "insight" and local_placement:
                alt = relocate_to_support(fields, citation.get("text") or "")
                if alt is not None:
                    alt["relocated_from"] = {k: span.get(k) for k in
                                             ("field", "start", "end", "quote", "producer")}
                    span = alt
                    result["relocated"] = result.get("relocated", 0) + 1
            spans[(claim_n, citation_index)] = span
            result["produced"] += 1
        elif reason == "not_proposed":
            result["not_proposed"] += 1
        else:
            # When the judge proposed a quote that cannot be located in our prose, the local
            # scorer may place the mark on the sentence the SOURCE most overlaps, under the same
            # floors, with an honest producer: local-overlap-scorer-v1 means "computed placement,
            # the checker could not quote it". Every other drop reason still refuses.
            if reason == "quote_not_found" and local_placement:
                alt = relocate_to_support(fields, citation.get("text") or "")
                if alt is not None:
                    alt["producer"] = "local-overlap-scorer-v1"
                    alt["unlocatable_quote"] = str(matching[0].get("span_quote") or "")[:160]
                    spans[(claim_n, citation_index)] = alt
                    result["produced"] += 1
                    result["scorer_placed"] = result.get("scorer_placed", 0) + 1
                    continue
            result["dropped"] += 1
            result["dropped_by_reason"][reason or "invalid"] = result["dropped_by_reason"].get(reason or "invalid", 0) + 1
            # Diagnostic: keep the PROPOSED quote (claimed to be a substring of OUR prose, never
            # source text), capped, so a near-miss can be told from invention.
            if len(result.setdefault("dropped_samples", [])) < 12:
                proposed = matching[0].get("span_quote") if isinstance(matching[0], dict) else None
                if isinstance(proposed, str) and proposed:
                    result["dropped_samples"].append(
                        {"reason": reason or "invalid", "proposed_quote": proposed[:160]})
    # Invalid indexes never name an eligible citation. They are recorded without inflating the
    # per-eligible denominator; a valid citation can never inherit their span.
    for claim_n, verdict in verdicts.items():
        claim = report["claims"][claim_n]
        if claim["kind"] != "item" or not isinstance(verdict.get("citation_spans"), list):
            continue
        eligible_indexes = {i for i, cite in enumerate(claim["citations"])
                            if cite["resolution"] != "unresolved" and cite.get("text")}
        for proposal in verdict["citation_spans"]:
            pi = proposal.get("citation_index") if isinstance(proposal, dict) else None
            if not isinstance(pi, int) or isinstance(pi, bool) or pi not in eligible_indexes:
                result["invalid_index_proposals"] = result.get("invalid_index_proposals", 0) + 1
    for claim_n, claim, citation_index, _ in eligible_rows:
        target = target_for_claim(brief, claim)
        citations = ((target or {}).get("verification") or {}).get("citations") or []
        if citation_index < len(citations) and (claim_n, citation_index) in spans:
            citations[citation_index]["span"] = spans[(claim_n, citation_index)]
    assert result["eligible"] == result["produced"] + result["dropped"] + result["not_proposed"]
    return result

"""The pass-2 judge prompts, word for word as the composer sends them.

The wording is part of the method: each clause below was added after a measured failure (noted
beside it), and changing it changes the verdicts. Options that switch wording on or off are
keyword arguments here; in the composer they were deployment flags.

  sharded         one claim per prompt. Adds the per-sentence relation ask (span_relation), the
                  per-citation "how it bears" sentence, and the two-texts warning.
  locating_quote  an alternative span instruction asking for a locating phrase. Default OFF: it
                  raised placement in the supporting paragraph from 22% to 96% in an A/B, but in
                  the single-call configuration 2 of 4 treatment runs lost verification entirely
                  (timeouts), and a total loss yields no spans at all.
"""
from __future__ import annotations

from .resolve import MAX_SRC_CHARS


def build_prompt(report: dict, *, citation_spans_enabled: bool = False,
                 claim_evidence_candidates: dict | None = None,
                 span_fields_by_claim: dict | None = None,
                 sharded: bool = False, locating_quote: bool = False) -> tuple[str, list]:
    """Pass 2 prompt. Only claims with at least one RESOLVED citation are included.

    Returns (prompt, checkable) where checkable[k] is the report index of prompt claim k + 1.
    """
    blocks, checkable = [], []
    for n, c in enumerate(report["claims"]):
        resolved = [x for x in c["citations"] if x["resolution"] != "unresolved" and x["text"]]
        if not resolved:
            continue
        checkable.append(n)
        annotate = (citation_spans_enabled and c.get("kind") == "item") or claim_evidence_candidates is not None
        # When the evidence pass is active the judge gets TWO views of the same source: the clean
        # text it has always judged against, and the captured document window a support_span must
        # be quoted from. Showing ONLY the raw window degraded the judge measurably: before the
        # change verdicts carried 1-3 supported and zero unsupported across five real briefs; the
        # first run on raw-only produced zero supported and two unsupported, because the judge was
        # reading markup and boilerplate instead of prose.
        srcs = "\n".join(
            (f"  SOURCE {j + 1} (citation_index {i}, {x['source']}): {x['text']}\n"
             f"    RESPONSE TEXT (quote support_span verbatim from HERE, not from the line "
             f"above): {(claim_evidence_candidates or {}).get((n, i), '')}")
            if annotate and (claim_evidence_candidates or {}).get((n, i))
            else f"  SOURCE {j + 1} (citation_index {i}, {x['source']}): {x['text']}"
            if annotate else f"  SOURCE {j + 1} ({x['source']}): {x['text']}"
            for j, (i, x) in enumerate((i, x) for i, x in enumerate(c["citations"])
                                       if x["resolution"] != "unresolved" and x["text"]))
        # A judge cannot quote a sentence it was never shown. The first attempt at anchoring into
        # supporting prose changed the validator and the instruction but not the INPUT: the prompt
        # carried only the claim, so a real brief produced 8 spans and every one landed on the
        # thesis.
        supporting = ""
        if citation_spans_enabled:
            fields = (span_fields_by_claim or {}).get(n) or {}
            lines = [f"    [{name}] {text}" for name, text in fields.items() if name != "insight"]
            if lines:
                supporting = "\n  SUPPORTING SENTENCES (a span_quote may come from one of these, "
                supporting += "and should where the source bears on it):\n" + "\n".join(lines)
        blocks.append(f"CLAIM {len(checkable)}:\n  {c['claim']}{supporting}\n{srcs}")
    # Per-SENTENCE relation: the judge already picks the sentence when it quotes span_quote; this
    # asks, in the same breath, how THAT sentence relates to THAT source (a sentence the source
    # states verbatim must not inherit a claim-level "goes further" label).
    span_relation_field = (
        ', "span_relation": "says"|"implies"|"suggests"|"connotes"|"follows"|"explains"|'
        '"posits"|"illustrates"|"contradicts"|"undetermined"'
        if sharded else "")
    span_instruction = (
        (', "citation_spans": [{"citation_index": <original citation-array index>, '
         '"span_quote": "<a phrase copied CHARACTER FOR CHARACTER from the CLAIM or from one of its '
         'SUPPORTING sentences — prefer a supporting sentence. It must be long enough to appear '
         'EXACTLY ONCE across those texts, normally six words or more. Do not shorten it to a single '
         'word, do not reword it, and do not summarise: if you cannot copy a phrase exactly, return '
         'null for this citation>"' + span_relation_field + '}|null]'
         if locating_quote else
         ', "citation_spans": [{"citation_index": <original citation-array index>, '
         '"span_quote": "<verbatim contiguous substring of the CLAIM or of one of its '
         'SUPPORTING sentences — prefer a supporting sentence, since the claim is what they '
         'exist to support>"' + span_relation_field + '}|null]')
        if citation_spans_enabled else "")
    evidence_instruction = (
        ', "evidence_spans": [{"citation_index": <original citation-array index>, '
        '"support_span": "<verbatim contiguous substring of that response-bound SOURCE>"}], '
        '"multi_span_required": <boolean>' if claim_evidence_candidates is not None else "")
    # Per-citation bearing sentence: a reader's popup should explain HOW each source bears on the
    # claim. OWN WORDS ONLY: the checker retains no source excerpts, so this must be a digest,
    # never a quotation.
    bears_instruction = (
        ', "citation_bears": [{"citation_index": <original citation-array index>, '
        '"how_it_bears": "<ONE sentence, at most 28 words, IN YOUR OWN WORDS and never a '
        'quotation: what this source says and how that supports, extends, or fails to support '
        'the claim>"}]'
        if sharded else "")
    judgement_instruction = (
        ', "relation": "says"|"implies"|"suggests"|"connotes"|"follows"|"explains"|"posits"|"illustrates"|"contradicts"|"undetermined", '
        '"strength": <integer 0-100>' if claim_evidence_candidates is not None else "")
    prompt = (
        "You are checking whether each CLAIM is supported by the SOURCE text quoted beneath "
        "it. Judge ONLY against the quoted text. Do not use outside knowledge. Do not be "
        "generous: a claim that merely sounds plausible, or that generalises well beyond what "
        "the source says, is 'partial' at best.\n\n"
        "For each claim return one object:\n"
        '  {"claim": <number>, "verdict": "supported"|"partial"|"unsupported", '
        '"why": "<one sentence, max 20 words, quoting or naming the deciding detail>"'
        + span_instruction + judgement_instruction + evidence_instruction + bears_instruction + "}\n\n"
        + ("Return one citation_bears row for EVERY SOURCE shown under the claim, whatever the "
           "verdict: say what that source states and how it supports, extends, or fails to "
           "support the claim. Own words only — never copy source text into how_it_bears.\n\n"
           if bears_instruction else "")
        + ("Return citation_spans ONLY for claims whose SOURCE lines show a citation_index "
           "(at most one row per such citation).\n"
           "span_quote: ask WHICH SENTENCE THIS SOURCE MOST DIRECTLY BEARS ON, then copy a "
           "contiguous phrase from that sentence exactly. Where a SUPPORTING SENTENCE is listed "
           "under the claim, that is almost always the answer: the claim is the conclusion those "
           "sentences argue for, so a source usually bears on the reasoning rather than on the "
           "conclusion itself. Quote the claim only when the source bears on the conclusion and "
           "no supporting sentence carries it. Use null when no exact contiguous quote is "
           "available.\n"
           # TWO QUOTES, TWO TEXTS: measured over 3 replicates, ALL 15 quote_not_found drops were
           # source-text phrases. The judge copied from the SOURCE for span_quote while the same
           # reply's evidence_spans legitimately asks for source text. Name the contrast and give
           # the countable check. Sent only with one claim per call: in an A/B it raised yield
           # from 33% to 68%, but the single whole-brief call lost all verification in 1 of 3.
           + ("THE TWO QUOTE FIELDS COME FROM DIFFERENT TEXTS — DO NOT SWAP THEM: span_quote is "
              "copied from the CLAIM/SUPPORTING sentences printed with the claim, NEVER from the "
              "SOURCE text; support_span (evidence_spans) is copied from the SOURCE text, never from "
              "the claim. CHECK before answering: the exact characters of every span_quote must "
              "appear in the claim block above — if they only appear in the SOURCE, you have quoted "
              "the wrong text; use null instead.\n"
              if sharded else "")
           + ("span_relation is about the SENTENCE you quoted, not the whole claim: 'says' means "
              "this source itself states that sentence's content; any other verb names the step "
              "that sentence takes beyond this source.\n"
              if span_relation_field else "")
           + "\n" if citation_spans_enabled else "")
        + ("Return every available supporting passage as an evidence_spans row for its cited source. "
           "support_span must be copied exactly from that SOURCE. When one contiguous passage cannot "
           "carry the claim, set multi_span_required true and return the several passages.\n\n"
           if claim_evidence_candidates is not None else "")
        + ("relation and strength are JUDGEMENTS, not evidence. Use says only when the source "
           "directly says the claim; name how the claim follows with the other implication verbs. strength "
           "is your honest 0-100 grade and may be low.\n\n"
           if claim_evidence_candidates is not None else "")
        + 'Reply with ONLY this JSON: {"verdicts": [ ... ]} — one entry per claim, in order.\n\n'
        + "\n\n".join(blocks)
    )
    return prompt, checkable


def build_claim_prompt(report: dict, n: int, *, citation_spans_enabled: bool = False,
                       claim_evidence_candidates: dict | None = None,
                       span_fields_by_claim: dict | None = None,
                       sharded: bool = True, locating_quote: bool = False) -> str:
    """One claim's prompt: the SAME builder over a one-claim view of the report, so the per-claim
    instruction and payload stay byte-identical to the whole-brief prompt for that claim. The
    claim is numbered 1 by construction."""
    sub = {"claims": [report["claims"][n]]}
    sub_candidates = None
    if claim_evidence_candidates is not None:
        sub_candidates = {(0, i): text for (cn, i), text in claim_evidence_candidates.items()
                          if cn == n}
    sub_fields = None
    if span_fields_by_claim is not None:
        sub_fields = {0: span_fields_by_claim.get(n) or {}}
    prompt, checkable = build_prompt(sub, citation_spans_enabled=citation_spans_enabled,
                                     claim_evidence_candidates=sub_candidates,
                                     span_fields_by_claim=sub_fields,
                                     sharded=sharded, locating_quote=locating_quote)
    assert checkable == [0], f"claim {n} is not checkable in the one-claim view"
    return prompt


def completion_prompt(claim: dict, candidate: dict) -> str:
    """The optional evidence-completion pass: does one more item complete a partial claim?"""
    resolved = [cite for cite in claim["citations"]
                if cite["resolution"] != "unresolved" and cite.get("text")]
    prior = "\n".join(f"CITED SOURCE {i + 1} ({cite['source']}): {cite['text']}"
                      for i, cite in enumerate(resolved))
    return (
        "Judge only the delimited fetched excerpts. Does the CANDIDATE, combined with the "
        "already cited excerpts, directly support the immutable CLAIM? The candidate must add "
        "a material element absent from the cited set. Do not use outside knowledge.\n\n"
        "Return ONLY JSON with exactly: combined_verdict ('completed' or 'not_completed'), "
        "candidate_adds_material_support (boolean), why (factual, <=200 characters).\n\n"
        f"CLAIM:\n{claim['claim']}\n\n{prior}\n\nCANDIDATE ({candidate['source']}): "
        f"{candidate['text'][:MAX_SRC_CHARS]}"
    )

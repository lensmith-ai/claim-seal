"""Source digests, the judge's evidence windows, and the sealed claim-evidence record.

The evidence pass asks the judge, for a supported claim, to quote the passage of the captured
source document that carries it (support_span). The quote is then located MECHANICALLY in the
digest-pinned normalized document. Only identities are kept: offsets, sha256 digests, counts.
The source text itself is never written into the record.

Two axes are kept apart on purpose:
  evidence  mechanical: "present" (a span was located), "not-located", "no-text"
  relation  a JUDGEMENT: the verb the judge chose (or, when it named none, a defaulted verb, and
            relation_source says which). A model-only "supported" never becomes support: a
            supported state needs a located span.

States never collapse a "could not establish" into a definite negative:
  supported    judge said supported AND every proposed span located
  unsupported  judge said unsupported (its own word, EVIDENCE-MODEL-UNSUPPORTED), or said
               supported/partial but named the relation contradicts
               (EVIDENCE-RELATION-CONTRADICTS: the source disagrees, and that wins)
  unresolved   we tried and could not establish support (partial, not located, ambiguous, drift)
  unknown      we could not even look (no body, failed capture, undecodable, no verdict)
"""
from __future__ import annotations

import copy
from array import array
import functools
import hashlib
import html
import json
import re
import unicodedata

from .normalize import COORDINATE_SYSTEM, NORMALIZER_ID, normalize, normalizer_sha256
from .relations import derive_coarse_relation, read_relation
from .resolve import citation_key, claim_units, load_sources, shape_brief
from .seal import CLAIM_EVIDENCE_CONTRACT, canonical, seal

_EVIDENCE_VALUES = frozenset({"present", "not-located", "no-text"})

# Only capture-time disambiguation input, not a retained-quote cap: an ambiguous quote may grow
# by at most this many code points of surrounding context to become unique.
EVIDENCE_SPAN_EXTENSION_MAX_CODEPOINTS = 1536

# The judge's input window. The judge must see norm/v1 itself, or a quoted span could not be
# mapped back to digest-pinned bytes; but norm/v1 is the WHOLE document, and the first real brief
# produced a 4,115,910 character prompt against a 110,000 budget. The window is a SLICE of norm/v1,
# which keeps the mapping deterministic. It bounds what the judge READS, not what is retained.
EVIDENCE_WINDOW_BEHIND = 300     # a little context before the title
EVIDENCE_WINDOW_AHEAD = 2500     # an article's body follows its title
EVIDENCE_PROMPT_BUDGET = 90000   # total characters across every candidate in one prompt
EVIDENCE_WINDOW_FLOOR = 600      # below this, withhold rather than offer a misleading sliver
#
# THE WINDOW STOPS AT THE CITED ARTICLE'S OWN END. BEHIND and AHEAD are the most the judge reads;
# the window is also clipped to the article its anchor sits in (article_bounds below). Unclipped, a
# feed item whose own text is short ran on into the NEXT item of the same feed, and the judge backed
# a citation of the short item with a sentence from the next article; the characters behind could
# likewise reach the PREVIOUS item. The clip is a slice of norm/v1 like the window, so a quoted span
# still maps to the same digest-pinned bytes. A short article gives a short window that is the WHOLE
# article, not a budget sliver, so the FLOOR (a per-citation budget rule) is unchanged. A judge quote
# located only outside the cited article is not evidence for that citation: it counts as not located
# (build_claim_evidence).

# A cited title that cannot be found in its captured document, in any written form. The judge then
# gets no window for that citation: the document's first characters may belong to another item.
EVIDENCE_TITLE_NOT_FOUND = "EVIDENCE-TITLE-NOT-FOUND"

LINEAGE_RAW_DIGEST = "LINEAGE-RAW-DIGEST"
LINEAGE_NORMALIZED_DIGEST = "LINEAGE-NORMALIZED-DIGEST"
LINEAGE_NORMALIZE_FAILED = "LINEAGE-NORMALIZE-FAILED"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------ capture
def _raw_body(item: dict):
    """The captured document as bytes, or None when the capture failed (body: null)."""
    body = item["body"] if "body" in item else item.get("text")
    if body is None:
        return None
    if isinstance(body, (bytes, bytearray)):
        return bytes(body)
    return str(body).encode("utf-8")


def digests_for(raw: bytes) -> dict:
    """Digests recorded at capture time; every later use recomputes them and refuses on drift."""
    out = {"raw_sha256": _sha(raw), "normalized_sha256": None, "normalize_error": None}
    try:
        out["normalized_sha256"] = _sha(normalize(NORMALIZER_ID, raw).encode("utf-8"))
    except (UnicodeDecodeError, LookupError, KeyError) as exc:
        out["normalize_error"] = type(exc).__name__
    return out


def capture_sources(sources: dict) -> dict:
    """A copy of the sources with capture digests recorded on every item that lacks them.

    Digests an item already carries are left alone, so material captured earlier (and changed
    since) is refused downstream rather than silently re-pinned.
    """
    out = copy.deepcopy(load_sources(sources))
    for sid, entry in out.items():
        for n, item in enumerate(entry["items"]):
            item.setdefault("id", f"{sid}-{n + 1}")
            raw = _raw_body(item)
            if raw is None or "raw_sha256" in item:
                continue
            got = digests_for(raw)
            item["raw_sha256"] = got["raw_sha256"]
            item["normalized_sha256"] = got["normalized_sha256"]
            if got["normalize_error"]:
                item["normalize_error"] = got["normalize_error"]
    return out


def source_lineage(sources: dict) -> list[dict]:
    """One row per source item: identity and digests only, re-checked against the capture.

    capture_check is "ok", "no-body", or the rule that refused (LINEAGE-RAW-DIGEST,
    LINEAGE-NORMALIZED-DIGEST, LINEAGE-NORMALIZE-FAILED).
    """
    rows = []
    for sid, entry in load_sources(sources).items():
        for n, item in enumerate(entry["items"]):
            raw = _raw_body(item)
            row = {"response_id": str(item.get("id") or f"{sid}-{n + 1}"), "source": sid,
                   "key": citation_key(item), "url": item.get("url"),
                   "retrieved_at": item.get("retrieved_at"), "raw_sha256": None, "byte_length": None,
                   "normalizer_id": NORMALIZER_ID, "normalizer_sha256": normalizer_sha256(),
                   "coordinate_system": COORDINATE_SYSTEM, "normalized_sha256": None,
                   "raw_artifact": None}
            if raw is None:
                row["capture_check"] = "no-body"
                rows.append(row)
                continue
            row["raw_sha256"], row["byte_length"] = _sha(raw), len(raw)
            check = "ok"
            if row["raw_sha256"] != item.get("raw_sha256"):
                check = LINEAGE_RAW_DIGEST
            else:
                try:
                    row["normalized_sha256"] = _sha(normalize(NORMALIZER_ID, raw).encode("utf-8"))
                    if row["normalized_sha256"] != item.get("normalized_sha256"):
                        check = LINEAGE_NORMALIZED_DIGEST
                except (UnicodeDecodeError, LookupError, KeyError):
                    check = LINEAGE_NORMALIZE_FAILED
            row["capture_check"] = check
            rows.append(row)
    return rows


# ------------------------------------------------------------------ binding
def _binding(source: str, citation: dict, sources: dict):
    """Return one deterministically-bound captured document, or a closed (state, reason)."""
    entry = (sources or {}).get(source)
    if not isinstance(entry, dict):
        return None, "unknown", "EVIDENCE-NO-BODY"
    ref = str(citation.get("ref") or citation.get("label") or "")
    candidates = [item for item in entry.get("items") or [] if citation_key(item) == ref]
    if len(candidates) != 1:
        return (None, "unresolved" if candidates else "unknown",
                "EVIDENCE-SPAN-AMBIGUOUS" if candidates else "EVIDENCE-NO-BODY")
    item = candidates[0]
    raw = _raw_body(item)
    if raw is None:
        return None, "unknown", "EVIDENCE-FETCH-FAILED"
    if not isinstance(item.get("raw_sha256"), str) or _sha(raw) != item["raw_sha256"]:
        return None, "unresolved", "EVIDENCE-RAW-DIGEST"
    try:
        normalized = normalize(NORMALIZER_ID, raw)
    except (UnicodeDecodeError, LookupError, KeyError):
        return None, "unknown", "EVIDENCE-NORMALIZE-FAILED"
    if (not isinstance(item.get("normalized_sha256"), str)
            or _sha(normalized.encode("utf-8")) != item["normalized_sha256"]):
        return None, "unresolved", "EVIDENCE-NORMALIZED-DIGEST"
    # Where the judge's window starts. A document that holds only this item may fall back to its
    # start: that is still the cited item. A document other items of the source also carry (the
    # same captured bytes, e.g. one feed for every item) may not: its start is another item, so a
    # title that cannot be found there is unresolved and the judge gets no window for it. A caller
    # that attaches a shared feed body to ONE item only cannot be told apart from a one-item
    # document; give such items their own title as it appears in the body.
    shared = sum(1 for other in entry.get("items") or []
                 if isinstance(other, dict) and other.get("raw_sha256") == item["raw_sha256"]) > 1
    anchors = title_anchors(normalized, item.get("title"))
    if not anchors:
        if shared:
            return None, "unresolved", EVIDENCE_TITLE_NOT_FOUND
        anchors = [0]
    return {"response_id": str(item.get("id") or ""), "retrieved_at": item.get("retrieved_at"),
            "response_sha256": item["normalized_sha256"], "normalized": normalized,
            "raw_title": item.get("title"), "anchors": anchors}, None, None


# ------------------------------------------------------------------ title anchors
# The judge's window is anchored on the cited item's title inside its captured document. A feed
# writes a title as XML text, so "What? <br> Data" arrives as "What? &lt;br&gt; Data" and "R & D"
# as "R &amp; D"; a numeric entity (&#10;) or CDATA changes it again. Searching for the parsed
# title verbatim missed those, and the window used to fall back to offset 0, which in a feed is
# the FIRST item: a claim was judged against an unrelated article. The exact written forms are
# tried first (raw, JSON string, XML-escaped, CDATA), then the PLAIN form on both sides: CDATA
# markers dropped, tags removed, entities decoded (twice, for double-escaped feeds), whitespace
# collapsed. Plain offsets map back to the document. Nothing found means no window.
_CDATA_MARK_RE = re.compile(r"<!\[CDATA\[|\]\]>")
_TAG_RE = re.compile(r"<[^<>]*>")
_ENTITY_RE = re.compile(r"&(?:#[0-9]+|#[xX][0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]*);")
_SPACE_RE = re.compile(r"\s+")


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def _written_forms(title: str) -> list[str]:
    """Exact forms a document may carry the title in: raw, JSON string, XML-escaped, CDATA."""
    nfc = _nfc(title)
    forms = [nfc]
    for literal in (json.dumps(title, ensure_ascii=True)[1:-1], json.dumps(title, ensure_ascii=False)[1:-1],
                    html.escape(nfc, quote=False)):
        if literal not in forms:
            forms.append(literal)
    forms.append(f"<![CDATA[{nfc}]]>")
    return forms


def _rewrite_mapped(text: str, offsets, pattern, repl):
    out, out_offsets, pos = [], array("q"), 0
    for m in pattern.finditer(text):
        out.append(text[pos:m.start()])
        out_offsets.extend(offsets[pos:m.start()])
        new = repl(m)
        out.append(new)
        out_offsets.extend([offsets[m.start()]] * len(new))
        pos = m.end()
    out.append(text[pos:])
    out_offsets.extend(offsets[pos:])
    return "".join(out), out_offsets


def _plain_mapped(text: str):
    """(plain text, offset in `text` of each plain character; an int array, not a list, so a
    multi-megabyte page costs 8 bytes per character rather than an int object each)."""
    offsets = array("q", range(len(text)))
    for pattern, repl in ((_CDATA_MARK_RE, lambda m: ""), (_TAG_RE, lambda m: " "),
                          (_ENTITY_RE, lambda m: html.unescape(m.group(0))), (_TAG_RE, lambda m: " "),
                          (_ENTITY_RE, lambda m: html.unescape(m.group(0))), (_TAG_RE, lambda m: " "),
                          (_SPACE_RE, lambda m: " ")):
        text, offsets = _rewrite_mapped(text, offsets, pattern, repl)
    return text, offsets


def plain_title(title) -> str:
    """A title in the plain form both sides are compared in."""
    if not isinstance(title, str):
        return ""
    return _plain_mapped(_nfc(title))[0].strip()


def title_anchors(normalized: str, title, limit: int = 6) -> list[int]:
    """Offsets in the document where the title starts: exact written forms first, then the plain
    form. An empty list means NOT FOUND; never substitute another offset for it."""
    if not isinstance(title, str) or not title.strip():
        return []
    for needle in _written_forms(title):
        starts, i = [], normalized.find(needle)
        while i >= 0 and len(starts) < limit:
            starts.append(i)
            i = normalized.find(needle, i + 1)
        if starts:
            return starts
    want = plain_title(title)
    if not want:
        return []
    plain, offsets = _plain_mapped(normalized)
    starts, i = [], plain.find(want)
    while i >= 0 and len(starts) < limit:
        starts.append(offsets[i])
        i = plain.find(want, i + 1)
    return starts


def best_anchor(normalized: str, starts, behind: int, ahead: int) -> int:
    """The anchor whose window carries the least markup (see evidence_window)."""
    def markup_share(ix: int) -> float:
        lo, hi = _window_span(normalized, ix, behind, ahead)
        w = normalized[lo:hi]
        return (w.count("<") + w.count(">")) / max(1, len(w))

    return min(starts, key=markup_share)


def _window_span(normalized: str, ix: int, behind: int, ahead: int) -> tuple[int, int]:
    """[lo, hi) of the judge's window at anchor ix: BEHIND/AHEAD, clipped to the cited article."""
    lo, hi = article_bounds(normalized, ix)
    return max(lo, ix - behind), min(hi, ix + ahead)


# ------------------------------------------------------------------ article bounds
# norm/v1 keeps every tag, so an article's edges are read from the markup, in this order:
#   feed  the innermost <item>...</item> or <entry>...</entry> holding the anchor;
#   HTML  else the innermost <article>...</article> holding it;
#         else, when the anchor sits inside an <hN> heading, from that heading's start to the next
#         heading of the same or a higher level (<h1>..<hN>), or the end of the text.
# An anchor in none of these (plain text, JSON, an HTML <title>) is not clipped: (0, len).
# Deterministic: tag names are matched case-insensitively, an element closes at its matching close
# tag (nesting counted), an unclosed or self-closed element bounds nothing.
_FEED_ELEMENT_RE = re.compile(r"<(/?)(item|entry)(?=[\s>/])", re.IGNORECASE)
_ARTICLE_ELEMENT_RE = re.compile(r"<(/?)(article)(?=[\s>/])", re.IGNORECASE)
_HEADING_OPEN_RE = re.compile(r"<h([1-6])(?=[\s>/])", re.IGNORECASE)


def _element_ranges(text: str, pattern) -> list[tuple[int, int]]:
    out, stack = [], []
    for m in pattern.finditer(text):
        name = m.group(2).lower()
        if m.group(1):
            if stack and stack[-1][1] == name:
                lo = stack.pop()[0]
                gt = text.find(">", m.end())
                out.append((lo, len(text) if gt < 0 else gt + 1))
            continue
        gt = text.find(">", m.end())
        if gt >= 0 and text[gt - 1] == "/":
            continue                    # <item/> holds nothing
        stack.append((m.start(), name))
    return out


def _heading_ranges(text: str) -> list[tuple[int, int, int]]:
    out = []
    for m in _HEADING_OPEN_RE.finditer(text):
        close = re.compile(rf"</h{m.group(1)}\s*>", re.IGNORECASE).search(text, m.end())
        if close:
            out.append((m.start(), close.end(), int(m.group(1))))
    return out


@functools.lru_cache(maxsize=16)
def _boundaries(text: str):
    """Element ranges of one document, computed once per document, not once per anchor."""
    return (_element_ranges(text, _FEED_ELEMENT_RE), _element_ranges(text, _ARTICLE_ELEMENT_RE),
            _heading_ranges(text))


def article_bounds(text: str, ix: int) -> tuple[int, int]:
    """[lo, hi) of the article the offset ix sits in (see the block above); (0, len) if none."""
    feed, articles, headings = _boundaries(text)
    for ranges in (feed, articles):
        holding = [r for r in ranges if r[0] <= ix < r[1]]
        if holding:
            return min(holding, key=lambda r: r[1] - r[0])
    for start, end, level in headings:
        if start <= ix < end:
            nxt = next((s for s, _e, lv in headings if s >= end and lv <= level), len(text))
            return start, nxt
    return 0, len(text)


# ------------------------------------------------------------------ judge windows
def evidence_window(normalized: str, raw_title, behind: int, ahead: int, anchors=None) -> str:
    """A slice of norm/v1 anchored on the title occurrence whose window is most PROSE-like.

    An HTML document carries its title at least twice: in <title> (surrounded by head markup) and
    in the body heading (surrounded by the article). Anchoring on the FIRST find handed the judge
    the head's markup. Candidates are scored by markup density and the cleanest window wins; a
    plain-text document has one occurrence and behaves exactly as before.

    The title is found in any written form (see title_anchors); `anchors` passes offsets already
    found. A title that is not found gives an EMPTY window, never the document's first characters:
    in a feed those belong to another item. The window ends at the cited article's own end (see
    article_bounds).
    """
    starts = list(anchors) if anchors else title_anchors(normalized, raw_title)
    if not starts:
        return ""
    ix = best_anchor(normalized, starts, behind, ahead)
    lo, hi = _window_span(normalized, ix, behind, ahead)
    return normalized[lo:hi]


def evidence_candidates(brief: dict, sources: dict, *, enabled: bool = True) -> dict:
    """{(claim n, citation index): window text} for the judge. Transient: never serialized."""
    if not enabled:
        return {}
    brief = shape_brief(brief)
    sources = load_sources(sources)
    candidates: dict[tuple[int, int], str] = {}
    bindings: dict[tuple[int, int], dict] = {}
    for claim_n, (_, _, _, claim) in enumerate(claim_units(brief)):
        for citation_index, citation in enumerate(claim.get("sources") or []):
            citation = citation if isinstance(citation, dict) else {}
            binding, _, _ = _binding(str(citation.get("id") or ""), citation, sources)
            if binding is not None:
                bindings[(claim_n, citation_index)] = binding
    if not bindings:
        return candidates
    behind, ahead = EVIDENCE_WINDOW_BEHIND, EVIDENCE_WINDOW_AHEAD
    # Scale every window down together if the citation count would blow the budget, rather than
    # truncating the prompt later. Shrinking uniformly keeps the judge's view even across claims.
    per = behind + ahead
    if per * len(bindings) > EVIDENCE_PROMPT_BUDGET:
        room = EVIDENCE_PROMPT_BUDGET // len(bindings)
        if room < EVIDENCE_WINDOW_FLOOR:
            return candidates          # too many citations to show any of them honestly
        behind = max(0, round(room * behind / per))
        ahead = room - behind
    for key, binding in bindings.items():
        window = evidence_window(binding["normalized"], binding.get("raw_title"), behind, ahead,
                                 binding.get("anchors"))
        if window:
            candidates[key] = window
    return candidates


# ------------------------------------------------------------------ the record
def judged_relation_and_strength(verdict: dict) -> tuple[str, int, str]:
    """Read the judge's relation word through read_relation; unreadable input is never a definite
    negative. A vocabulary verb is the judge's; a synonym is read as its verb and an unknown word as
    "undetermined", both marked compat-default (and kept raw on the row by the caller).

    Returns the verb, the strength, and WHERE THE VERB CAME FROM ("judge" or "compat-default").
    A verdict of "partial" tells us the claim is inferential; it does not tell us the writer meant
    "implies" rather than "suggests". Defaulting to a verb and presenting it as the judge's word
    would overclaim precision.
    """
    semantic = str(verdict.get("verdict") or "").lower()
    # read_relation is shared with the claim-level guard in support.py and the per-sentence intake
    # in spans.py (added in this library: it also trims), so "contradicts " or "contradicted"
    # cannot pass one guard and fail another. The mapping below is the composer's.
    relation, how = read_relation(verdict.get("relation"))
    source = "judge"
    if how != "verb":
        source = "compat-default"
        if how == "empty":
            relation = {"supported": "says", "partial": "implies"}.get(semantic, "undetermined")
    strength = verdict.get("strength")
    if not isinstance(strength, int) or isinstance(strength, bool) or not 0 <= strength <= 100:
        strength = 100 if relation == "says" else (25 if derive_coarse_relation(relation) == "inferred" else 0)
    return relation, strength, source


def _mechanical_evidence(binding: dict | None, outcome: dict) -> str:
    """Derived from the binding and exact location, never from model output."""
    if binding is None:
        return "no-text"
    return "present" if isinstance(outcome.get("evidence_spans"), list) and outcome["evidence_spans"] else "not-located"


def _all_occurrences(text: str, needle: str) -> list[int]:
    """Overlapping exact occurrences, so an ambiguous quote is never silently selected."""
    starts: list[int] = []
    start = text.find(needle)
    while start >= 0:
        starts.append(start)
        start = text.find(needle, start + 1)
    return starts


def _extend_ambiguous_span(normalized: str, proposed: str) -> tuple[int, int, dict] | None:
    """Locate one unique same-document superset, or refuse without choosing an occurrence."""
    occurrences = _all_occurrences(normalized, proposed)
    if len(occurrences) == 1:
        start = occurrences[0]
        return start, start + len(proposed), {"applied": False,
                                                "proposed_sha256": _sha(proposed.encode("utf-8")),
                                                "proposed_codepoints": len(proposed),
                                                "before_codepoints": 0, "after_codepoints": 0}
    if len(occurrences) < 2:
        return None
    for total in range(1, EVIDENCE_SPAN_EXTENSION_MAX_CODEPOINTS + 1):
        unique: list[tuple[int, int, dict]] = []
        for start in occurrences:
            end = start + len(proposed)
            before = min(total // 2, start)
            after = min(total - before, len(normalized) - end)
            remaining = total - before - after
            before += min(remaining, start - before)
            remaining = total - before - after
            after += min(remaining, len(normalized) - end - after)
            extended = normalized[start - before:end + after]
            if normalized.count(extended) == 1:
                unique.append((start - before, end + after,
                               {"applied": True, "proposed_sha256": _sha(proposed.encode("utf-8")),
                                "proposed_codepoints": len(proposed), "before_codepoints": before,
                                "after_codepoints": after}))
        if len(unique) == 1:
            return unique[0]
    return None


def claim_ref(kind: str, item_index: int, analysis_index, claim: dict) -> dict:
    """A claim's identity on its evidence rows: its position and a digest of its text."""
    field = next((name for name in ("text", "insight", "headline") if isinstance(claim.get(name), str)), "text")
    text = claim.get(field) if isinstance(claim.get(field), str) else ""
    return {"kind": kind, "item_index": item_index, "analysis_index": analysis_index,
            "field": field, "claim_sha256": _sha(text.encode("utf-8"))}


def build_claim_evidence(brief: dict, sources: dict, model_verdicts: dict | None,
                         deciders: dict | None = None) -> dict:
    """Build the identity-only outcome rows and seal them.

    `model_verdicts` maps claim number (claim_units order) to the judge's parsed entry. Sources
    must carry capture digests (see capture_sources); missing or drifted digests are refused.

    `deciders` (only with the optional Jev first pass, claim_seal.jev) maps claim number to the
    decider block written into every row of that claim, inside the seal. A row of a claim Jev
    decided records relation_source "jev-first-pass". None (the default) adds nothing.
    """
    brief = shape_brief(brief)
    sources = load_sources(sources)
    outcomes: list[dict] = []

    def append_outcome(binding: dict | None, outcome: dict) -> None:
        outcome["evidence"] = _mechanical_evidence(binding, outcome)
        outcome["relation_coarse"] = derive_coarse_relation(outcome["relation"])
        outcome["relation_is_judgement"] = True
        if outcome["relation"] == "says" and outcome["evidence"] != "present":
            # says requires a located span. Failing to locate one does NOT establish that the
            # source is unrelated; it establishes that we could not check.
            outcome.update({"relation": "undetermined", "relation_coarse": "undetermined", "strength": 0})
        outcomes.append(outcome)

    for claim_n, (kind, item_index, analysis_index, claim) in enumerate(claim_units(brief)):
        ref = claim_ref(kind, item_index, analysis_index, claim)
        verdict = (model_verdicts or {}).get(claim_n) or {}
        for citation_index, citation in enumerate(claim.get("sources") or []):
            citation = citation if isinstance(citation, dict) else {}
            source = citation.get("id")
            binding, forced_state, forced_reason = _binding(str(source or ""), citation, sources)
            relation, strength, relation_source = judged_relation_and_strength(verdict)
            outcome = {"claim_ref": ref, "citation_index": citation_index, "source": source,
                       "response_id": binding.get("response_id") if binding else None,
                       "retrieved_at": binding.get("retrieved_at") if binding else None,
                       "support_state": forced_state or "unresolved",
                       "state_reason": forced_reason or "EVIDENCE-SPAN-NOT-PROPOSED",
                       "evidence": "no-text", "relation": relation,
                       "relation_coarse": derive_coarse_relation(relation),
                       "relation_is_judgement": True, "relation_source": relation_source,
                       "corroboration": 0, "strength": strength,
                       "evidence_spans": [], "span_counts": {"proposed": 0, "located": 0},
                       "display_policy": {"text_retained": False, "text_displayable": False}}
            decider = deciders.get(claim_n) if deciders is not None else None
            if decider is not None:
                outcome["decider"] = copy.deepcopy(decider)
                if decider.get("by") == "jev":
                    outcome["relation_source"] = "jev-first-pass"
            raw_word = verdict.get("relation")
            if read_relation(raw_word)[1] in ("synonym", "unknown"):
                # Added in this library: the judge's own word and how it was read, kept in the
                # (sealed) row whenever it was not a vocabulary verb.
                outcome["relation_raw"] = str(raw_word).strip()[:60]
                outcome["relation_read"] = read_relation(raw_word)[1]
            if binding is None:
                append_outcome(binding, outcome)
                continue
            semantic = str(verdict.get("verdict") or "").lower()
            if semantic in ("supported", "partial") and relation == "contradicts":
                # Added in this library (not in the composer), revised on a measured finding: a
                # judge that names "contradicts" has said the source disagrees, whatever verdict
                # word came with it. In a preregistered agreement test on public refutation items,
                # 6 of 80 came back partial + contradicts; recording those as anything weaker hid
                # real refutations. The row is unsupported, keeps the verb, and names the override.
                outcome.update({"support_state": "unsupported",
                                "state_reason": "EVIDENCE-RELATION-CONTRADICTS"})
                append_outcome(binding, outcome)
                continue
            if semantic == "unsupported":
                outcome.update({"support_state": "unsupported",
                                "state_reason": "EVIDENCE-MODEL-UNSUPPORTED"})
                append_outcome(binding, outcome)
                continue
            if semantic == "partial":
                # "Partial" is NOT "unsupported": recording it so would assert more than the judge
                # said. Support was not established, so the state stays unresolved, while an
                # exactly located span may still establish mechanical evidence.
                outcome.update({"support_state": "unresolved",
                                "state_reason": "EVIDENCE-MODEL-PARTIAL"})
            elif semantic != "supported":
                outcome.update({"support_state": "unknown", "state_reason": "EVIDENCE-VERIFIER-NOT-RUN"})
                append_outcome(binding, outcome)
                continue
            proposals = verdict.get("evidence_spans")
            matching = ([p for p in proposals if isinstance(p, dict) and p.get("citation_index") == citation_index]
                        if isinstance(proposals, list) else [])
            outcome["span_counts"]["proposed"] = len(matching)
            if not matching:
                append_outcome(binding, outcome)
                continue
            failures = []
            for proposal in matching:
                span = proposal.get("support_span")
                if not isinstance(span, str) or not span:
                    failures.append("not-proposed")
                    continue
                located = _extend_ambiguous_span(binding["normalized"], span)
                if not _all_occurrences(binding["normalized"], span):
                    failures.append("not-located")
                elif located is None:
                    failures.append("ambiguous")
                else:
                    start, end, extension = located
                    # The quote itself (before any disambiguating extension) must sit inside the
                    # cited article: a sentence found only in the next item is not this citation's
                    # evidence, whatever the judge said (see EVIDENCE_WINDOW_FLOOR's note).
                    core = (start + extension["before_codepoints"], end - extension["after_codepoints"])
                    if not any(lo <= core[0] and core[1] <= hi for lo, hi in
                               (article_bounds(binding["normalized"], a) for a in binding["anchors"])):
                        failures.append("not-located")
                        continue
                    located_text = binding["normalized"][start:end]
                    evidence_span = {
                        "normalizer": NORMALIZER_ID, "coordinate_system": COORDINATE_SYSTEM,
                        "start": start, "end": end, "sha256": _sha(located_text.encode("utf-8")),
                        "response_sha256": binding["response_sha256"],
                    }
                    if extension["applied"]:
                        evidence_span["extension"] = extension
                    if (binding["normalized"][start:end] != located_text or
                            _sha(binding["normalized"][start:end].encode("utf-8")) != evidence_span["sha256"]):
                        raise AssertionError("extended evidence span failed capture-time mapping")
                    outcome["evidence_spans"].append(evidence_span)
            outcome["span_counts"]["located"] = len(outcome["evidence_spans"])
            if outcome["evidence_spans"]:
                outcome["state_reason"] = ("EVIDENCE-SPANS-PARTIALLY-LOCATED" if failures
                                           else "EVIDENCE-SPANS-LOCATED")
                if semantic == "supported" and not failures:
                    outcome["support_state"] = "supported"
            elif "ambiguous" in failures:
                outcome["state_reason"] = "EVIDENCE-SPAN-AMBIGUOUS"
            elif "not-located" in failures:
                outcome["state_reason"] = "EVIDENCE-SPAN-NOT-LOCATED"
            append_outcome(binding, outcome)
    claims: dict[bytes, list[dict]] = {}
    for outcome in outcomes:
        if outcome["evidence"] not in _EVIDENCE_VALUES:
            raise AssertionError("invalid mechanical evidence state")
        claims.setdefault(canonical(outcome["claim_ref"]), []).append(outcome)
    for rows in claims.values():
        corroboration = len({str(row.get("source")) for row in rows
                             if row["evidence"] == "present" and row["relation_coarse"] in {"stated", "inferred"}
                             and row.get("source")})
        for row in rows:
            row["corroboration"] = corroboration
    record = {"contract": CLAIM_EVIDENCE_CONTRACT, "outcomes": outcomes,
              "verification": {"capture_time_result": "verified",
                               "rule_ids": sorted({row["state_reason"] for row in outcomes})}}
    return seal(record)

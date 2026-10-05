"""Pass 1, RESOLUTION: does each cited reference exist in the source material? Pure code.

This catches a fabricated citation: a reference to something that was never in front of the
writer. That is the failure a language model cannot be trusted to self-report, and the one that
most embarrasses a reader who follows it up. It costs nothing, so it always runs first, and an
unresolved citation is never sent to pass 2: asking a model whether a claim is supported by a
source we could not find would produce a confident answer about nothing.

INPUTS
  brief    {"items": [{"insight": str, "why_it_matters"?: str, "detail"?: [str],
                       "sources": [{"id": source id, "ref": cited title}],
                       "analysis"?: [{"text": str, "sources": [...]}]}],
            "connection"?: {"text": str, "sources": [...]}}
           A claim's text is read from "text", then "insight", then "headline".
           One rule for a lone value where a list is expected (shape_brief): a string "detail"
           is ONE detail sentence, [detail], never split; a single object in "sources" (a
           citation) or "analysis" (a statement), or in "items", is a one-element list. Missing,
           None and [] are no value. Shapes that cannot be read without inventing or losing
           content raise BriefError: e.g. a detail of 0, False or {}, a string analysis
           statement (alone or in a list), a connection that is a list (even of one) or a string,
           a claim with no text and a citation with no "id" (so {} and [{}] are refused alike).
  sources  {source id: {"label"?: str, "items": [{"title": str, "text": str, "url"?: str,
                                                 "body"?: str, "retrieved_at"?: str,
                                                 "id"?: str}]}}
           or {source id: [items]}, optionally wrapped as {"sources": {...}}.
           A citation names an item by its "title" (or "key" when present). Pass 2 reads the
           item's title and text joined (TEXT_FIELDS). "body" is the full captured document
           the evidence pass quotes from; it defaults to the text.
"""
from __future__ import annotations

import copy
import re

MAX_SRC_CHARS = 700          # per cited item, in the verification prompt
FUZZY_MIN = 0.6              # token-overlap floor for a fuzzy title match
TEXT_FIELDS = ("title", "text")


class SourceError(ValueError):
    """The source material does not have the shape the checker needs."""


class BriefError(ValueError):
    """The brief has a shape the checker cannot read without inventing or losing content."""


def _kind(value) -> str:
    return {str: "a string", dict: "an object", list: "a list", bool: "a boolean"}.get(
        type(value), "a number" if isinstance(value, (int, float)) else type(value).__name__)


def detail_list(detail, where: str = "detail") -> list:
    """An item's detail as a list. A string is ONE sentence: [detail], never split.

    Iterating a string walks its characters, so without this a writer that put the detail
    paragraph in a string got one "line" per character. The rule is deliberately the plain one:
    splitting into sentences would put a sentence splitter's guesses into the span field names
    ("detail[2]") that the checked record cites. Missing, None, [] and "" are no detail.
    Anything else that is not a list or a string is refused, including 0, False and {}: an
    object's keys would be read as sentences and its values lost, and a number or boolean has no
    sentence to read. (shape_brief also refuses a list entry that is not a string; this
    per-item reader leaves such an entry for its caller to skip, so the attribution heuristic,
    which must never raise, can still be called on a hand-built item.)
    """
    if detail is None or isinstance(detail, str):
        return [detail] if detail else []
    if not isinstance(detail, list):
        raise BriefError(f"{where} must be a list of strings or one string, not {_kind(detail)}")
    return detail


def _object_list(value, where: str, what: str):
    """(list, changed) for a field that holds a list of objects. A lone object is [object].

    Missing, None and [] are empty. Any other value that is not a list or an object (a string,
    a number, a boolean, including "", 0 and False) is refused.
    """
    if value is None:
        return value, False
    if isinstance(value, dict):
        return [value], True
    if not isinstance(value, list):
        raise BriefError(f"{where} must be a list of {what} objects or one {what} object, "
                         f"not {_kind(value)}")
    return value, False


def _shape_claim(obj: dict, where: str, apply: bool) -> bool:
    """Check (and with apply, fix) one claim object and its sources. True when a fix is needed.

    A claim with no text, and a citation with no source id, are refused rather than read: the
    first would be checked as an empty claim and the second as a citation of nothing, so either
    would put a verdict (and a sealed row) on content nobody wrote. This is what keeps a lone {}
    from becoming [{}], an invented claim or citation, and it treats [{}] the same way.
    """
    if not (obj.get("text") or obj.get("insight") or obj.get("headline")):
        raise BriefError(f'{where} must have claim text in "text", "insight" or "headline"')
    sources, changed = _object_list(obj.get("sources"), f"{where}.sources", "citation")
    for n, cite in enumerate(sources or []):
        if not isinstance(cite, dict):
            raise BriefError(f"{where}.sources[{n}] must be a citation object "
                             f'{{"id": ..., "ref": ...}}, not {_kind(cite)}')
        if not cite.get("id"):
            raise BriefError(f'{where}.sources[{n}] must name its source with "id"')
    if changed and apply:
        obj["sources"] = sources
    return changed


def _shape_detail(obj: dict, where: str, apply: bool) -> bool:
    """Check (and with apply, fix) one object's detail. True when a fix is needed."""
    detail = obj.get("detail")
    shaped = detail_list(detail, f"{where}.detail")
    for k, sentence in enumerate(shaped):
        if not isinstance(sentence, str):  # it would be skipped unseen
            raise BriefError(f"{where}.detail[{k}] must be a string, not {_kind(sentence)}")
    if detail and shaped is not detail:
        if apply:
            obj["detail"] = shaped
        return True
    return False


def _shape(brief: dict, apply: bool) -> bool:
    changed = False
    items, fix = _object_list(brief.get("items"), "items", "item")
    if fix and apply:
        brief["items"] = items
    changed |= fix
    for i, item in enumerate(items or []):
        where = f"items[{i}]"
        if not isinstance(item, dict):
            raise BriefError(f"{where} must be an object, not {_kind(item)}")
        changed |= _shape_detail(item, where, apply)
        changed |= _shape_claim(item, where, apply)
        analysis, fix = _object_list(item.get("analysis"), f"{where}.analysis", "statement")
        if fix and apply:
            item["analysis"] = analysis
        changed |= fix
        for k, statement in enumerate(analysis or []):
            if not isinstance(statement, dict):  # it has no sources, so it would be dropped unseen
                raise BriefError(f"{where}.analysis[{k}] must be a statement object "
                                 f'{{"text": ..., "sources": [...]}}, not {_kind(statement)}')
            changed |= _shape_claim(statement, f"{where}.analysis[{k}]", apply)
    conn = brief.get("connection")
    if conn is not None and not isinstance(conn, dict) and conn != []:
        # One connection per brief. A non-empty list is refused, a list of one included:
        # the rule only ever wraps a lone value into a list, never unwraps one, and a list of
        # two would have no reading at all. A string has no sources to check.
        raise BriefError(f"connection must be one object {{\"text\": ..., \"sources\": [...]}}, "
                         f"not {_kind(conn)}")
    if isinstance(conn, dict):
        changed |= _shape_detail(conn, "connection", apply)
        changed |= _shape_claim(conn, "connection", apply)
    return changed


def shape_brief(brief: dict, *, in_place: bool = False) -> dict:
    """The brief in the shape every pass reads, under one rule for a lone value where a list is
    expected: a string "detail" is [detail]; a single object in "items", "sources" or
    "analysis" is [object]. Missing, None and [] (and "" for a detail) are empty and left as
    they are.

    Refused with BriefError, because no reading of them is safe, under one reason: content that
    would be invented, or dropped without being seen. A detail that is an object, a number or a
    boolean (0, False and {} included), or a detail entry that is not a string; items, sources or
    analysis that are a string, a number or a boolean; an item, a citation or an analysis
    statement that is not an object (a string statement has no sources, so it would be dropped
    unseen, whether alone or inside a list); a connection that is not one object (a non-empty
    list, even of one, or a string); an item, statement or connection with no claim text, and a
    citation with no source "id" (either would be checked, and sealed, as content nobody wrote;
    this is why a lone {} is refused rather than read as [{}], and [{}] the same way).

    By default the caller's brief is never modified: a well-shaped brief comes back as the same
    object, and one that needs a fix comes back as a fixed copy. in_place=True fixes the brief
    itself; the passes that annotate the brief they are given (verify, resolve_only, and
    judge's copy) use it, so a span's field name ("detail[0]") addresses the brief it sits in.
    """
    if not isinstance(brief, dict):
        raise BriefError(f"the brief must be an object with an \"items\" list, not {_kind(brief)}")
    if in_place:
        _shape(brief, apply=True)
        return brief
    if not _shape(brief, apply=False):
        return brief
    shaped = copy.deepcopy(brief)
    _shape(shaped, apply=True)
    return shaped


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _tokens(s: str) -> set:
    return {t for t in re.split(r"[^a-z0-9]+", _norm(s)) if len(t) > 2}


def _overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def load_sources(sources: dict) -> dict:
    """Normalize the accepted source shapes to {source id: {"label": str, "items": [dict]}}."""
    if not isinstance(sources, dict):
        raise SourceError("sources must be a JSON object keyed by source id")
    if isinstance(sources.get("sources"), dict) and set(sources) <= {"sources", "about"}:
        sources = sources["sources"]
    out = {}
    for sid, entry in sources.items():
        if isinstance(entry, list):
            entry = {"items": entry}
        if not isinstance(entry, dict) or not isinstance(entry.get("items", []), list):
            raise SourceError(f"source {sid!r} must be a list of items or {{'items': [...]}}")
        out[str(sid)] = {"label": entry.get("label") or str(sid),
                         "items": [it for it in entry.get("items") or [] if isinstance(it, dict)]}
    return out


def citation_key(item: dict) -> str:
    return str(item.get("key") or item.get("title") or "")


def index_sources(sources: dict, text_fields: tuple = TEXT_FIELDS) -> dict:
    """Flatten the sources into {source_id: [{key, title, text, url}, ...]}.

    `key` is what a citation is expected to name; `text` is what pass 2 gets to read. The item's
    own URL is carried so a resolved citation can be followed: resolution knows exactly which
    item a citation matched, so this is the one place a reference turns back into a link, and
    the model never sees or invents it.
    """
    out: dict = {}
    for sid, entry in load_sources(sources).items():
        rows = []
        for it in entry["items"]:
            missing = [field for field in text_fields if field not in it]
            if missing:
                raise SourceError(f"{sid}: text field(s) missing from source item: {missing}")
            key = citation_key(it)
            text = " ".join(filter(None, (str(it.get(field) or "") for field in text_fields)))
            rows.append({"key": key, "title": it.get("title") or key, "text": text,
                         "url": (it.get("url") or None)})
        out[sid] = rows
    return out


def claim_units(brief: dict):
    """(kind, index, analysis_index, obj) for every checkable claim, in a fixed order:
    each item, then its analysis statements, then the connection."""
    units = []
    for i, it in enumerate(brief.get("items") or []):
        units.append(("item", i, None, it))
        for analysis_i, statement in enumerate(it.get("analysis") or []):
            if isinstance(statement, dict):
                units.append(("analysis", i, analysis_i, statement))
    conn = brief.get("connection")
    if isinstance(conn, dict):
        units.append(("connection", 0, None, conn))
    return units


def resolve(brief: dict, sources: dict, *, text_fields: tuple = TEXT_FIELDS) -> dict:
    """Pass 1. Attach a resolution (exact / fuzzy / unresolved) to every citation. No model."""
    brief = shape_brief(brief)
    idx = index_sources(sources, text_fields)
    report = {"claims": [], "totals": {"citations": 0, "exact": 0, "fuzzy": 0, "unresolved": 0}}
    for kind, i, analysis_i, obj in claim_units(brief):
        claim_text = obj.get("text") or obj.get("insight") or obj.get("headline") or ""
        cites = []
        for s in (obj.get("sources") or []):
            sid = s.get("id")
            ref = s.get("ref") or s.get("label") or ""
            rows = idx.get(sid) or []
            hit, how, best = None, "unresolved", 0.0
            for r in rows:
                if _norm(r["key"]) == _norm(ref) and _norm(ref):
                    hit, how, best = r, "exact", 1.0
                    break
            if hit is None:
                for r in rows:
                    ov = _overlap(r["key"], ref)
                    if ov > best:
                        hit, best = r, ov
                how = "fuzzy" if (hit is not None and best >= FUZZY_MIN) else "unresolved"
                if how == "unresolved":
                    hit = None
            report["totals"]["citations"] += 1
            report["totals"][how] += 1
            cites.append({"source": sid, "ref": ref, "resolution": how,
                          "match_score": round(best, 2) if how == "fuzzy" else None,
                          "text": (hit["text"][:MAX_SRC_CHARS] if hit else None),
                          "matched_title": (hit["title"] if hit else None),
                          "url": (hit.get("url") if hit else None)})
        report["claims"].append({"kind": kind, "index": i, "analysis_index": analysis_i,
                                 "claim": claim_text, "citations": cites})
    return report


def target_for_claim(brief: dict, claim: dict) -> dict | None:
    """The exact record that owns a resolved claim, including analysis statements."""
    if claim["kind"] == "item":
        return brief["items"][claim["index"]]
    if claim["kind"] == "analysis":
        return (brief["items"][claim["index"]].get("analysis") or [])[claim["analysis_index"]]
    return brief.get("connection")


CITATION_ROW_FIELDS = ("source", "ref", "resolution", "match_score", "url", "matched_title")


def resolve_only(brief: dict, sources: dict, receipts: dict, *,
                 text_fields: tuple = TEXT_FIELDS) -> dict:
    """Pass 1 only, for writing whose genre overstates on purpose (parody, caricature).

    A support verdict on a deliberate exaggeration is meaningless, and a run of partial and
    unsupported would read as a quality failure rather than the genre working. What still has
    to be true is that the thing being exaggerated EXISTS in the material, and that is pass 1.
    Every claim is marked "cites-real" or "unresolved", never "supported", because nobody
    checked that and the word would imply somebody had. Mutates `brief` in place, including
    the shape fixes of shape_brief.
    """
    shape_brief(brief, in_place=True)
    report = resolve(brief, sources, text_fields=text_fields)
    counts = {"cites-real": 0, "unresolved": 0}
    for c in report["claims"]:
        cites = c["citations"]
        ok = bool(cites) and any(x["resolution"] != "unresolved" for x in cites)
        v = {"verdict": "cites-real" if ok else "unresolved",
             "why": ("every source it names is real material from the given sources; the claim "
                     "itself is not checked for support") if ok else
                    "no cited reference matched anything in the given sources",
             "citations": [{k: x[k] for k in CITATION_ROW_FIELDS} for x in cites]}
        counts[v["verdict"]] += 1
        target = target_for_claim(brief, c)
        if isinstance(target, dict):
            target["verification"] = v
            for src_entry, cite in zip(target.get("sources") or [], cites):
                if cite.get("url"):
                    src_entry["url"] = cite["url"]
                    src_entry["resolved_title"] = cite.get("matched_title")
    rec = {"mode": "resolve-only", "status": "ok",
           "resolution": report["totals"], "verdicts": counts,
           "note": "pass 1 only: citations are checked for being real, not for support"}
    receipts["verification"] = rec
    return rec

"""Optional, default-OFF evidence completion for claims judged "partial".

For a partial claim, a deterministic, versioned scorer picks ONE other item from the given
sources that covers the claim words the cited set leaves uncovered, and asks the model whether
the two together directly support the claim. A "completed" answer appends that item as a
resolved citation and marks the claim "supported-completed", never plain "supported". Capped;
failed calls consume the budget. Receipts keep identities only, never source excerpts.
"""
from __future__ import annotations

from .prompts import completion_prompt
from .resolve import TEXT_FIELDS, _norm, _tokens, index_sources, resolve, target_for_claim

COMPLETION_CAP = 6
COMPLETION_SCORER_VERSION = "residual-overlap/v1"
_COMPLETION_STOP_WORDS = {
    "about", "after", "again", "also", "and", "are", "been", "but", "can", "does", "for",
    "from", "have", "into", "its", "more", "not", "only", "over", "that", "the", "their",
    "then", "this", "those", "they", "was", "what", "when", "with", "would",
}


def _completion_identity(source: str, key: str, title: str) -> dict:
    """Stable, excerpt-free item identity retained in completion receipts."""
    return {"source": source, "key": key, "title": title}


def _completion_candidate(claim: dict, fetched_index: dict) -> tuple[dict | None, dict]:
    """Pick one candidate using a deterministic, versioned score."""
    resolved = [cite for cite in claim["citations"]
                if cite["resolution"] != "unresolved" and cite.get("text")]
    cited = {(cite["source"], _norm(cite.get("matched_title") or cite.get("ref") or ""))
             for cite in resolved}
    claim_tokens = _tokens(claim["claim"]) - _COMPLETION_STOP_WORDS
    cited_tokens = set().union(*(_tokens(cite["text"]) for cite in resolved)) if resolved else set()
    residual = claim_tokens - cited_tokens
    ranked = []
    for source, rows in fetched_index.items():
        for row in rows:
            if not row.get("text") or (source, _norm(row["key"])) in cited:
                continue
            row_tokens = _tokens(row["text"])
            residual_overlap = len(residual & row_tokens)
            claim_overlap = len(claim_tokens & row_tokens)
            if residual_overlap:
                ranked.append(((residual_overlap, claim_overlap), source, _norm(row["key"]), row))
    if not ranked:
        return None, {"scorer_version": COMPLETION_SCORER_VERSION, "residual_tokens": len(residual)}
    ranked.sort(key=lambda entry: (-entry[0][0], -entry[0][1], entry[1], entry[2]))
    score, source, _, row = ranked[0]
    return {"source": source, **row}, {"scorer_version": COMPLETION_SCORER_VERSION,
                                       "score": {"residual_overlap": score[0], "claim_overlap": score[1]},
                                       "residual_tokens": len(residual)}


def _completion_decision(raw: object) -> tuple[str, bool, str]:
    if not isinstance(raw, dict):
        return "invalid", False, "completion output did not parse as a JSON object"
    verdict = str(raw.get("combined_verdict") or "").lower()
    adds = raw.get("candidate_adds_material_support")
    why = str(raw.get("why") or "")[:200]
    if verdict not in {"completed", "not_completed"} or not isinstance(adds, bool):
        return "invalid", False, why or "completion output failed the required schema"
    return ("completed", True, why) if verdict == "completed" and adds else ("not-completed", False, why)


def complete_partials(brief: dict, sources: dict, receipts: dict, *, enabled: bool,
                      cap: int = COMPLETION_CAP, model=None, decision_call=None,
                      text_fields: tuple = TEXT_FIELDS) -> dict:
    """Optionally complete existing partial claims. MUTATES `brief`.

    `decision_call(prompt) -> dict` returns the parsed decision directly (tests); otherwise
    `model(prompt) -> text` is called and its JSON extracted.
    """
    if not enabled:
        return {"status": "off", "cap": cap, "attempted": 0, "completed": 0,
                "not_completed": 0, "invalid": 0, "budget_skipped": 0, "events": []}
    from .support import _call, extract_json  # local: judge imports this module
    report = resolve(brief, sources, text_fields=text_fields)
    fetched_index = index_sources(sources, text_fields)
    rec = {"status": "ok", "cap": cap, "attempted": 0, "completed": 0,
           "not_completed": 0, "invalid": 0, "budget_skipped": 0, "events": []}
    for claim in report["claims"]:
        target = target_for_claim(brief, claim)
        verification = target.get("verification") if isinstance(target, dict) else None
        if not isinstance(verification, dict) or verification.get("verdict") != "partial":
            continue
        original = [cite for cite in claim["citations"] if cite["resolution"] != "unresolved" and cite.get("text")]
        event = {"claim": {"kind": claim["kind"], "index": claim["index"], "analysis_index": claim["analysis_index"]},
                 "original": [_completion_identity(c["source"], c["ref"], c.get("matched_title") or c["ref"]) for c in original]}
        candidate, deterministic = _completion_candidate(claim, fetched_index)
        event["deterministic"] = deterministic
        if not original or candidate is None:
            event.update({"status": "skipped", "why": "no eligible resolved candidate in the sources"})
            rec["events"].append(event)
            continue
        event["candidate"] = _completion_identity(candidate["source"], candidate["key"], candidate["title"])
        if rec["attempted"] >= cap:
            event.update({"status": "skipped", "why": "completion budget exhausted"})
            rec["budget_skipped"] += 1
            rec["events"].append(event)
            continue
        # The enabled/cap gates sit immediately before the model call.
        rec["attempted"] += 1
        try:
            if decision_call is not None:
                raw = decision_call(completion_prompt(claim, candidate))
            else:
                output, executed = _call(model, completion_prompt(claim, candidate), 180)
                raw = extract_json(output) if output else None
                event["model"] = {"executed": executed}
            status, promote, why = _completion_decision(raw)
        except Exception as exc:  # noqa: BLE001 - a failed call is an invalid decision
            status, promote, why = "invalid", False, f"{type(exc).__name__}: {exc}"[:200]
        event.update({"status": status, "why": why})
        verification["completion"] = {"status": status, "original_verdict": "partial",
                                      "candidate": event["candidate"], "deterministic": deterministic,
                                      "model": event.get("model"), "why": why}
        if promote and verification.get("verdict") == "partial":
            appended = {"id": candidate["source"], "ref": candidate["key"], "label": candidate["title"]}
            if candidate.get("url"):
                appended.update({"url": candidate["url"], "resolved_title": candidate["title"]})
            target.setdefault("sources", []).append(appended)
            verification.setdefault("citations", []).append(
                {"source": candidate["source"], "ref": candidate["key"], "resolution": "exact",
                 "match_score": None, "url": candidate.get("url"), "matched_title": candidate["title"]})
            verification["verdict"] = "supported-completed"
            rec["completed"] += 1
        elif status == "not-completed":
            rec["not_completed"] += 1
        else:
            rec["invalid"] += 1
        rec["events"].append(event)
    return rec

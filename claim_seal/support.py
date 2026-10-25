"""Pass 2, SUPPORT: for claims whose citations resolved, does the cited text support the claim?

The model is pluggable: pass any callable that takes a prompt string and returns the model's
text (see claim_seal.adapters). If the callable accepts a `timeout` keyword it is given the
remaining time budget.

WHAT A CLAIM VERDICT MEANS
  supported    the cited text states or directly implies the claim
  partial      the cited text is related and consistent, but does not carry the whole claim
  unsupported  the cited text does not support the claim
  unresolved   the citation did not match anything in the sources (pass 1; no model opinion)
  unchecked    the model gave no usable verdict for this claim. Never treated as passing.

Fails OPEN but LOUD: when the model fails, claims are marked unchecked rather than left to look
verified, and the overall status says so.
"""
from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import inspect
import json
import re
import time

from .attribution import attribution_checks
from .completion import COMPLETION_DISABLED, CompletionDisabled
from .evidence import (build_claim_evidence, capture_sources, claim_ref, evidence_candidates,
                       source_lineage)
from .prompts import build_claim_prompt, build_prompt
from .relations import SPAN_RELATIONS, read_relation
from .resolve import (CITATION_ROW_FIELDS, TEXT_FIELDS, load_sources, resolve, shape_brief,
                      target_for_claim)
from .spans import SPAN_CONTRACT, span_fields, span_receipt

# Per-claim deadline and pool size. The deadline is ABSOLUTE from submission: queue wait and any
# retry live inside it. Both are starting values to be measured against your model, not constants.
VERIFY_CLAIM_DEADLINE_S = 120
VERIFY_POOL_DEFAULT = 2
SHARD_REASONS = ("timeout", "transport", "malformed", "rate-limit", "pool-exhausted")

VERIFICATION_STATUSES = ("ok", "unsupported", "unresolved", "verifier-error", "partly-checked",
                         "off")


class ModelText(str):
    """The model's reply text, optionally carrying the name of the model that actually answered.

    Adapters return this so a receipt can record what EXECUTED, which may differ from what was
    configured. A plain str works too; the executed model is then recorded as unknown (None).
    """

    def __new__(cls, text: str, model: str | None = None):
        obj = super().__new__(cls, text)
        obj.model = model
        return obj


def extract_json(text):
    """Pull the JSON object out of model output (code fences, preamble, raw newlines)."""
    if not text:
        return None
    text = re.sub(r"```(json)?", "", str(text))
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    blob = text[start:end + 1]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        pass
    # The first brace is not always the answer's brace: a preamble can contain one. Try each
    # later `{` as a candidate start; the first that parses is the earliest real object.
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    for i in starts[:200]:
        if i <= start:
            continue
        for strict in (True, False):
            try:
                return json.loads(text[i:end + 1], strict=strict)
            except json.JSONDecodeError:
                continue
    # strict=False tolerates literal control characters inside strings (a raw newline in long
    # prose) and nothing else: no trailing commas, single quotes or unquoted keys.
    try:
        return json.loads(blob, strict=False)
    except json.JSONDecodeError:
        return None


def _accepts_timeout(fn) -> bool:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "timeout" or p.kind is inspect.Parameter.VAR_KEYWORD for p in params)


def _call(model, prompt: str, timeout: float | None):
    """(text or None, executed model name or None)."""
    out = model(prompt, timeout=timeout) if _accepts_timeout(model) else model(prompt)
    if out is None:
        return None, None
    return str(out), getattr(out, "model", None)


def _configured_name(model) -> str | None:
    name = getattr(model, "name", None)
    return name if isinstance(name, str) else None


# ---- per-claim (sharded) verification ----------------------------------------------------------
def merge_shard_verdicts(shards: list, checkable: list) -> tuple[dict, list]:
    """Pure merge by ORIGINAL claim id. A duplicate claim id or an id outside `checkable` is
    rejected (recorded, never guessed); a broken shard contributes nothing and its claim
    degrades to unchecked."""
    verdicts: dict = {}
    rejected: list = []
    seen: set = set()
    for shard in shards:
        n = shard.get("claim")
        if n in seen:
            rejected.append({"claim": n, "reason": "duplicate-claim-id"})
            continue
        seen.add(n)
        if n not in checkable:
            rejected.append({"claim": n, "reason": "claim-not-checkable"})
            continue
        if shard.get("status") == "ok" and isinstance(shard.get("verdict"), dict):
            verdicts[n] = shard["verdict"]
    return verdicts, rejected


def _shard_verdict_payload(v: dict, *, citation_spans_enabled: bool, claim_evidence: bool) -> dict:
    """The same projection the whole-brief parser applies to one verdict entry, plus the
    per-citation bearing sentences only the per-claim prompt asks for."""
    return {
        "verdict": str(v.get("verdict") or "").lower(),
        "why": str(v.get("why") or "")[:200],
        **({"citation_bears": v.get("citation_bears")}
           if isinstance(v.get("citation_bears"), list) else {}),
        **({"citation_spans": v.get("citation_spans")} if citation_spans_enabled else {}),
        # Kept on both paths (added in this library): the contradicts override reads it even
        # when the evidence pass is off and the prompt did not ask for it.
        "relation": v.get("relation"),
        **({"strength": v.get("strength"),
            "evidence_spans": v.get("evidence_spans"),
            "multi_span_required": v.get("multi_span_required")}
           if claim_evidence else {})}


def run_claim_shard(n: int, prompt: str, model, *, citation_spans_enabled: bool,
                    claim_evidence: bool, deadline_mono: float, submitted_mono: float) -> dict:
    """One claim, one bounded call. Retry only transport errors and empty answers, never
    malformed output; a rate limit (429) fails closed at once so a retry cannot amplify load."""
    started = time.monotonic()
    shard = {"claim": n, "status": "unchecked", "reason": None, "attempts": 0,
             "queue_wait_s": round(started - submitted_mono, 3),
             "prompt_sha256_8": hashlib.sha256(prompt.encode()).hexdigest()[:8],
             "model_executed": None, "wall_s": None, "verdict": None}
    remaining = deadline_mono - started
    if remaining <= 0:
        shard.update({"reason": "pool-exhausted",
                      "error": "queue wait consumed the per-claim deadline"})
        return shard
    out = None
    for attempt in (1, 2):
        shard["attempts"] = attempt
        remaining = deadline_mono - time.monotonic()
        if remaining <= 0:
            shard["reason"] = "timeout"
            break
        try:
            out, executed = _call(model, prompt, remaining)
        except Exception as exc:  # noqa: BLE001 - classified, recorded, fail-closed
            if "429" in str(exc):
                shard.update({"reason": "rate-limit", "error": str(exc)[:200]})
                break
            shard.update({"reason": "transport", "error": str(exc)[:200]})
            continue
        shard["model_executed"] = executed
        if out:
            break
        shard["reason"] = "timeout"  # empty answer: retryable, budget already counted down
    shard["wall_s"] = round(time.monotonic() - started, 3)
    if not out:
        shard["reason"] = shard["reason"] or "timeout"
        return shard
    obj = extract_json(out)
    entries = obj.get("verdicts") if isinstance(obj, dict) else None
    entry = None
    if isinstance(entries, list):
        matches = [v for v in entries if isinstance(v, dict) and str(v.get("claim")) == "1"]
        entry = matches[0] if len(matches) == 1 else None
    if entry is None:
        shard.update({"reason": "malformed",
                      "error": "shard output did not parse to exactly one claim-1 verdict"})
        return shard
    shard.update({"status": "ok", "reason": None,
                  "verdict": _shard_verdict_payload(entry,
                                                    citation_spans_enabled=citation_spans_enabled,
                                                    claim_evidence=claim_evidence)})
    return shard


def _verify_sharded(report: dict, checkable: list, model, *, citation_spans_enabled: bool,
                    claim_evidence_candidates: dict | None, span_fields_by_claim: dict | None,
                    locating_quote: bool, pool_size: int, deadline_s: float) -> dict:
    pool_size = max(1, int(pool_size))
    deadline_s = float(deadline_s)
    prompts = {n: build_claim_prompt(report, n, citation_spans_enabled=citation_spans_enabled,
                                     claim_evidence_candidates=claim_evidence_candidates,
                                     span_fields_by_claim=span_fields_by_claim,
                                     sharded=True, locating_quote=locating_quote)
               for n in checkable}
    shards = []
    submitted = time.monotonic()
    deadline_mono = submitted + deadline_s
    claim_evidence = claim_evidence_candidates is not None
    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as pool:
        futures = {pool.submit(run_claim_shard, n, prompts[n], model,
                               citation_spans_enabled=citation_spans_enabled,
                               claim_evidence=claim_evidence,
                               deadline_mono=deadline_mono, submitted_mono=submitted): n
                   for n in checkable}
        for fut in concurrent.futures.as_completed(futures):
            n = futures[fut]
            try:
                shard = fut.result()
            except Exception as exc:  # noqa: BLE001 - a worker exception is a terminal shard
                shard = {"claim": n, "status": "unchecked", "reason": "transport",
                         "attempts": 0, "error": f"worker: {type(exc).__name__}: {exc}"[:200],
                         "verdict": None}
            shards.append(shard)
    verdicts, rejected = merge_shard_verdicts(shards, checkable)
    return {"verdicts": verdicts, "shards": shards, "rejected": rejected,
            "pool_size": pool_size, "deadline_s": deadline_s}


# ---- the public support gate ------------------------------------------------------------------
SUPPORT_GATE_RULE = "support-not-located"


def located_claims(report: dict, brief: dict, record) -> set:
    """Report claim numbers that have digest-bound, located support in the claim-evidence record:
    a row for THAT claim (same position, same claim-text digest) whose support_state is
    supported, whose evidence is present, and whose every evidence span carries its offsets, its
    own digest and the digest of the source document it was located in."""
    rows = (record or {}).get("outcomes") if isinstance(record, dict) else None
    if not isinstance(rows, list):
        return set()
    supported: dict = {}
    for row in rows:
        spans = row.get("evidence_spans") if isinstance(row, dict) else None
        if (not isinstance(row, dict) or row.get("support_state") != "supported"
                or row.get("evidence") != "present" or not isinstance(spans, list) or not spans):
            continue
        if not all(isinstance(s, dict) and isinstance(s.get("start"), int) and isinstance(s.get("end"), int)
                   and isinstance(s.get("sha256"), str) and isinstance(s.get("response_sha256"), str)
                   for s in spans):
            continue
        ref = row.get("claim_ref") or {}
        supported.setdefault((ref.get("kind"), ref.get("item_index"), ref.get("analysis_index")),
                             []).append(ref)
    found = set()
    for n, c in enumerate(report["claims"]):
        target = target_for_claim(brief, c)
        if not isinstance(target, dict):
            continue
        expected = claim_ref(c["kind"], c["index"], c["analysis_index"], target)
        if expected in supported.get((c["kind"], c["index"], c["analysis_index"]), []):
            found.add(n)
    return found


# ---- the verification step ---------------------------------------------------------------------
def verify(brief: dict, sources: dict, receipts: dict, model, *,
           citation_spans_enabled: bool = True, sharded: bool = True,
           local_placement: bool = True, locating_quote: bool = False,
           claim_evidence_candidates: dict | None = None, claim_evidence_sink=None,
           completion: bool = False, completion_cap: int = 6,
           pool_size: int = VERIFY_POOL_DEFAULT,
           claim_deadline_s: float = VERIFY_CLAIM_DEADLINE_S,
           text_fields: tuple = TEXT_FIELDS, gate_support: bool = True) -> dict:
    """Run both passes. MUTATES `brief`: shape_brief fixes it in place (a string detail becomes
    [detail]), every claim gains a `verification` block, and the record
    is written to receipts["verification"] and returned. Most callers want judge() instead.

    THE SUPPORT GATE (gate_support=True, the default and what judge() uses): a claim is recorded
    "supported" only when the claim-evidence record returned by `claim_evidence_sink` holds a
    supported, located, digest-bound row for it (located_claims). Otherwise the judge's
    "supported" is recorded "unresolved" with verdict_override {"rule": "support-not-located",
    "judge_verdict": "supported"}; with no sink there is no record, so no claim is supported.
    gate_support=False returns the model's raw reading, ungated: a "supported" there is the
    model's word only and must never be shown to a reader as support.

    completion=True is refused (CompletionDisabled): see claim_seal.completion.

    A model exception on the whole-brief path propagates (the caller owns that failure); on the
    per-claim path it is classified per claim.
    """
    if completion:
        raise CompletionDisabled(COMPLETION_DISABLED)
    shape_brief(brief, in_place=True)
    labels = {sid: entry["label"] for sid, entry in load_sources(sources).items()}
    report = resolve(brief, sources, text_fields=text_fields)
    span_fields_by_claim = ({n: span_fields(brief, c) for n, c in enumerate(report["claims"])}
                            if citation_spans_enabled else None)
    prompt, checkable = build_prompt(report, span_fields_by_claim=span_fields_by_claim,
                                     citation_spans_enabled=citation_spans_enabled,
                                     claim_evidence_candidates=claim_evidence_candidates,
                                     sharded=sharded, locating_quote=locating_quote)
    rec: dict = {"resolution": report["totals"], "model": None, "status": "skipped",
                 "claims_checked": len(checkable),
                 "claims_total": len(report["claims"])}
    verdicts: dict = {}
    span_status = None
    executed_model = None
    if checkable and sharded:
        started = time.monotonic()
        out = _verify_sharded(report, checkable, model,
                              citation_spans_enabled=citation_spans_enabled,
                              claim_evidence_candidates=claim_evidence_candidates,
                              span_fields_by_claim=span_fields_by_claim,
                              locating_quote=locating_quote, pool_size=pool_size,
                              deadline_s=claim_deadline_s)
        verdicts = out["verdicts"]
        ok_shards = [s for s in out["shards"] if s.get("status") == "ok"]
        models = sorted({s.get("model_executed") for s in out["shards"] if s.get("model_executed")})
        executed_model = models[0] if len(models) == 1 else (models or None)
        rec.update({
            "model": executed_model or _configured_name(model),
            "wall_s": round(time.monotonic() - started, 3),
            "attempts": sum(s.get("attempts", 0) for s in out["shards"]),
            "sharding": {"pool_size": out["pool_size"], "deadline_s": out["deadline_s"],
                         "rejected": out["rejected"],
                         "shards": [{k: s.get(k) for k in
                                     ("claim", "status", "reason", "attempts", "queue_wait_s",
                                      "wall_s", "model_executed", "prompt_sha256_8", "error")}
                                    for s in sorted(out["shards"], key=lambda s: s["claim"])]}})
        if ok_shards:
            rec["status"] = "ok"
        else:
            rec["status"] = "failed"
            rec["error"] = ("sharded verification: no shard returned a usable verdict "
                            f"({len(checkable)} claims, reasons "
                            + ",".join(sorted({s.get('reason') or 'unknown'
                                               for s in out['shards']})) + ")")
            if citation_spans_enabled:
                span_status = "model-output-invalid"
    elif checkable:
        # ONE RETRY on an empty answer, and only on an empty answer: a malformed answer is a
        # format problem a retry cannot fix. Both attempts are recorded.
        started = time.monotonic()
        attempts = 0
        out = None
        for attempt in range(2):
            attempts = attempt + 1
            # A heavier ask (spans) gets a longer allowance rather than a quieter failure.
            out, executed_model = _call(model, prompt, 300 if citation_spans_enabled else 180)
            if out:
                break
        rec.update({"model": executed_model or _configured_name(model),
                    "wall_s": round(time.monotonic() - started, 3), "attempts": attempts})
        obj = extract_json(out) if out else None
        if isinstance(obj, dict) and isinstance(obj.get("verdicts"), list):
            for v in obj["verdicts"]:
                if not isinstance(v, dict):
                    continue
                try:
                    n = int(v.get("claim"))
                except (TypeError, ValueError):
                    continue
                if 1 <= n <= len(checkable):
                    verdicts[checkable[n - 1]] = {
                        "verdict": str(v.get("verdict") or "").lower(),
                        "why": str(v.get("why") or "")[:200],
                        **({"citation_spans": v.get("citation_spans")} if citation_spans_enabled else {}),
                        # Kept on both paths (added in this library): see _shard_verdict_payload.
                        "relation": v.get("relation"),
                        **({"strength": v.get("strength"),
                            "evidence_spans": v.get("evidence_spans"),
                            "multi_span_required": v.get("multi_span_required")}
                           if claim_evidence_candidates is not None else {})}
            rec["status"] = "ok"
        else:
            rec["status"] = "failed"
            # A timeout (no output) and a malformed answer are different failures.
            rec["error"] = ("verification returned no model output after "
                            + str(rec.get("attempts") or 1) + " attempt(s) (timeout)"
                            if not out else "verification output did not parse")
            if citation_spans_enabled and span_status is None:
                span_status = "model-output-invalid"

    evidence_record = claim_evidence_sink(verdicts) if claim_evidence_sink is not None else None
    located = located_claims(report, brief, evidence_record) if gate_support else set()
    counts = {"supported": 0, "partial": 0, "unsupported": 0, "unresolved": 0, "unchecked": 0}
    for n, c in enumerate(report["claims"]):
        cites = c["citations"]
        any_resolved = any(x["resolution"] != "unresolved" for x in cites)
        judged = verdicts.get(n) or {}
        conflict = (judged.get("verdict") in ("supported", "partial")
                    and read_relation(judged.get("relation"))[0] == "contradicts")
        if not cites or not any_resolved:
            v = {"verdict": "unresolved",
                 "why": "no cited reference matched anything in the given sources"}
        elif n in verdicts and verdicts[n]["verdict"] in ("supported", "partial", "unsupported"):
            # Publish a COPY without the model's raw proposals: validated spans are attached per
            # citation by span_receipt; rejected proposals never reach the output.
            v = {"verdict": verdicts[n]["verdict"], "why": verdicts[n]["why"]}
            # The model's `why` names sources by prompt ordinal ("Source 2 says ..."), a numbering
            # the reader never sees. Translate ordinals to source names using the SAME resolved
            # list the prompt numbered from.
            resolved = [x for x in cites if x["resolution"] != "unresolved" and x.get("text")]

            def _name(m, resolved=resolved):
                j = int(m.group(1)) - 1
                if 0 <= j < len(resolved):
                    return labels.get(resolved[j].get("source"), resolved[j].get("source") or m.group(0))
                return m.group(0)
            v["why"] = re.sub(r"\bSource\s+(\d+)\b", _name, v.get("why") or "")
            if conflict:
                # Added in this library (not in the composer), revised on a measured finding: a
                # judge that names "contradicts" has said the source disagrees, so a supported or
                # partial verdict word with it is recorded UNSUPPORTED. In a preregistered agreement
                # test, 6 of 80 public refutation items came back partial + contradicts; treating
                # them as unchecked (the previous rule) lost real refutations. The override is
                # recorded, so a reader can see which word the judge actually used.
                v["verdict_override"] = {"rule": "relation-contradicts",
                                         "judge_verdict": v["verdict"]}
                v["verdict"] = "unsupported"
                v["why"] = (f"recorded unsupported: the judge named the relation contradicts "
                            f"(its verdict word was {v['verdict_override']['judge_verdict']}). "
                            + v["why"])[:400]
            elif gate_support and v["verdict"] == "supported" and n not in located:
                # Public support needs a located passage: the judge's word alone is a model
                # reading, and the sealed row for this claim already says it is not support.
                v["verdict_override"] = {"rule": SUPPORT_GATE_RULE, "judge_verdict": "supported"}
                v["verdict"] = "unresolved"
                v["why"] = (("recorded unresolved: the judge said supported, but "
                             + ("no passage it quoted was located in the digest-pinned source. "
                                if evidence_record is not None else
                                "the evidence pass was off, so no passage could be located. "))
                            + v["why"])[:400]
        else:
            v = {"verdict": "unchecked",
                 "why": "the verification step did not return a usable verdict for this claim"}
        # Every citation row is BUILT HERE from the resolver's own fields, whatever the model
        # returned: a model-supplied citations array can never reach the output.
        v["citations"] = [{k: x[k] for k in CITATION_ROW_FIELDS} for x in cites]
        # Per-citation bearing sentences (per-claim path only). Own-words digests by instruction;
        # surrounding quotes stripped and capped so a runaway answer cannot bloat the output.
        for row in (verdicts.get(n, {}).get("citation_bears") or []):
            if not isinstance(row, dict):
                continue
            bi = row.get("citation_index")
            text = row.get("how_it_bears")
            if (isinstance(bi, int) and not isinstance(bi, bool) and 0 <= bi < len(v["citations"])
                    and isinstance(text, str) and text.strip()):
                v["citations"][bi]["bears"] = text.strip().strip('"“”')[:240]
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
        target = target_for_claim(brief, c)
        if isinstance(target, dict):
            target["verification"] = v
            # Attach the resolved URL to the source entries themselves, so a reader can open and
            # check what a citation names.
            for src_entry, cite in zip(target.get("sources") or [], cites):
                if cite.get("url"):
                    src_entry["url"] = cite["url"]
                    src_entry["resolved_title"] = cite.get("matched_title")
    if citation_spans_enabled:
        if span_status is not None:
            rec["spans"] = {"status": span_status, "contract": SPAN_CONTRACT}
        else:
            spans = span_receipt(report, verdicts, brief, local_placement=local_placement)
            # What actually answered, never the configured name.
            spans["model_executed"] = executed_model
            rec["spans"] = spans
    rec["verdicts"] = counts
    receipts["verification"] = rec
    return rec


def overall_status(rec: dict) -> str:
    """One word for the whole brief, from the verification record.

    The record reports mechanics (ok = the model answered, failed = it did not, skipped =
    nothing was checkable) and per-claim counts. A reader needs the OUTCOME:
      unsupported     at least one claim the sources do not back
      unresolved      no claim could be tied to the sources at all
      verifier-error  the check itself broke
      partly-checked  per-claim mode: some claims checked, some terminally unchecked (never ok)
      ok              otherwise
    """
    raw = rec.get("status")
    if raw in ("off",):
        return raw
    if raw == "failed":
        return "verifier-error"
    counts = rec.get("verdicts") if isinstance(rec.get("verdicts"), dict) else {}
    total = sum(n for n in counts.values() if isinstance(n, int))
    if counts.get("unsupported", 0) > 0:
        return "unsupported"
    if total > 0 and counts.get("unresolved", 0) == total:
        return "unresolved"
    if (rec.get("sharding") and counts.get("unchecked", 0) > 0
            and (counts.get("supported", 0) + counts.get("partial", 0)
                 + counts.get("supported-completed", 0)) > 0):
        return "partly-checked"
    return "ok"


def judge(brief: dict, sources: dict, model, *, spans: bool = True, sharded: bool = True,
          local_placement: bool = True, locating_quote: bool = False, evidence: bool = True,
          completion: bool = False, completion_cap: int = 6,
          pool_size: int = VERIFY_POOL_DEFAULT,
          claim_deadline_s: float = VERIFY_CLAIM_DEADLINE_S,
          text_fields: tuple = TEXT_FIELDS) -> dict:
    """Check a brief against its sources: pass 1, pass 2 through `model`, spans, evidence, seal.

    The input brief and sources are not modified. Returns:
      {"status":         overall_status(...),
       "brief":          an annotated copy, in the shape of shape_brief: every claim gains "verification" with its verdict,
                         why, and citation rows (resolution, url, span with sentence_relation),
       "verification":   the run record (counts, model, timings, span receipt, shards),
       "claim_evidence": the sealed claim-evidence record (when evidence=True),
       "lineage":        {"sources": one digest row per source item},
       "attribution":    attribution-overclaim results per spanned sentence (a heuristic;
                         informational; not part of the seal)}

    Defaults are the configuration the composer runs in production: spans on, one claim per
    model call, local span placement on, the locating-quote wording off, the evidence pass on,
    completion off. completion=True is refused in this release (CompletionDisabled): the
    citation it appends would sit outside the sealed evidence record.
    """
    if completion:
        raise CompletionDisabled(COMPLETION_DISABLED)
    doc = shape_brief(copy.deepcopy(brief), in_place=True)
    captured = capture_sources(sources)
    receipts: dict = {}
    candidates = evidence_candidates(doc, captured) if evidence else None
    holder: dict = {}

    def sink(verdicts):
        holder["claim_evidence"] = build_claim_evidence(doc, captured, verdicts)
        return holder["claim_evidence"]

    rec = verify(doc, captured, receipts, model,
                 citation_spans_enabled=spans, sharded=sharded, local_placement=local_placement,
                 locating_quote=locating_quote, claim_evidence_candidates=candidates,
                 claim_evidence_sink=sink if evidence else None,
                 completion=completion, completion_cap=completion_cap,
                 pool_size=pool_size, claim_deadline_s=claim_deadline_s, text_fields=text_fields)
    result = {"status": overall_status(rec), "brief": doc, "verification": rec}
    if evidence:
        result["claim_evidence"] = holder.get("claim_evidence")
    result["lineage"] = {"sources": source_lineage(captured)}
    # Attribution overclaim (heuristic, informational, NOT sealed): see claim_seal.attribution.
    result["attribution"] = attribution_checks(
        doc, {sid: entry["label"] for sid, entry in load_sources(captured).items()})
    return result


__all__ = ["ModelText", "extract_json", "judge", "verify", "overall_status",
           "merge_shard_verdicts", "run_claim_shard", "SPAN_RELATIONS"]

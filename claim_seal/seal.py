"""The seal: an unkeyed SHA-256 digest chain over every outcome row of a claim-evidence record.

WHAT IT COVERS. Every outcome row, in order, whole: the claim it belongs to (by digest), the
citation, the support state and its reason, every evidence span (offsets, digests, any
extension), the relation verb, whether the judge chose that verb or it was defaulted
(relation_source), strength, corroboration, and the source binding. The chain starts from the
contract name and the row count, so appending, deleting or reordering rows breaks it too.

  chain_0 = sha256(canonical({"contract": ..., "outcome_count": N}))
  chain_k = sha256(chain_{k-1} as ascii + canonical({"outcome": row_k, "stage_a_binding": ...}))

canonical() is JSON with sorted keys, no whitespace, UTF-8, non-ASCII kept as is.

WHAT IT DOES NOT COVER. Only record["outcomes"] is chained. The record's own
verification.rule_ids and capture_time_result are not, and nothing outside the record is: not
the claim verdicts and their reasons, not the per-sentence spans, relations or goes-further
labels in the annotated brief, not the attribution-overclaim results, not the run record.
Editing those leaves verify_seal() True.

WHAT IT DOES NOT DO. It is unkeyed: anyone who can rewrite a row can also recompute and rewrite
the chain value beside it, and verify_seal() will then pass. It is therefore TAMPER-EVIDENT ONLY
ALONGSIDE VERSION HISTORY: commit the record (or publish its seal value somewhere you do not
control alone), and a later rewrite shows up as a change to a value that was already witnessed.
It is not a signature and not authentication; it says nothing about who wrote the record, and
nothing about whether the claims are true.
"""
from __future__ import annotations

import hashlib
import json

CLAIM_EVIDENCE_CONTRACT = "claim-evidence/v1"
CLAIM_EVIDENCE_SEAL_ALGORITHM = "sha256-digest-chain/v1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def chain_value(outcomes: list[dict]) -> str:
    """The chain's final value over `outcomes` (each a JSON-serializable dict)."""
    chain = _sha(canonical({"contract": CLAIM_EVIDENCE_CONTRACT, "outcome_count": len(outcomes)}))
    for outcome in outcomes:
        binding = {key: outcome.get(key) for key in ("response_id", "retrieved_at")}
        binding["response_sha256"] = [span.get("response_sha256")
                                      for span in outcome.get("evidence_spans") or []
                                      if isinstance(span, dict)]
        chain = _sha(chain.encode("ascii") + canonical({"outcome": outcome, "stage_a_binding": binding}))
    return chain


def seal(record: dict) -> dict:
    """Compute the chain over record["outcomes"] and write it to record["verification"]["seal"].

    Returns the same record. key_id is always None: there is no key.
    """
    if not isinstance(record, dict) or not isinstance(record.get("outcomes"), list):
        raise ValueError("seal() needs a record with an 'outcomes' list")
    record.setdefault("contract", CLAIM_EVIDENCE_CONTRACT)
    if record["contract"] != CLAIM_EVIDENCE_CONTRACT:
        raise ValueError(f"unknown contract {record['contract']!r}")
    verification = record.setdefault("verification", {})
    verification["seal"] = {"algorithm": CLAIM_EVIDENCE_SEAL_ALGORITHM, "key_id": None,
                            "value": chain_value(record["outcomes"])}
    return record


def seal_problem(record: object) -> str | None:
    """None when the seal recomputes; otherwise a short reason it does not."""
    if not isinstance(record, dict):
        return "not a JSON object"
    if record.get("contract") != CLAIM_EVIDENCE_CONTRACT:
        return f"contract is {record.get('contract')!r}, expected {CLAIM_EVIDENCE_CONTRACT!r}"
    outcomes = record.get("outcomes")
    if not isinstance(outcomes, list):
        return "no outcomes list"
    verification = record.get("verification")
    found = verification.get("seal") if isinstance(verification, dict) else None
    if not isinstance(found, dict) or not found.get("value"):
        return "no seal value"
    if found.get("algorithm") != CLAIM_EVIDENCE_SEAL_ALGORITHM:
        return f"algorithm is {found.get('algorithm')!r}, expected {CLAIM_EVIDENCE_SEAL_ALGORITHM!r}"
    if found.get("key_id") is not None:
        return "key_id is set, but this chain is unkeyed; a key claim is not something it can carry"
    if found.get("value") != chain_value(outcomes):
        return "the recorded chain does not recompute from the rows beside it"
    return None


def verify_seal(record: object) -> bool:
    """True when the unkeyed chain recomputes. Tamper evidence, not authentication."""
    return seal_problem(record) is None

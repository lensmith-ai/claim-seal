"""Shared fixtures. Every source text here is synthetic, written for these tests."""
from __future__ import annotations

import copy
import json
import os
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from claim_seal.support import ModelText  # noqa: E402

FETCHED_AT = "2025-01-15T09:30:00Z"

# A small corpus with the shapes per-claim checking has to handle: a connection claim, a claim
# with two citations, a fabricated citation, a repetitive claim, and one longer source.
CORPUS = {
    "items": [
        {"insight": "The relay handles ten thousand messages an hour after the queue rewrite.",
         "analysis": [],
         "sources": [{"id": "alpha", "ref": "Queue rewrite ships", "label": "Alpha Weekly"}]},
        {"insight": "Two labs report the same memory saving from the compaction pass.",
         "analysis": [],
         "sources": [{"id": "alpha", "ref": "Compaction pass measured", "label": "Alpha Weekly"},
                     {"id": "beta", "ref": "Independent replication note", "label": "Beta Journal"}]},
        {"insight": "A vendor promises a faster planner next spring.",
         "analysis": [],
         "sources": [{"id": "alpha", "ref": "Nothing with this title exists", "label": "Alpha Weekly"}]},
        {"insight": "The audit trail records the audit trail twice when the audit trail rotates.",
         "analysis": [],
         "sources": [{"id": "beta", "ref": "Rotation bug report", "label": "Beta Journal"}]},
    ],
    "connection": {
        "text": "Both the queue rewrite and the compaction pass trade peak speed for steadier tails.",
        "sources": [{"id": "alpha", "ref": "Queue rewrite ships", "label": "Alpha Weekly"},
                    {"id": "beta", "ref": "Independent replication note", "label": "Beta Journal"}],
    },
}

CORPUS_SOURCES = {
    "alpha": {"label": "Alpha Weekly", "items": [
        {"title": "Queue rewrite ships", "text": "The relay now clears ten thousand messages an hour, up from four thousand, after the queue was rewritten around a single append log. The team notes the p99 latency held steady while the mean fell. A source long enough that a whole-brief judge spends its budget here."},
        {"title": "Compaction pass measured", "text": "A compaction pass cut resident memory by a third on the reference workload."},
    ]},
    "beta": {"label": "Beta Journal", "items": [
        {"title": "Independent replication note", "text": "A second lab reproduced the one-third memory saving from the compaction pass on different hardware."},
        {"title": "Rotation bug report", "text": "When the log rotates, the audit trail writes its closing record twice."},
    ]},
}

FROZEN_BY_TEXT = {
    "The relay handles ten thousand messages an hour after the queue rewrite.":
        {"verdict": "supported", "why": "the cited text states the throughput figure"},
    "Two labs report the same memory saving from the compaction pass.":
        {"verdict": "supported", "why": "both cited texts report the memory saving"},
    "The audit trail records the audit trail twice when the audit trail rotates.":
        {"verdict": "partial", "why": "the cited text names one duplicate record, not two"},
    "Both the queue rewrite and the compaction pass trade peak speed for steadier tails.":
        {"verdict": "partial", "why": "the trade-off is implied, not stated"},
}
FROZEN_WHOLE_BRIEF_REPLY = {"verdicts": [
    {"claim": 1, "verdict": "supported", "why": "the cited text states the throughput figure"},
    {"claim": 2, "verdict": "supported", "why": "both cited texts report the memory saving"},
    {"claim": 3, "verdict": "partial", "why": "the cited text names one duplicate record, not two"},
    {"claim": 4, "verdict": "partial", "why": "the trade-off is implied, not stated"},
]}

# What the whole-brief path produces over CORPUS with the frozen reply (the composer's own
# frozen baseline, with this library's wording for an unresolved claim).
BASELINE = {
    "rec_status": "ok",
    "verdict_counts": {"supported": 2, "partial": 2, "unsupported": 0, "unresolved": 1, "unchecked": 0},
    "claims_checked": 4,
    "claims_total": 5,
    "item_verdicts": ["supported", "supported", "unresolved", "partial"],
    "connection_verdict": "partial",
}


def corpus():
    return copy.deepcopy(CORPUS), copy.deepcopy(CORPUS_SOURCES)


class Scripted:
    """A deterministic fake model: reply(prompt) -> str | raises. Records every prompt."""

    def __init__(self, reply, name="scripted-model", executed="executed-model"):
        self.reply, self.name, self.executed = reply, name, executed
        self.prompts = []
        self._lock = threading.Lock()

    def __call__(self, prompt, timeout=None):
        with self._lock:
            self.prompts.append(prompt)
        out = self.reply(prompt)
        if out is None or self.executed is None:
            return out
        return ModelText(out, model=self.executed)


def fixed(reply_obj, **kw):
    return Scripted(lambda prompt: json.dumps(reply_obj), **kw)


def per_claim_model(behavior=None, executed="shard-model"):
    """Fake for the per-claim path: answers a one-claim prompt from FROZEN_BY_TEXT.
    behavior(claim_text, call_no) may return a raw reply string, raise, or return None for the
    frozen answer. call_no counts calls per claim (1-based)."""
    calls: dict = {}
    lock = threading.Lock()

    def reply(prompt):
        text = next((t for t in FROZEN_BY_TEXT if t in prompt), None)
        with lock:
            calls[text] = calls.get(text, 0) + 1
            call_no = calls[text]
        if behavior is not None:
            out = behavior(text, call_no)
            if out is not None:
                return out
        entry = FROZEN_BY_TEXT.get(text)
        assert entry is not None, "prompt did not contain a known claim text"
        return json.dumps({"verdicts": [dict(entry, claim=1)]})

    return Scripted(reply, executed=executed), calls

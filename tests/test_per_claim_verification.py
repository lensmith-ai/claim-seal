"""Per-claim (sharded) verification: one model call per claim, so a slow claim costs only itself.

Ported from the composer's per-claim arms: 1 contract preservation, 2 fault isolation,
4 deadline terminality, 5 concurrency equivalence, 6 prompt equivalence and numbering,
7 backward compatibility; plus the merge, the partly-checked status, the bearing sentences and
the per-sentence relation. (The composer's checkpoint/recovery arms belong to its service and
are not part of this library.)
"""
from __future__ import annotations

import json
import time
import unittest

from helpers import (BASELINE, FROZEN_BY_TEXT, FROZEN_WHOLE_BRIEF_REPLY, Scripted, corpus, fixed,
                     per_claim_model)

from claim_seal.support import VERIFICATION_STATUSES
from claim_seal.support import merge_shard_verdicts, overall_status, verify
from claim_seal.prompts import build_claim_prompt, build_prompt
from claim_seal.resolve import resolve


def run_sharded(model, **kwargs):
    brief, sources = corpus()
    receipts: dict = {}
    kwargs.setdefault("gate_support", False)  # the raw model reading: the support gate has its own tests
    rec = verify(brief, sources, receipts, model, citation_spans_enabled=kwargs.pop("spans", False),
                 sharded=True, **kwargs)
    return brief, rec


def surface(brief, rec):
    return {
        "rec_status": rec["status"],
        "verdict_counts": rec["verdicts"],
        "claims_checked": rec["claims_checked"],
        "claims_total": rec["claims_total"],
        "item_verdicts": [it["verification"]["verdict"] for it in brief["items"]],
        "connection_verdict": brief["connection"]["verification"]["verdict"],
    }


class Arm1ContractPreservation(unittest.TestCase):
    def test_whole_brief_path_matches_the_frozen_baseline(self):
        brief, sources = corpus()
        rec = verify(brief, sources, {}, fixed(FROZEN_WHOLE_BRIEF_REPLY),
                     citation_spans_enabled=False, sharded=False, gate_support=False)
        self.assertEqual(surface(brief, rec), BASELINE)

    def test_per_claim_path_matches_the_whole_brief_path(self):
        model, _ = per_claim_model()
        brief, rec = run_sharded(model)
        self.assertEqual(surface(brief, rec), BASELINE)
        self.assertEqual(rec["sharding"]["rejected"], [])
        self.assertEqual(len(rec["sharding"]["shards"]), 4)
        whole, sources = corpus()
        verify(whole, sources, {}, fixed(FROZEN_WHOLE_BRIEF_REPLY), citation_spans_enabled=False,
               sharded=False, gate_support=False)
        self.assertEqual([i["verification"] for i in brief["items"]],
                         [i["verification"] for i in whole["items"]])


class Arm2FaultIsolation(unittest.TestCase):
    def test_one_slow_claim_costs_only_itself(self):
        victim = "Two labs report the same memory saving from the compaction pass."
        model, calls = per_claim_model(lambda text, n: "" if text == victim else None)
        brief, rec = run_sharded(model)
        self.assertEqual(brief["items"][1]["verification"]["verdict"], "unchecked")
        self.assertEqual(brief["items"][0]["verification"]["verdict"], "supported")
        self.assertEqual(brief["items"][3]["verification"]["verdict"], "partial")
        self.assertEqual(brief["connection"]["verification"]["verdict"], "partial")
        self.assertEqual(rec["status"], "ok")
        self.assertEqual(rec["verdicts"]["unchecked"], 1)
        bad = [s for s in rec["sharding"]["shards"] if s["status"] != "ok"]
        self.assertEqual(len(bad), 1)
        self.assertEqual(bad[0]["reason"], "timeout")
        self.assertEqual(bad[0]["attempts"], 2)  # an empty answer IS retried, then terminal
        self.assertEqual(calls[victim], 2)


class Arm4DeadlineTerminality(unittest.TestCase):
    VICTIM = "The relay handles ten thousand messages an hour after the queue rewrite."

    def _bad(self, rec):
        return [s for s in rec["sharding"]["shards"] if s["status"] != "ok"][0]

    def test_malformed_output_is_terminal_and_never_retried(self):
        model, calls = per_claim_model(lambda t, n: "not json {{" if t == self.VICTIM else None)
        _, rec = run_sharded(model)
        self.assertEqual(self._bad(rec)["reason"], "malformed")
        self.assertEqual(self._bad(rec)["attempts"], 1)
        self.assertEqual(calls[self.VICTIM], 1)

    def test_duplicate_claim_entries_are_malformed(self):
        two = json.dumps({"verdicts": [{"claim": 1, "verdict": "supported", "why": "a"},
                                       {"claim": 1, "verdict": "unsupported", "why": "b"}]})
        model, _ = per_claim_model(lambda t, n: two if t == self.VICTIM else None)
        _, rec = run_sharded(model)
        self.assertEqual(self._bad(rec)["reason"], "malformed")

    def test_rate_limit_fails_closed_immediately(self):
        def behavior(text, n):
            if text == self.VICTIM:
                raise RuntimeError("provider returned HTTP 429 Too Many Requests")
            return None
        model, calls = per_claim_model(behavior)
        _, rec = run_sharded(model)
        bad = self._bad(rec)
        self.assertEqual((bad["reason"], bad["attempts"], calls[self.VICTIM]), ("rate-limit", 1, 1))
        self.assertIn("429", bad["error"])

    def test_transport_error_is_retried_within_the_same_deadline(self):
        def behavior(text, n):
            if text == self.VICTIM:
                raise ConnectionError("connection reset")
            return None
        model, calls = per_claim_model(behavior)
        _, rec = run_sharded(model)
        bad = self._bad(rec)
        self.assertEqual((bad["reason"], bad["attempts"], calls[self.VICTIM]), ("transport", 2, 2))

    def test_queue_saturation_is_terminal_within_the_absolute_deadline(self):
        def behavior(text, n):
            time.sleep(0.35)
            return None
        model, _ = per_claim_model(behavior)
        started = time.monotonic()
        _, rec = run_sharded(model, pool_size=1, claim_deadline_s=0.3)
        wall = time.monotonic() - started
        self.assertEqual(len(rec["sharding"]["shards"]), 4)
        for shard in rec["sharding"]["shards"]:
            self.assertIn(shard["status"], ("ok", "unchecked"))
            if shard["status"] != "ok":
                self.assertIn(shard["reason"], ("timeout", "pool-exhausted"))
        self.assertGreaterEqual(
            len([s for s in rec["sharding"]["shards"] if s["reason"] == "pool-exhausted"]), 1)
        self.assertLess(wall, 5.0)

    def test_the_remaining_budget_is_offered_to_a_model_that_accepts_a_timeout(self):
        seen = []

        def model(prompt, timeout=None):
            seen.append(timeout)
            text = next(t for t in FROZEN_BY_TEXT if t in prompt)
            return json.dumps({"verdicts": [dict(FROZEN_BY_TEXT[text], claim=1)]})
        run_sharded(model, claim_deadline_s=50)
        self.assertTrue(seen and all(isinstance(t, float) and 0 < t <= 50 for t in seen), seen)


class Arm5ConcurrencyEquivalence(unittest.TestCase):
    def test_pool_sizes_are_semantically_equivalent(self):
        surfaces = []
        for pool in (1, 2, 4):
            model, _ = per_claim_model()
            brief, rec = run_sharded(model, pool_size=pool)
            self.assertEqual(rec["sharding"]["pool_size"], pool)
            surfaces.append(surface(brief, rec))
        self.assertEqual(surfaces[0], surfaces[1])
        self.assertEqual(surfaces[1], surfaces[2])

    def test_shard_receipts_carry_the_measurement_fields(self):
        model, _ = per_claim_model()
        _, rec = run_sharded(model)
        for shard in rec["sharding"]["shards"]:
            for field in ("claim", "status", "attempts", "queue_wait_s", "wall_s",
                          "model_executed", "prompt_sha256_8"):
                self.assertIn(field, shard)
            self.assertIsNotNone(shard["wall_s"])
            self.assertEqual(shard["model_executed"], "shard-model")


class Arm6PromptEquivalence(unittest.TestCase):
    def test_single_claim_prompt_is_byte_identical_to_whole_brief_single_claim(self):
        brief, sources = corpus()
        report = resolve(brief, sources)
        one = {"claims": [report["claims"][0]]}
        whole, checkable = build_prompt(one, citation_spans_enabled=True, sharded=True,
                                        span_fields_by_claim={0: {"insight": "x"}})
        self.assertEqual(checkable, [0])
        per = build_claim_prompt(report, 0, citation_spans_enabled=True,
                                 span_fields_by_claim={0: {"insight": "x"}})
        self.assertEqual(per, whole)

    def test_numbering_maps_back_when_unresolved_precedes_checkable(self):
        model, _ = per_claim_model()
        brief, _ = run_sharded(model)
        self.assertEqual(brief["items"][2]["verification"]["verdict"], "unresolved")
        self.assertEqual(brief["items"][3]["verification"]["verdict"], "partial")
        self.assertEqual(brief["items"][3]["verification"]["why"],
                         "the cited text names one duplicate record, not two")

    def test_two_texts_instruction_is_sent_only_per_claim(self):
        brief, sources = corpus()
        report = resolve(brief, sources)
        off, _ = build_prompt(report, citation_spans_enabled=True, sharded=False)
        on, _ = build_prompt(report, citation_spans_enabled=True, sharded=True)
        self.assertNotIn("DO NOT SWAP", off)
        self.assertIn("THE TWO QUOTE FIELDS COME FROM DIFFERENT TEXTS", on)
        self.assertIn("appear in the claim block above", on)


class Arm7BackwardCompatibility(unittest.TestCase):
    def test_all_shards_failing_is_a_failed_record(self):
        model, _ = per_claim_model(lambda text, n: "")
        brief, rec = run_sharded(model)
        self.assertEqual(rec["status"], "failed")
        self.assertIn("no shard returned a usable verdict", rec["error"])
        for item in (brief["items"][0], brief["items"][1], brief["items"][3]):
            self.assertEqual(item["verification"]["verdict"], "unchecked")

    def test_no_checkable_claims_stays_skipped(self):
        brief = {"items": [{"insight": "Nothing resolvable.", "sources": [
            {"id": "alpha", "ref": "does not exist"}]}]}
        model, _ = per_claim_model()
        rec = verify(brief, {"alpha": []}, {}, model, citation_spans_enabled=False, sharded=True)
        self.assertEqual(rec["status"], "skipped")
        self.assertEqual(brief["items"][0]["verification"]["verdict"], "unresolved")

    def test_empty_claim_list_is_valid(self):
        model, _ = per_claim_model()
        rec = verify({"items": []}, {}, {}, model, citation_spans_enabled=False, sharded=True)
        self.assertEqual(rec["status"], "skipped")
        self.assertEqual(rec["claims_total"], 0)


class MergePurity(unittest.TestCase):
    def test_duplicate_and_foreign_claim_ids_are_rejected(self):
        shards = [
            {"claim": 0, "status": "ok", "verdict": {"verdict": "supported", "why": "a"}},
            {"claim": 0, "status": "ok", "verdict": {"verdict": "unsupported", "why": "b"}},
            {"claim": 9, "status": "ok", "verdict": {"verdict": "supported", "why": "c"}},
            {"claim": 1, "status": "unchecked", "reason": "timeout", "verdict": None},
        ]
        verdicts, rejected = merge_shard_verdicts(shards, [0, 1])
        self.assertEqual(set(verdicts), {0})
        self.assertEqual(verdicts[0]["verdict"], "supported")
        self.assertEqual([(r["claim"], r["reason"]) for r in rejected],
                         [(0, "duplicate-claim-id"), (9, "claim-not-checkable")])


class PartlyCheckedClassification(unittest.TestCase):
    def test_per_claim_mixed_outcome_is_partly_checked_never_ok(self):
        rec = {"status": "ok", "sharding": {"pool_size": 2},
               "verdicts": {"supported": 2, "partial": 1, "unsupported": 0, "unresolved": 0, "unchecked": 1}}
        self.assertEqual(overall_status(rec), "partly-checked")

    def test_whole_brief_mixed_outcome_is_unchanged(self):
        rec = {"status": "ok",
               "verdicts": {"supported": 2, "partial": 1, "unsupported": 0, "unresolved": 0, "unchecked": 1}}
        self.assertEqual(overall_status(rec), "ok")

    def test_unsupported_outranks_partly_checked(self):
        rec = {"status": "ok", "sharding": {"pool_size": 2},
               "verdicts": {"supported": 1, "partial": 0, "unsupported": 1, "unresolved": 0, "unchecked": 1}}
        self.assertEqual(overall_status(rec), "unsupported")

    def test_partly_checked_is_a_named_status(self):
        self.assertIn("partly-checked", VERIFICATION_STATUSES)

    def test_all_shards_failed_classifies_verifier_error(self):
        rec = {"status": "failed", "sharding": {"pool_size": 2},
               "verdicts": {"supported": 0, "partial": 0, "unsupported": 0, "unresolved": 0, "unchecked": 4}}
        self.assertEqual(overall_status(rec), "verifier-error")

    def test_every_claim_unresolved_is_unresolved(self):
        rec = {"status": "skipped", "verdicts": {"supported": 0, "unresolved": 2}}
        self.assertEqual(overall_status(rec), "unresolved")


class BearsSentences(unittest.TestCase):
    CLAIM = "The relay handles ten thousand messages an hour after the queue rewrite."

    def _run(self, rows):
        reply = json.dumps({"verdicts": [dict(FROZEN_BY_TEXT[self.CLAIM], claim=1, citation_bears=rows)]})
        model, _ = per_claim_model(lambda t, n: reply if t == self.CLAIM else None)
        brief, _ = run_sharded(model)
        return brief["items"][0]["verification"]["citations"]

    def test_bears_sentence_lands_on_the_right_citation(self):
        cites = self._run([{"citation_index": 0,
                            "how_it_bears": "The article reports the throughput jump, which carries the claim directly."}])
        self.assertEqual(cites[0]["bears"],
                         "The article reports the throughput jump, which carries the claim directly.")

    def test_invalid_bears_rows_are_ignored_never_guessed(self):
        cites = self._run([{"citation_index": 9, "how_it_bears": "out of range"},
                           {"citation_index": True, "how_it_bears": "bool index"},
                           {"citation_index": 0, "how_it_bears": "   "}, "not-a-dict"])
        for cite in cites:
            self.assertNotIn("bears", cite)

    def test_bears_is_capped_and_dequoted(self):
        bears = self._run([{"citation_index": 0, "how_it_bears": '"' + ("x" * 500) + '"'}])[0]["bears"]
        self.assertEqual(len(bears), 240)
        self.assertNotIn('"', bears)

    def test_whole_brief_path_never_carries_bears(self):
        reply = dict(FROZEN_WHOLE_BRIEF_REPLY)
        reply["verdicts"] = [dict(v, citation_bears=[{"citation_index": 0, "how_it_bears": "x"}])
                             for v in reply["verdicts"]]
        brief, sources = corpus()
        verify(brief, sources, {}, fixed(reply), citation_spans_enabled=False, sharded=False)
        for it in brief["items"]:
            for cite in it["verification"]["citations"]:
                self.assertNotIn("bears", cite)

    def test_bears_instruction_is_sent_only_per_claim(self):
        brief, sources = corpus()
        report = resolve(brief, sources)
        off, _ = build_prompt(report, sharded=False)
        on, _ = build_prompt(report, sharded=True)
        self.assertNotIn("citation_bears", off)
        self.assertIn("citation_bears", on)
        self.assertIn("Own words only", on)


class SentenceRelations(unittest.TestCase):
    CLAIM = "The relay handles ten thousand messages an hour after the queue rewrite."

    def _run(self, relation):
        reply = json.dumps({"verdicts": [dict(FROZEN_BY_TEXT[self.CLAIM], claim=1, citation_spans=[
            {"citation_index": 0, "span_quote": "ten thousand messages an hour",
             "span_relation": relation}])]})
        model, _ = per_claim_model(lambda t, n: reply if t == self.CLAIM else None)
        brief, _ = run_sharded(model, spans=True)
        return brief["items"][0]["verification"]["citations"][0].get("span")

    def test_says_lands_on_the_validated_span(self):
        span = self._run("says")
        self.assertIsNotNone(span)
        self.assertEqual(span["sentence_relation"], "says")

    def test_every_vocabulary_verb_is_carried(self):
        for verb in ("implies", "suggests", "connotes", "follows", "explains", "posits",
                     "illustrates", "contradicts", "undetermined"):
            self.assertEqual(self._run(verb)["sentence_relation"], verb)

    def test_a_word_outside_the_vocabulary_is_read_as_undetermined_and_kept(self):
        span = self._run("hallucinates")
        self.assertIsNotNone(span)
        self.assertEqual((span["sentence_relation"], span["sentence_relation_raw"],
                          span["sentence_relation_read"]), ("undetermined", "hallucinates", "unknown"))

    def test_prompt_asks_only_per_claim(self):
        brief, sources = corpus()
        report = resolve(brief, sources)
        off, _ = build_prompt(report, citation_spans_enabled=True, sharded=False)
        on, _ = build_prompt(report, citation_spans_enabled=True, sharded=True)
        self.assertNotIn("span_relation", off)
        self.assertIn('"span_relation": "says"', on)
        self.assertIn("about the SENTENCE you quoted", on)


class WholeBriefRetry(unittest.TestCase):
    """One retry, and only on an empty answer: a malformed answer is a format problem a retry
    cannot fix, so it must not double the wait on a path that is already failing."""

    def run_replies(self, replies):
        queue = list(replies)
        model = Scripted(lambda prompt: queue.pop(0))
        brief, sources = corpus()
        rec = verify(brief, sources, {}, model, citation_spans_enabled=False, sharded=False)
        return rec, model

    def test_an_empty_answer_is_retried_once_and_the_second_answer_is_used(self):
        rec, model = self.run_replies(["", json.dumps(FROZEN_WHOLE_BRIEF_REPLY)])
        self.assertEqual((rec["status"], rec["attempts"], len(model.prompts)), ("ok", 2, 2))

    def test_two_empty_answers_fail_as_a_timeout(self):
        rec, model = self.run_replies(["", ""])
        self.assertEqual((rec["status"], rec["attempts"], len(model.prompts)), ("failed", 2, 2))
        self.assertIn("timeout", rec["error"])

    def test_a_malformed_answer_is_not_retried(self):
        rec, model = self.run_replies(["not json {{", json.dumps(FROZEN_WHOLE_BRIEF_REPLY)])
        self.assertEqual((rec["status"], rec["attempts"], len(model.prompts)), ("failed", 1, 1))
        self.assertEqual(rec["error"], "verification output did not parse")

    def test_a_good_first_answer_makes_one_call(self):
        rec, model = self.run_replies([json.dumps(FROZEN_WHOLE_BRIEF_REPLY)])
        self.assertEqual((rec["attempts"], len(model.prompts)), (1, 1))


if __name__ == "__main__":
    unittest.main()

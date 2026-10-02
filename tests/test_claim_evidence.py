"""The claim-evidence record: mechanical evidence vs judged relation, its states, and its seal.

Ported from the composer's evidence-record arms, plus the arms this library's brief names:
an edited, upgraded or appended row breaks the seal; a relabelled defaulted verb breaks it;
contradicts never comes out supported. Fixture texts are synthetic.
"""
from __future__ import annotations

import copy
import json
import re
import unittest
from pathlib import Path

import helpers  # noqa: F401  (puts the repo on sys.path)

from claim_seal import evidence as ev
from claim_seal.evidence import build_claim_evidence, capture_sources, evidence_candidates, evidence_window
from claim_seal.normalize import NORMALIZER_ID, normalize
from claim_seal.relations import derive_coarse_relation
from claim_seal.seal import CLAIM_EVIDENCE_SEAL_ALGORITHM, chain_value, seal, seal_problem, verify_seal

FETCHED_AT = "2025-01-15T09:30:00Z"
EVIDENCE_PATH = Path(ev.__file__)


def brief(claim="Synthetic claim.", refs=("Synthetic Title",)):
    return {"items": [{"insight": claim, "sources": [{"id": "feed", "ref": r, "label": r} for r in refs]}]}


def sources(body='{"title":"Synthetic Title","body":"Synthetic evidence."}', refs=("Synthetic Title",)):
    items = [{"id": f"feed-{n + 1}", "title": r, "text": "parsed text", "body": body,
              "url": f"https://example.test/{n + 1}", "retrieved_at": FETCHED_AT}
             for n, r in enumerate(refs)]
    return capture_sources({"feed": {"label": "Feed", "items": items}})


def model(verdict="supported", support_span="Synthetic evidence.", **extra):
    return {0: {"verdict": verdict, **extra,
                "evidence_spans": [{"citation_index": 0, "support_span": support_span}]}}


def build(doc=None, sc=None, verdicts=None):
    return build_claim_evidence(doc or brief(), sc if sc is not None else sources(),
                                verdicts if verdicts is not None else model())


def outcome(rec):
    return rec["outcomes"][0]["support_state"], rec["outcomes"][0]["state_reason"]


class ClaimEvidenceTests(unittest.TestCase):
    def test_exact_trace_is_identity_only_and_has_no_source_text(self):
        rec = build()
        o = rec["outcomes"][0]
        self.assertEqual(o["support_state"], "supported")
        self.assertEqual(o["evidence_spans"][0]["normalizer"], "norm/v1")
        self.assertNotIn("text", o["evidence_spans"][0])
        self.assertFalse(o["display_policy"]["text_retained"])
        self.assertEqual(o["response_id"], "feed-1")
        self.assertEqual(o["retrieved_at"], FETCHED_AT)
        encoded = json.dumps(rec, ensure_ascii=False)
        self.assertNotIn("Synthetic evidence.", encoded)
        self.assertNotIn('"body"', encoded)

    def test_model_only_supported_verdict_never_produces_supported(self):
        rec = build(verdicts={0: {"verdict": "supported"}})
        self.assertEqual(outcome(rec), ("unresolved", "EVIDENCE-SPAN-NOT-PROPOSED"))

    def test_absent_proposed_phrase_refuses(self):
        self.assertEqual(outcome(build(verdicts=model(support_span="Invented passage"))),
                         ("unresolved", "EVIDENCE-SPAN-NOT-LOCATED"))

    def test_mutated_source_byte_refuses_digest_binding(self):
        sc = sources()
        sc["feed"]["items"][0]["body"] += " drift"
        self.assertEqual(outcome(build(sc=sc)), ("unresolved", "EVIDENCE-RAW-DIGEST"))

    def test_digests_never_recorded_refuse(self):
        sc = sources()
        del sc["feed"]["items"][0]["raw_sha256"]
        self.assertEqual(outcome(build(sc=sc)), ("unresolved", "EVIDENCE-RAW-DIGEST"))

    def test_normalized_digest_drift_refuses_even_when_raw_digest_is_intact(self):
        sc = sources()
        sc["feed"]["items"][0]["normalized_sha256"] = "0" * 64
        self.assertEqual(outcome(build(sc=sc)), ("unresolved", "EVIDENCE-NORMALIZED-DIGEST"))

    def test_duplicate_phrase_never_selects_arbitrarily(self):
        rec = build(sc=sources(body='{"a":"Synthetic evidence.","b":"Synthetic evidence."}'))
        self.assertEqual(outcome(rec), ("unresolved", "EVIDENCE-SPAN-AMBIGUOUS"))

    def test_two_items_with_the_cited_title_refuse_rather_than_pick(self):
        sc = sources(refs=("Synthetic Title", "Synthetic Title"))
        self.assertEqual(outcome(build(sc=sc)), ("unresolved", "EVIDENCE-SPAN-AMBIGUOUS"))

    def test_failed_capture_is_unknown_not_supported(self):
        sc = sources()
        sc["feed"]["items"][0]["body"] = None
        self.assertEqual(outcome(build(sc=sc)), ("unknown", "EVIDENCE-FETCH-FAILED"))

    def test_uncited_source_is_unknown(self):
        self.assertEqual(outcome(build(sc={})), ("unknown", "EVIDENCE-NO-BODY"))
        sc = sources()
        sc["feed"]["items"] = []
        self.assertEqual(outcome(build(sc=sc))[0], "unknown")

    def test_undecodable_body_is_unknown_not_unresolved(self):
        sc = sources()
        raw = b"\xff\xfe\x00 not utf-8 \xff"
        sc["feed"]["items"][0].update({"body": raw, **{k: v for k, v in ev.digests_for(raw).items()
                                                      if k != "normalize_error"}})
        self.assertEqual(outcome(build(sc=sc)), ("unknown", "EVIDENCE-NORMALIZE-FAILED"))

    def test_supported_without_span_is_unresolved(self):
        rec = build(verdicts={0: {"verdict": "supported",
                                  "evidence_spans": [{"citation_index": 0, "support_span": None}]}})
        self.assertEqual(rec["outcomes"][0]["state_reason"], "EVIDENCE-SPAN-NOT-PROPOSED")

    def test_quote_from_the_parsed_text_not_the_captured_body_refuses(self):
        rec = build(sc=sources(body='{"description":"encoded-not-the-parsed-text"}'),
                    verdicts=model(support_span="parsed text"))
        self.assertEqual(rec["outcomes"][0]["state_reason"], "EVIDENCE-SPAN-NOT-LOCATED")

    def test_multi_span_required_accepts_each_located_support(self):
        rec = build(verdicts=model(multi_span_required=True))
        self.assertEqual(outcome(rec), ("supported", "EVIDENCE-SPANS-LOCATED"))

    def test_default_policy_never_leaks_span_text(self):
        sentinel = "CORPUS-SENTINEL-9f3a"
        rec = build(sc=sources(body='{"body":"' + sentinel + '"}'), verdicts=model(support_span=sentinel))
        self.assertNotIn(sentinel, json.dumps(rec))
        self.assertFalse(rec["outcomes"][0]["display_policy"]["text_displayable"])

    def test_states_unsupported_unresolved_unknown_do_not_collapse(self):
        unsupported = build(verdicts=model(verdict="unsupported"))
        unresolved = build(verdicts=model(support_span="missing"))
        sc = sources()
        sc["feed"]["items"][0]["body"] = None
        unknown = build(sc=sc)
        self.assertEqual([r["outcomes"][0]["support_state"] for r in (unsupported, unresolved, unknown)],
                         ["unsupported", "unresolved", "unknown"])

    def test_model_says_unsupported_is_recorded_as_unsupported(self):
        self.assertEqual(outcome(build(verdicts=model(verdict="unsupported"))),
                         ("unsupported", "EVIDENCE-MODEL-UNSUPPORTED"))

    def test_no_verifier_verdict_is_unknown_and_never_supported(self):
        self.assertEqual(outcome(build(verdicts=model(verdict=""))),
                         ("unknown", "EVIDENCE-VERIFIER-NOT-RUN"))

    def test_partial_is_unresolved_never_unsupported(self):
        self.assertEqual(outcome(build(verdicts=model(verdict="partial"))),
                         ("unresolved", "EVIDENCE-SPANS-LOCATED"))
        self.assertEqual(outcome(build(verdicts={0: {"verdict": "partial"}})),
                         ("unresolved", "EVIDENCE-MODEL-PARTIAL"))

    def test_every_rule_name_in_the_module_is_asserted_somewhere_in_this_file(self):
        """Fails the moment the evidence module gains a rule no test names."""
        prod = set(re.findall(r'"(EVIDENCE-[A-Z-]+)"', EVIDENCE_PATH.read_text(encoding="utf-8")))
        asserted = set(re.findall(r'"(EVIDENCE-[A-Z-]+)"', Path(__file__).read_text(encoding="utf-8")))
        self.assertEqual(sorted(prod - asserted), [])


class ContradictsNeverSupported(unittest.TestCase):
    """A judge that names "contradicts" with a supported or partial verdict word has said the
    source disagrees: the row is UNSUPPORTED under EVIDENCE-RELATION-CONTRADICTS and keeps the verb."""

    def test_supported_or_partial_with_contradicts_is_an_unsupported_row_that_keeps_the_verb(self):
        for verdict in ("supported", "partial"):
            rec = build(verdicts=model(verdict=verdict, relation="contradicts", strength=90))
            o = rec["outcomes"][0]
            self.assertEqual((o["support_state"], o["state_reason"], o["relation"], o["relation_coarse"],
                              o["relation_source"]),
                             ("unsupported", "EVIDENCE-RELATION-CONTRADICTS", "contradicts", "contradicted",
                              "judge"), verdict)
            self.assertEqual(o["corroboration"], 0)
            self.assertTrue(verify_seal(rec), verdict)
            tampered = copy.deepcopy(rec)
            tampered["outcomes"][0]["support_state"] = "supported"
            self.assertFalse(verify_seal(tampered), verdict)

    def test_padded_capitalised_and_alias_forms_are_read_the_same_way(self):
        for verdict in ("supported", "partial"):
            for word, source in (("contradicts ", "judge"), (" Contradicts\n", "judge"), ("CONTRADICTS", "judge"),
                                 ("\tcontradicts", "judge"), ("contradicted", "compat-default"),
                                 (" CONTRADICTED ", "compat-default")):
                o = build(verdicts=model(verdict=verdict, relation=word))["outcomes"][0]
                self.assertEqual((o["support_state"], o["state_reason"], o["relation"], o["relation_source"]),
                                 ("unsupported", "EVIDENCE-RELATION-CONTRADICTS", "contradicts", source),
                                 (verdict, repr(word)))

    def test_an_unsupported_verdict_with_contradicts_keeps_the_judges_own_reason(self):
        o = build(verdicts=model(verdict="unsupported", relation="contradicts"))["outcomes"][0]
        self.assertEqual((o["support_state"], o["state_reason"]), ("unsupported", "EVIDENCE-MODEL-UNSUPPORTED"))

    def test_contradicts_never_supported_across_every_verdict(self):
        for verdict in ("supported", "partial", "unsupported", "", "nonsense"):
            o = build(verdicts=model(verdict=verdict, relation="contradicts"))["outcomes"][0]
            self.assertNotEqual(o["support_state"], "supported", verdict)


class TwoAxisTests(unittest.TestCase):
    def test_present_evidence_carries_a_judged_inference_and_low_strength(self):
        o = build(verdicts=model(verdict="partial", relation="inferred", strength=25))["outcomes"][0]
        self.assertEqual((o["evidence"], o["relation"], o["relation_coarse"], o["strength"],
                          o["corroboration"]), ("present", "implies", "inferred", 25, 1))

    def test_stated_without_a_located_span_is_undetermined_not_unrelated(self):
        o = build(verdicts=model(relation="stated", support_span="Nowhere in the source"))["outcomes"][0]
        self.assertNotEqual(o["evidence"], "present")
        self.assertEqual((o["relation"], o["strength"], o["corroboration"]), ("undetermined", 0, 0))

    def test_model_evidence_assertion_does_not_set_mechanical_evidence(self):
        o = build(verdicts={0: {"verdict": "supported", "evidence": "present", "relation": "stated"}})
        self.assertEqual(o["outcomes"][0]["evidence"], "not-located")

    def test_unreadable_or_unnamed_verbs_are_undetermined(self):
        self.assertEqual(build(verdicts={0: {"verdict": "something-we-do-not-know"}})
                         ["outcomes"][0]["relation"], "undetermined")
        self.assertEqual(build(verdicts={0: {"verdict": "unsupported"}})["outcomes"][0]["relation"],
                         "undetermined")

    def test_two_spans_from_one_source_count_as_one_corroboration(self):
        first, second = "First premise.", "Second premise."
        rec = build(sc=sources(body=first + " " + second), verdicts={0: {
            "verdict": "partial", "relation": "explains",
            "evidence_spans": [{"citation_index": 0, "support_span": first},
                               {"citation_index": 0, "support_span": second}]}})
        o = rec["outcomes"][0]
        self.assertEqual((o["corroboration"], o["span_counts"]), (1, {"proposed": 2, "located": 2}))
        self.assertEqual({s["sha256"] for s in o["evidence_spans"]},
                         {ev._sha(first.encode()), ev._sha(second.encode())})
        self.assertNotIn(first, json.dumps(rec))

    def test_partial_location_records_both_counts_without_stronger_state(self):
        rec = build(sc=sources(body="Located premise."), verdicts={0: {
            "verdict": "partial", "relation": "implies",
            "evidence_spans": [{"citation_index": 0, "support_span": "Located premise."},
                               {"citation_index": 0, "support_span": "Not in the response."}]}})
        o = rec["outcomes"][0]
        self.assertEqual((o["evidence"], o["span_counts"], o["support_state"], o["state_reason"]),
                         ("present", {"proposed": 2, "located": 1}, "unresolved",
                          "EVIDENCE-SPANS-PARTIALLY-LOCATED"))

    def test_closed_verb_mapping_is_a_single_derivable_coarse_axis(self):
        expected = {"says": "stated", "contradicts": "contradicted", "undetermined": "undetermined",
                    "implies": "inferred", "suggests": "inferred", "connotes": "inferred",
                    "follows": "inferred", "explains": "inferred", "posits": "inferred",
                    "illustrates": "inferred"}
        self.assertEqual({v: derive_coarse_relation(v) for v in expected}, expected)


class RelationProvenanceTests(unittest.TestCase):
    def test_a_verb_the_judge_chose_is_marked_as_the_judges(self):
        o = build(verdicts=model(verdict="partial", relation="suggests"))["outcomes"][0]
        self.assertEqual((o["relation"], o["relation_source"]), ("suggests", "judge"))

    def test_a_defaulted_verb_is_marked_as_defaulted_and_distinguishable(self):
        chosen = build(verdicts={0: {"verdict": "partial", "relation": "implies"}})["outcomes"][0]
        defaulted = build(verdicts={0: {"verdict": "partial"}})["outcomes"][0]
        self.assertEqual(chosen["relation"], defaulted["relation"])
        self.assertEqual(defaulted["relation_source"], "compat-default")
        self.assertNotEqual(chosen["relation_source"], defaulted["relation_source"])


class SpanExtensionTests(unittest.TestCase):
    QUOTE = "shared phrase"
    BODY = ('{"body":"first unique left ' + QUOTE + ' alpha; second ordinary ' + QUOTE +
            ' beta; second ordinary ' + QUOTE + ' beta"}')

    def run_quote(self, body, quote):
        return build(sc=sources(body=body), verdicts={0: {
            "verdict": "supported", "relation": "says", "strength": 100,
            "evidence_spans": [{"citation_index": 0, "support_span": quote}]}})

    def test_extended_span_is_hash_verified_against_the_document(self):
        rec = self.run_quote(self.BODY, self.QUOTE)
        span = rec["outcomes"][0]["evidence_spans"][0]
        normalized = normalize(NORMALIZER_ID, self.BODY.encode())
        located = normalized[span["start"]:span["end"]]
        self.assertEqual(ev._sha(located.encode()), span["sha256"])
        self.assertEqual(span["response_sha256"], ev._sha(normalized.encode()))
        self.assertEqual(normalized.count(located), 1)
        self.assertTrue(span["extension"]["applied"])
        self.assertNotIn(self.QUOTE, json.dumps(rec))

    def test_extension_at_document_edges_stays_inside_the_document(self):
        for body in ("repeat-start repeat-middle repeat-middle",
                     ("x" * 2000 + " repeat-middle ") * 3 + "x" * 2000 + " repeat-end!"):
            span = self.run_quote(body, "repeat")["outcomes"][0]["evidence_spans"][0]
            normalized = normalize(NORMALIZER_ID, body.encode())
            self.assertTrue(0 <= span["start"] <= span["end"] <= len(normalized))
            self.assertEqual(normalized.count(normalized[span["start"]:span["end"]]), 1)

    def test_bound_refuses_when_unique_context_lies_beyond_the_limit(self):
        body = ("p" * 1536 + "repeat" + "x" * 1536 + "1 ") + ("p" * 1536 + "repeat" + "x" * 1536 + "2")
        self.assertEqual(outcome(self.run_quote(body, "repeat")), ("unresolved", "EVIDENCE-SPAN-AMBIGUOUS"))

    def test_seal_covers_every_extension_field(self):
        rec = self.run_quote(self.BODY, self.QUOTE)
        for field, value in (("applied", False), ("proposed_sha256", "0" * 64),
                             ("proposed_codepoints", 1), ("before_codepoints", 0), ("after_codepoints", 0)):
            tampered = copy.deepcopy(rec)
            tampered["outcomes"][0]["evidence_spans"][0]["extension"][field] = value
            self.assertFalse(verify_seal(tampered), field)


class SealTests(unittest.TestCase):
    def two_row_record(self):
        first, second = "First premise.", "Second premise."
        return build(doc=brief(refs=("Synthetic Title", "Second Title")),
                     sc=sources(body=first + " " + second, refs=("Synthetic Title", "Second Title")),
                     verdicts={0: {"verdict": "partial", "relation": "illustrates", "evidence_spans": [
                         {"citation_index": 0, "support_span": first},
                         {"citation_index": 1, "support_span": second}]}})

    def test_a_fresh_record_verifies(self):
        rec = self.two_row_record()
        self.assertTrue(verify_seal(rec))
        self.assertIsNone(seal_problem(rec))
        self.assertEqual(rec["verification"]["seal"]["algorithm"], CLAIM_EVIDENCE_SEAL_ALGORITHM)
        self.assertIsNone(rec["verification"]["seal"]["key_id"])

    def test_an_edited_row_breaks_the_seal(self):
        rec = self.two_row_record()
        for field, value in (("support_state", "supported"), ("state_reason", "EVIDENCE-MODEL-UNSUPPORTED"),
                             ("strength", 99), ("corroboration", 9), ("evidence", "no-text"),
                             ("source", "elsewhere"), ("citation_index", 7), ("retrieved_at", "later")):
            tampered = copy.deepcopy(rec)
            tampered["outcomes"][1][field] = value
            self.assertFalse(verify_seal(tampered), field)
        tampered = copy.deepcopy(rec)
        tampered["outcomes"][0]["evidence_spans"][0]["start"] += 1
        self.assertFalse(verify_seal(tampered), "span offset")
        tampered = copy.deepcopy(rec)
        tampered["outcomes"][0]["claim_ref"]["claim_sha256"] = "0" * 64
        self.assertFalse(verify_seal(tampered), "claim digest")

    def test_an_upgraded_verb_breaks_the_seal(self):
        rec = self.two_row_record()
        for verb in ("says", "implies"):
            tampered = copy.deepcopy(rec)
            tampered["outcomes"][0]["relation"] = verb
            self.assertFalse(verify_seal(tampered), verb)
        tampered = copy.deepcopy(rec)
        tampered["outcomes"][0]["relation_coarse"] = "stated"
        self.assertFalse(verify_seal(tampered))

    def test_an_appended_deleted_or_reordered_row_breaks_the_seal(self):
        rec = self.two_row_record()
        appended = copy.deepcopy(rec)
        appended["outcomes"].append(copy.deepcopy(appended["outcomes"][0]))
        deleted = copy.deepcopy(rec)
        deleted["outcomes"].pop()
        emptied = copy.deepcopy(rec)
        emptied["outcomes"] = []
        reordered = copy.deepcopy(rec)
        reordered["outcomes"].reverse()
        appended_span = copy.deepcopy(rec)
        appended_span["outcomes"][0]["evidence_spans"].append(
            copy.deepcopy(appended_span["outcomes"][0]["evidence_spans"][0]))
        for name, tampered in (("appended", appended), ("deleted", deleted), ("emptied", emptied),
                               ("reordered", reordered), ("appended span", appended_span)):
            self.assertFalse(verify_seal(tampered), name)

    def test_a_relabelled_defaulted_verb_breaks_the_seal(self):
        rec = build(verdicts={0: {"verdict": "partial", "evidence_spans": [
            {"citation_index": 0, "support_span": "Synthetic evidence."}]}})
        self.assertEqual(rec["outcomes"][0]["relation_source"], "compat-default")
        self.assertTrue(verify_seal(rec))
        relabelled = copy.deepcopy(rec)
        relabelled["outcomes"][0]["relation_source"] = "judge"
        self.assertFalse(verify_seal(relabelled))

    def test_the_seal_is_not_authentication(self):
        rec = build()
        claimed = copy.deepcopy(rec)
        claimed["verification"]["seal"].update({"algorithm": "hmac-sha256", "key_id": "invented"})
        self.assertFalse(verify_seal(claimed))
        keyed = copy.deepcopy(rec)
        keyed["verification"]["seal"]["key_id"] = "invented"
        self.assertFalse(verify_seal(keyed))

    def test_the_honest_limit_an_actor_who_rewrites_row_and_chain_passes(self):
        """Documented limit, pinned so nobody mistakes the chain for a signature: rewrite a row
        AND recompute the chain, and verify_seal passes. Only version history catches this."""
        rec = build()
        forged = copy.deepcopy(rec)
        forged["outcomes"][0]["strength"] = 1
        seal(forged)
        self.assertTrue(verify_seal(forged))
        self.assertNotEqual(forged["verification"]["seal"]["value"], rec["verification"]["seal"]["value"])

    def test_the_chain_matches_the_documented_construction(self):
        rec = self.two_row_record()
        self.assertEqual(rec["verification"]["seal"]["value"], chain_value(rec["outcomes"]))
        import hashlib
        canon = lambda v: json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        chain = hashlib.sha256(canon({"contract": "claim-evidence/v1",
                                      "outcome_count": len(rec["outcomes"])})).hexdigest()
        for row in rec["outcomes"]:
            binding = {"response_id": row["response_id"], "retrieved_at": row["retrieved_at"],
                       "response_sha256": [s["response_sha256"] for s in row["evidence_spans"]]}
            chain = hashlib.sha256(chain.encode("ascii") + canon({"outcome": row, "stage_a_binding": binding})).hexdigest()
        self.assertEqual(chain, rec["verification"]["seal"]["value"])


class JudgeWindowTests(unittest.TestCase):
    def test_window_is_bounded_and_is_a_slice_of_the_normalized_document(self):
        body = '{"title":"Synthetic Title","body":"' + "filler " * 200000 + 'Synthetic evidence."}'
        cands = evidence_candidates(brief(), sources(body=body))
        self.assertEqual(len(cands), 1)
        window = next(iter(cands.values()))
        self.assertLess(len(window), 10000)
        self.assertIn(window, normalize(NORMALIZER_ID, body.encode()))

    def test_total_stays_within_the_prompt_budget_as_citations_multiply(self):
        refs = tuple(f"Title {n}" for n in range(60))
        big = '{"title":"Title 0","body":"' + "filler " * 5000 + '"}'
        total = sum(len(v) for v in evidence_candidates(brief(refs=refs), sources(body=big, refs=refs)).values())
        self.assertLessEqual(total, 110000)

    def test_window_anchors_on_the_title_not_the_head_of_the_document(self):
        body = '{"lead":"' + "x" * 50000 + '","title":"Synthetic Title","body":"Synthetic evidence."}'
        window = next(iter(evidence_candidates(brief(), sources(body=body)).values()))
        self.assertIn("Synthetic Title", window)

    def test_body_heading_beats_the_head_title(self):
        head = "<html><head><meta x> <title>Big News</title> <link y><script z></script></head>"
        body = "<h1>Big News</h1> The first sentence states the claim plainly. More prose here."
        self.assertIn("first sentence states the claim", evidence_window(head + body, "Big News", 40, 80))

    def test_missing_title_falls_back_to_the_head(self):
        self.assertEqual(evidence_window("abcdef" * 50, None, 10, 30), ("abcdef" * 50)[:30])

    def test_disabled_pass_offers_no_windows(self):
        self.assertEqual(evidence_candidates(brief(), sources(), enabled=False), {})


if __name__ == "__main__":
    unittest.main()

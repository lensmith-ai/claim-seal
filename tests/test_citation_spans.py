"""Per-sentence citation spans: validation, the forgiving locator, local placement, the receipt.

Ported from the composer's span arms (validation, normalization, local placement, span fields,
supporting sentences in the prompt, the end-to-end receipt).
"""
from __future__ import annotations

import json
import unittest

import helpers  # noqa: F401
from helpers import fixed

from claim_seal.prompts import build_prompt
from claim_seal.resolve import resolve
from claim_seal.spans import find_forgiving, relocate_to_support, span_fields, span_receipt, validate_span
from claim_seal.support import verify


def brief(claim="Rocket launch proves Alpha works."):
    return {"items": [{"insight": claim, "sources": [
        {"id": "feed", "ref": "Alpha", "label": "Alpha"}, {"id": "feed", "ref": "Beta", "label": "Beta"}]}]}


def sources():
    return {"feed": [{"title": "Alpha", "text": "The Rocket launch proves Alpha works."},
                     {"title": "Beta", "text": "Independent confirmation."}]}


CITATION = {"resolution": "exact", "text": "source"}


class ValidateSpanTests(unittest.TestCase):
    def test_exact_unique_phrase_returns_canonical_code_point_span(self):
        insight = "A\U0001F600 Rocket launch proves Alpha works."
        span, reason = validate_span(insight, CITATION, {"citation_index": 0, "span_quote": "Rocket launch"})
        self.assertIsNone(reason)
        self.assertEqual(span, {"field": "insight", "start": len("A\U0001F600 "),
                                "end": len("A\U0001F600 Rocket launch"), "quote": "Rocket launch",
                                "coordinate_system": "unicode-code-point-v1",
                                "producer": "model-exact-quote-v1"})
        self.assertEqual(insight[span["start"]:span["end"]], span["quote"])

    def test_null_or_omitted_quote_is_not_proposed(self):
        for proposal in ({"citation_index": 0, "span_quote": None}, {"citation_index": 0}):
            self.assertEqual(validate_span("Alpha works.", CITATION, proposal), (None, "not_proposed"))

    def test_paraphrase_absent_and_repeated_quote_drop(self):
        for insight, quote, expected in (("Alpha works.", "Alpha really works.", "quote_not_found"),
                                         ("Alpha works.", "Beta works.", "quote_not_found"),
                                         ("Alpha works. Alpha works.", "Alpha works.", "quote_ambiguous")):
            self.assertEqual(validate_span(insight, CITATION, {"citation_index": 0, "span_quote": quote}),
                             (None, expected))

    def test_whitespace_variant_locates_via_forgiving_match(self):
        span, reason = validate_span("Alpha works.", CITATION, {"citation_index": 0, "span_quote": "Alpha  works."})
        self.assertIsNone(reason)
        self.assertEqual((span["quote"], span["producer"]), ("Alpha works.", "model-quote-normalized-v1"))

    def test_unresolved_completion_bad_index_bool_index_and_empty_are_unspanned(self):
        self.assertEqual(validate_span("Alpha works.", {"resolution": "unresolved"},
                                       {"citation_index": 0, "span_quote": "Alpha"})[1], "unresolved_citation")
        self.assertEqual(validate_span("Alpha works.", {"resolution": "exact", "completion": True},
                                       {"citation_index": 0, "span_quote": "Alpha"})[1],
                         "evidence_completion_citation")
        self.assertEqual(validate_span("Alpha works.", CITATION, {"citation_index": "0", "span_quote": "Alpha"})[1],
                         "invalid_citation_index")
        self.assertEqual(validate_span("Alpha works.", CITATION, {"citation_index": True, "span_quote": "Alpha"})[1],
                         "invalid_citation_index")
        self.assertEqual(validate_span("", CITATION, {"citation_index": 0, "span_quote": ""})[1], "empty_quote")

    def test_a_quote_in_a_detail_anchors_there_and_a_quote_in_two_fields_is_refused(self):
        span, _ = validate_span({"insight": "The claim itself.", "detail[0]": "A supporting sentence here."},
                                CITATION, {"citation_index": 0, "span_quote": "supporting sentence"})
        self.assertEqual(span["field"], "detail[0]")
        self.assertEqual(validate_span({"insight": "shared boundary", "detail[0]": "also shared boundary here"},
                                       CITATION, {"citation_index": 0, "span_quote": "shared boundary"}),
                         (None, "quote_ambiguous_across_fields"))

    def test_span_fields_carries_the_supporting_prose(self):
        doc = {"items": [{"insight": "T", "why_it_matters": "W", "detail": ["D0", "D1", ""]}]}
        self.assertEqual(sorted(span_fields(doc, {"kind": "item", "index": 0, "claim": "T"})),
                         ["detail[0]", "detail[1]", "insight", "why_it_matters"])


T = "A lab is previewing Bootstrappable Builds for teams."
T_SMART = "The post’s claim “AI Mode” is unproven."
T_WS = "agents   need\na ledger"


class ForgivingLocatorTests(unittest.TestCase):
    """Offsets are the provenance record: assert on the SLICE, and on what must NOT match."""

    def slice_of(self, text, quote):
        got = find_forgiving(text, quote)
        return None if got is None else text[got[0]:got[1]]

    def test_recovers_and_returns_original_text(self):
        for text, quote, expected in (
                (T, "bootstrappable builds", "Bootstrappable Builds"),
                (T, "  Bootstrappable Builds  ", "Bootstrappable Builds"),
                (T_SMART, 'post\'s claim "AI Mode"', "post’s claim “AI Mode”"),
                (T_WS, "agents need a ledger", "agents   need\na ledger"),
                (T_WS, "agents  need a  ledger", "agents   need\na ledger")):
            self.assertEqual(self.slice_of(text, quote), expected, quote)

    def test_refuses_rather_than_guesses(self):
        for text, quote in ((T, "more continuous"), (T, "six-line handoff and cost scorecard"),
                            ("Monitoring and monitoring", "MONITORING")):
            self.assertIsNone(find_forgiving(text, quote), quote)

    def test_exact_path_wins_and_the_fallback_records_how_it_located(self):
        span, _ = validate_span({"insight": "Monitoring and monitoring"}, CITATION,
                                {"citation_index": 0, "span_quote": "monitoring"})
        self.assertEqual((span["start"], span["producer"]), (15, "model-exact-quote-v1"))
        span, _ = validate_span({"insight": "A lab previewed Bootstrappable Builds today."}, CITATION,
                                {"citation_index": 0, "span_quote": "bootstrappable builds"})
        self.assertEqual((span["quote"], span["producer"]), ("Bootstrappable Builds", "model-quote-normalized-v1"))


SOURCE = ("The vendor report says agent work often ships with no owner and no name attached, "
          "and that hidden output then lands back on the humans as review burden.")
FIELDS = {
    "insight": "Unowned agent output quietly becomes human work.",
    "why_it_matters": ("Agent work with no owner and no name attached lands back on humans "
                       "as review burden. Naming things is cheap."),
    "detail[0]": "The calendar is a separate complaint entirely, about scheduling drift.",
}


def placement_receipt(local_placement, span_quote=FIELDS["insight"], source_text=SOURCE, propose=True):
    report = {"claims": [{"kind": "item", "index": 0, "claim": FIELDS["insight"],
                          "citations": [{"resolution": "exact", "text": source_text}]}]}
    doc = {"items": [{"insight": FIELDS["insight"], "why_it_matters": FIELDS["why_it_matters"],
                      "detail": [FIELDS["detail[0]"]],
                      "verification": {"verdict": "supported", "why": "", "citations": [{"resolution": "exact"}]}}]}
    proposals = [{"citation_index": 0, "span_quote": span_quote}] if propose else []
    rec = span_receipt(report, {0: {"verdict": "supported", "why": "", "citation_spans": proposals}}, doc,
                       local_placement=local_placement)
    return rec, doc["items"][0]["verification"]["citations"][0].get("span")


class LocalPlacementTests(unittest.TestCase):
    def test_relocates_to_the_overlapping_supporting_sentence(self):
        got = relocate_to_support(FIELDS, SOURCE)
        self.assertEqual((got["field"], got["producer"]), ("why_it_matters", "local-overlap-relocation-v1"))
        self.assertEqual(FIELDS[got["field"]][got["start"]:got["end"]], got["quote"])
        self.assertIn("review burden", got["quote"])
        self.assertNotIn("Naming things", got["quote"])

    def test_thin_overlap_tie_and_empty_source_are_refused(self):
        self.assertIsNone(relocate_to_support(FIELDS, "A completely unrelated essay about sourdough hydration ratios."))
        self.assertIsNone(relocate_to_support(
            {"insight": "thesis", "why_it_matters": "Agents ship unnamed ownerless output daily. "
                                                    "Agents ship unnamed ownerless output nightly."},
            "agents ship unnamed ownerless output"))
        self.assertIsNone(relocate_to_support(FIELDS, ""))
        self.assertIsNone(relocate_to_support(FIELDS, None))

    def test_on_relocates_a_thesis_span_and_keeps_the_audit_trail(self):
        rec, span = placement_receipt(True)
        self.assertEqual((rec["produced"], rec.get("relocated")), (1, 1))
        self.assertEqual((span["field"], span["relocated_from"]["field"]), ("why_it_matters", "insight"))

    def test_off_changes_nothing(self):
        rec, span = placement_receipt(False)
        self.assertNotIn("relocated", rec)
        self.assertEqual(span["field"], "insight")

    def test_a_span_already_in_the_paragraph_is_untouched(self):
        rec, span = placement_receipt(True, span_quote="lands back on humans")
        self.assertNotIn("relocated", rec)
        self.assertEqual(span["field"], "why_it_matters")
        self.assertNotIn("relocated_from", span)

    def test_no_model_span_creates_no_mark(self):
        rec, span = placement_receipt(True, propose=False)
        self.assertEqual((rec["produced"], rec["not_proposed"], span), (0, 1, None))

    def test_unlocatable_quote_gets_an_honestly_labelled_scorer_mark_only_when_on(self):
        quote = "a phrase from the source that is nowhere in our prose"
        rec, span = placement_receipt(True, span_quote=quote)
        self.assertEqual((rec["produced"], rec.get("scorer_placed"), rec["dropped"]), (1, 1, 0))
        self.assertEqual(span["producer"], "local-overlap-scorer-v1")
        self.assertTrue(span["unlocatable_quote"].startswith("a phrase from the source"))
        rec, span = placement_receipt(False, span_quote=quote)
        self.assertIsNone(span)
        self.assertEqual(rec["dropped_by_reason"].get("quote_not_found"), 1)

    def test_thin_overlap_still_refuses_with_placement_on(self):
        rec, span = placement_receipt(True, span_quote="unfindable words", source_text="zebra quark flotilla")
        self.assertIsNone(span)
        self.assertEqual(rec["dropped"], 1)
        self.assertNotIn("scorer_placed", rec)


class EndToEndSpanTests(unittest.TestCase):
    def run_verify(self, reply, *, spans=True, executed="actual-executed-model"):
        doc, receipts = brief(), {}
        verify(doc, sources(), receipts, fixed(reply, executed=executed), citation_spans_enabled=spans,
               sharded=False, local_placement=False)
        return doc, receipts

    def test_prompt_uses_original_indexes_and_the_citation_keeps_its_span_without_excerpt(self):
        doc, receipts = self.run_verify({"verdicts": [{"claim": 1, "verdict": "supported", "why": "Alpha says so.",
                                                       "citation_spans": [{"citation_index": 0, "span_quote": "Rocket launch"}]}]})
        cite = doc["items"][0]["verification"]["citations"][0]
        self.assertEqual(cite["span"]["quote"], "Rocket launch")
        self.assertNotIn("text", cite)
        spans = receipts["verification"]["spans"]
        self.assertEqual((spans["eligible"], spans["produced"], spans["dropped"], spans["not_proposed"]), (2, 1, 0, 1))
        self.assertEqual((spans["model_executed"], spans["contract"]), ("actual-executed-model", "citation-span/v1"))
        self.assertNotIn("citation_spans", doc["items"][0]["verification"])

    def test_duplicate_index_and_out_of_range_index_cannot_acquire_a_span(self):
        doc, receipts = self.run_verify({"verdicts": [{"claim": 1, "verdict": "supported", "why": "w",
                                                       "citation_spans": [{"citation_index": 0, "span_quote": "Rocket launch"},
                                                                          {"citation_index": 0, "span_quote": "Alpha works"},
                                                                          {"citation_index": 9, "span_quote": "Rocket launch"}]}]})
        self.assertTrue(all("span" not in c for c in doc["items"][0]["verification"]["citations"]))
        spans = receipts["verification"]["spans"]
        self.assertEqual(spans["dropped_by_reason"]["duplicate_citation_index"], 1)
        self.assertEqual(spans["invalid_index_proposals"], 1)
        self.assertEqual(sum(spans["dropped_by_reason"].values()), spans["dropped"])

    def test_malformed_output_has_a_noninvented_span_receipt_and_model_errors_propagate(self):
        doc, receipts = self.run_verify({"not_verdicts": []})
        self.assertEqual(receipts["verification"]["spans"]["status"], "model-output-invalid")
        self.assertNotIn("eligible", receipts["verification"]["spans"])

        def boom(prompt):
            raise TimeoutError("fixture timeout")
        with self.assertRaises(TimeoutError):
            verify(brief(), sources(), {}, boom, citation_spans_enabled=True, sharded=False)

    def test_model_executed_is_what_answered_or_none_never_the_configured_name(self):
        _, receipts = self.run_verify({"verdicts": []}, executed=None)
        self.assertIsNone(receipts["verification"]["spans"]["model_executed"])
        self.assertEqual(receipts["verification"]["model"], "scripted-model")

    def test_supporting_sentences_appear_in_the_prompt_only_with_spans_on(self):
        report = {"claims": [{"kind": "item", "index": 0, "analysis_index": None, "claim": "The thesis sentence.",
                              "citations": [{"source": "feed", "ref": "r", "resolution": "exact",
                                             "text": "some source text", "matched_title": "t"}]}]}
        on, _ = build_prompt(report, citation_spans_enabled=True,
                             span_fields_by_claim={0: {"insight": "The thesis sentence.", "detail[0]": "A supporting sentence."}})
        self.assertIn("SUPPORTING SENTENCES", on)
        self.assertIn("A supporting sentence.", on)
        alone, _ = build_prompt(report, citation_spans_enabled=True,
                                span_fields_by_claim={0: {"insight": "The thesis sentence."}})
        self.assertNotIn("SUPPORTING SENTENCES", alone)
        off, _ = build_prompt(report, citation_spans_enabled=False,
                              span_fields_by_claim={0: {"detail[0]": "A supporting sentence."}})
        self.assertNotIn("A supporting sentence.", off)

    def test_prompt_enumerates_original_citation_indexes_and_asks_for_relation_with_evidence(self):
        report = resolve(brief(), sources())
        prompt, _ = build_prompt(report, citation_spans_enabled=True)
        self.assertIn("citation_index 0", prompt)
        self.assertIn("citation_index 1", prompt)
        prompt, _ = build_prompt(report, claim_evidence_candidates={(0, 0): "The Rocket launch proves Alpha works."})
        self.assertIn('"relation": "says"|"implies"|"suggests"|"connotes"|"follows"|"explains"|"posits"|"illustrates"|"contradicts"|"undetermined"', prompt)
        self.assertIn('"strength": <integer 0-100>', prompt)
        self.assertIn("RESPONSE TEXT (quote support_span verbatim from HERE", prompt)

    def test_the_sink_receives_relation_and_strength(self):
        captured = {}
        reply = {"verdicts": [{"claim": 1, "verdict": "partial", "why": "A measured inference.",
                               "relation": "suggests", "strength": 25}]}
        verify(brief(), sources(), {}, fixed(reply), citation_spans_enabled=False, sharded=False,
               claim_evidence_candidates={}, claim_evidence_sink=captured.update)
        self.assertEqual((captured[0]["relation"], captured[0]["strength"]), ("suggests", 25))


if __name__ == "__main__":
    unittest.main()

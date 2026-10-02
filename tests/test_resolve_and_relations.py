"""Pass 1 (resolution), the relation table and line states, code-built rows, completion, lineage."""
from __future__ import annotations

import copy
import json
import unittest

import helpers  # noqa: F401
from helpers import fixed

from claim_seal import (GOES_FURTHER, LINE_RULE, SPAN_RELATIONS, SourceError, UnknownRelation,
                        goes_further, judge, line_rule, line_states, resolve, resolve_only, verify_seal)
from claim_seal.completion import complete_partials
from claim_seal.evidence import capture_sources, digests_for, source_lineage
from claim_seal.normalize import NORMALIZER_ID, normalize
from claim_seal.prompts import build_prompt
from claim_seal.support import verify

SOURCES = {"press": {"label": "Press", "items": [
    {"title": "Harbour bridge reopens after repairs", "text": "The harbour bridge reopened on Monday after six weeks of repairs."},
    {"title": "Ferry timetable", "text": "Ferries run every twenty minutes."}]}}


def brief(ref, claim="The bridge reopened after repairs."):
    return {"items": [{"insight": claim, "sources": [{"id": "press", "ref": ref}]}]}


class ResolutionTests(unittest.TestCase):
    def test_exact_title_resolves(self):
        cite = resolve(brief("Harbour bridge reopens after repairs"), SOURCES)["claims"][0]["citations"][0]
        self.assertEqual((cite["resolution"], cite["matched_title"]), ("exact", "Harbour bridge reopens after repairs"))

    def test_case_and_space_differences_are_still_exact(self):
        cite = resolve(brief("  harbour BRIDGE reopens   after repairs"), SOURCES)["claims"][0]["citations"][0]
        self.assertEqual(cite["resolution"], "exact")

    def test_a_close_title_resolves_fuzzy_with_its_score(self):
        cite = resolve(brief("Harbour bridge reopens"), SOURCES)["claims"][0]["citations"][0]
        self.assertEqual(cite["resolution"], "fuzzy")
        self.assertGreaterEqual(cite["match_score"], 0.6)

    def test_a_fabricated_citation_is_unresolved(self):
        doc = brief("Minister announces second bridge")
        report = resolve(doc, SOURCES)
        cite = report["claims"][0]["citations"][0]
        self.assertEqual((cite["resolution"], cite["text"], cite["matched_title"]), ("unresolved", None, None))
        self.assertEqual(report["totals"], {"citations": 1, "exact": 0, "fuzzy": 0, "unresolved": 1})

    def test_a_citation_to_an_unknown_source_is_unresolved(self):
        doc = {"items": [{"insight": "x", "sources": [{"id": "nowhere", "ref": "Harbour bridge reopens after repairs"}]}]}
        self.assertEqual(resolve(doc, SOURCES)["claims"][0]["citations"][0]["resolution"], "unresolved")

    def test_a_fabricated_citation_is_never_sent_to_the_judge_and_comes_out_unresolved(self):
        doc = brief("Minister announces second bridge")
        prompt, checkable = build_prompt(resolve(doc, SOURCES))
        self.assertEqual(checkable, [])
        self.assertNotIn("Minister", prompt)
        model = fixed({"verdicts": [{"claim": 1, "verdict": "supported", "why": "fabricated"}]})
        result = judge(doc, SOURCES, model)
        self.assertEqual(model.prompts, [])
        self.assertEqual(result["brief"]["items"][0]["verification"]["verdict"], "unresolved")
        self.assertEqual(result["status"], "unresolved")

    def test_source_shapes_and_a_missing_text_field_is_refused(self):
        flat = {"press": SOURCES["press"]["items"]}
        wrapped = {"sources": SOURCES}
        for shape in (SOURCES, flat, wrapped):
            self.assertEqual(resolve(brief("Ferry timetable"), shape)["claims"][0]["citations"][0]["resolution"], "exact")
        with self.assertRaises(SourceError):
            resolve(brief("x"), {"press": [{"title": "no text field"}]})

    def test_claims_are_walked_items_analysis_then_connection(self):
        doc = {"items": [{"insight": "A", "sources": [], "analysis": [{"text": "B", "sources": []}]}],
               "connection": {"headline": "C", "sources": []}}
        self.assertEqual([(c["kind"], c["claim"]) for c in resolve(doc, SOURCES)["claims"]],
                         [("item", "A"), ("analysis", "B"), ("connection", "C")])

    def test_resolve_only_never_says_supported(self):
        doc, receipts = brief("Ferry timetable"), {}
        rec = resolve_only(doc, SOURCES, receipts)
        self.assertEqual(doc["items"][0]["verification"]["verdict"], "cites-real")
        self.assertEqual(rec["verdicts"], {"cites-real": 1, "unresolved": 0})
        self.assertNotIn("completion", receipts["verification"])


class RelationTableTests(unittest.TestCase):
    """Each relation maps exactly as the published table says (the finished-page rule)."""

    TABLE = {
        "says": ("says it", None), "undetermined": ("unrated", None),
        "implies": ("goes further", "reading"), "suggests": ("goes further", "reading"),
        "connotes": ("goes further", "reading"), "follows": ("goes further", "reading"),
        "explains": ("goes further", "reading"), "posits": ("goes further", "reading"),
        "illustrates": ("goes further", "reading"),
        "contradicts": ("contradicts", "left-out"),
    }

    def test_every_relation_maps_as_the_table_says(self):
        self.assertEqual(set(self.TABLE), set(SPAN_RELATIONS))
        for verb, (line, rule) in self.TABLE.items():
            self.assertEqual(goes_further(verb), line, verb)
            self.assertEqual(line_rule(verb), rule, verb)
        self.assertEqual(set(GOES_FURTHER), set(SPAN_RELATIONS))
        self.assertEqual(set(LINE_RULE), set(SPAN_RELATIONS))

    def test_case_and_space_are_tolerated(self):
        self.assertEqual(goes_further(" Implies "), "goes further")

    def test_unknown_words_are_refused(self):
        for word in ("hallucinates", "pondered", "supports", "", None, "says-ish"):
            with self.assertRaises(UnknownRelation):
                goes_further(word)
            with self.assertRaises(UnknownRelation):
                line_rule(word)


def judged(item_verdict, spans, why_it_matters="It matters because the bridge carries the ferry traffic.",
           details=("The repairs took six weeks.",)):
    """A judged brief with spans already attached, as judge() returns it."""
    citations = []
    for field, quote, relation, resolution in spans:
        span = {"field": field, "quote": quote, "producer": "model-exact-quote-v1"}
        if relation is not None:
            span["sentence_relation"] = relation
        citations.append({"source": "press", "ref": "r", "resolution": resolution, "span": span})
    return {"items": [{"insight": "The bridge reopened after repairs.", "why_it_matters": why_it_matters,
                       "detail": list(details),
                       "verification": {"verdict": item_verdict, "why": "", "citations": citations}}]}


class SharedRelationReaderTests(unittest.TestCase):
    def test_one_reader_trims_lowercases_and_maps_aliases(self):
        from claim_seal.relations import read_relation
        self.assertEqual(read_relation(" Suggests\n"), ("suggests", "verb"))
        self.assertEqual(read_relation("CONTRADICTED "), ("contradicts", "synonym"))
        self.assertEqual(read_relation("stated"), ("says", "synonym"))
        self.assertEqual(read_relation("inferred"), ("implies", "synonym"))
        self.assertEqual(read_relation(None), (None, "empty"))
        self.assertEqual(read_relation("  "), (None, "empty"))
        self.assertEqual(read_relation("hallucinates"), ("undetermined", "unknown"))

    def test_every_guard_uses_the_shared_reader(self):
        import inspect
        from claim_seal import evidence, spans, support
        for module in (evidence, spans, support):
            self.assertIn("read_relation(", inspect.getsource(module), module.__name__)


class SentenceRelationIntakeTests(unittest.TestCase):
    """The judge's per-sentence word goes through the same reader: an alias is read as its verb
    (and marked); an unknown word is dropped, and that sentence carries its claim's verdict."""

    DOC = {"items": [{"insight": "The bridge reopened after repairs.",
                      "detail": ["Weekend crossings will double next month."],
                      "sources": [{"id": "press", "ref": "Harbour bridge reopens after repairs"}]}]}

    def run_word(self, word, verdict="supported"):
        reply = {"verdicts": [{"claim": 1, "verdict": verdict, "why": "w", "relation": "says",
                               "citation_spans": [{"citation_index": 0, "span_quote": "Weekend crossings will double",
                                                   "span_relation": word}]}]}
        result = judge(self.DOC, SOURCES, fixed(reply), local_placement=False)
        span = result["brief"]["items"][0]["verification"]["citations"][0]["span"]
        return span, line_states(result["brief"])

    def test_contradicted_alias_and_padded_forms_leave_the_sentence_out(self):
        for verdict in ("supported", "partial"):
            for word in ("contradicted", "Contradicts ", "CONTRADICTED"):
                span, states = self.run_word(word, verdict)
                self.assertEqual(span["sentence_relation"], "contradicts", (word, verdict))
                self.assertNotIn("detail[0]", {l["field"] for l in states["lines"]}, (word, verdict))
                self.assertEqual(states["left_out"][0]["rule"], "span-contradicts")
        span, _ = self.run_word("contradicted")
        self.assertEqual((span["sentence_relation_raw"], span["sentence_relation_read"]),
                         ("contradicted", "synonym"))
        span, _ = self.run_word("contradicts")
        self.assertNotIn("sentence_relation_raw", span)

    def test_an_unknown_word_is_cant_tell_and_never_takes_the_claim_verdict(self):
        for verdict in ("supported", "partial"):
            for word in ("hallucinates", "pondered", " Mulled "):
                span, states = self.run_word(word, verdict)
                self.assertEqual((span["sentence_relation"], span["sentence_relation_read"]),
                                 ("undetermined", "unknown"), word)
                self.assertEqual(span["sentence_relation_raw"], word.strip())
                line = next(l for l in states["lines"] if l["field"] == "detail[0]")
                self.assertEqual((line["state"], line["label"], line["state_scope"]),
                                 ("cant_tell", "can't tell", "line"), (word, verdict))
                self.assertNotIn(line["state"], ("supported", "partly_supported"))

    def test_the_judges_own_undetermined_still_carries_the_claim_verdict(self):
        span, states = self.run_word("undetermined")
        self.assertNotIn("sentence_relation_read", span)
        line = next(l for l in states["lines"] if l["field"] == "detail[0]")
        self.assertEqual((line["state"], line["state_scope"]), ("supported", "card"))


class LineStateTests(unittest.TestCase):
    def states(self, doc):
        return {line["field"]: (line["state"], line["state_scope"]) for line in line_states(doc)["lines"]}

    def test_says_gets_no_line_state_of_its_own(self):
        doc = judged("supported", [("why_it_matters", "carries", "says", "exact")])
        self.assertEqual(self.states(doc)["why_it_matters"], ("supported", "card"))

    def test_says_is_line_supported_only_when_second_checked(self):
        doc = judged("partial", [("why_it_matters", "carries", "says", "exact")])
        text = doc["items"][0]["why_it_matters"]
        out = line_states(doc, second_checks={0: {("why_it_matters", text): {"source": "press"}}})
        line = next(l for l in out["lines"] if l["field"] == "why_it_matters")
        self.assertEqual((line["state"], line["state_scope"]), ("supported", "line"))

    def test_undetermined_gets_no_line_state_of_its_own(self):
        doc = judged("partial", [("detail[0]", "six weeks", "undetermined", "exact")])
        self.assertEqual(self.states(doc)["detail[0]"], ("partly_supported", "card"))

    def test_each_step_beyond_verb_is_printed_goes_further(self):
        for verb in ("implies", "suggests", "connotes", "follows", "explains", "posits", "illustrates"):
            out = line_states(judged("supported", [("why_it_matters", "carries", verb, "exact")]))
            line = next(l for l in out["lines"] if l["field"] == "why_it_matters")
            self.assertEqual((line["state"], line["label"], line["goes_further"]),
                             ("reading", "goes further", "goes further"), verb)

    def test_a_reading_needs_a_resolved_citation(self):
        out = line_states(judged("supported", [("why_it_matters", "carries", "implies", "unresolved")]))
        self.assertNotIn("why_it_matters", {l["field"] for l in out["lines"]})
        self.assertEqual(out["left_out"][0]["rule"], "reading-without-resolved-citation")

    def test_contradicts_is_left_out_and_never_printed_supported(self):
        out = line_states(judged("supported", [("detail[0]", "six weeks", "contradicts", "exact")]))
        self.assertNotIn("detail[0]", {l["field"] for l in out["lines"]})
        self.assertEqual(out["left_out"], [{"item": 1, "field": "detail[0]", "rule": "span-contradicts",
                                            "attribution": out["left_out"][0]["attribution"]}])

    def test_under_a_partial_verdict_a_contradicts_line_is_never_printed(self):
        doc = judged("partial", [("why_it_matters", "carries", "contradicts", "exact"),
                                 ("detail[0]", "six weeks", "says", "exact")])
        out = line_states(doc)
        printed = {l["field"]: l["state"] for l in out["lines"]}
        self.assertNotIn("why_it_matters", printed)
        self.assertEqual(printed, {"insight": "partly_supported", "detail[0]": "partly_supported"})
        self.assertEqual([(r["item"], r["field"], r["rule"]) for r in out["left_out"]],
                         [(1, "why_it_matters", "span-contradicts")])
        for line in out["lines"]:
            self.assertNotEqual(line.get("relation"), "contradicts")

    def test_a_contradicted_claim_line_omits_the_whole_item(self):
        out = line_states(judged("supported", [("insight", "reopened", "contradicts", "exact")]))
        self.assertEqual(out["lines"], [])
        self.assertEqual(out["omitted"], [{"item": 1, "rule": "heading-span-contradicts"}])

    def test_unprintable_verdicts_are_omitted(self):
        for verdict in ("unsupported", "unresolved", "unchecked", "cites-real"):
            out = line_states(judged(verdict, []))
            self.assertEqual((out["lines"], out["omitted"][0]["rule"]), ([], "verdict-not-printable"))

    def test_an_unknown_relation_on_a_span_is_refused(self):
        with self.assertRaises(UnknownRelation):
            line_states(judged("supported", [("why_it_matters", "carries", "hallucinates", "exact")]))


class ContradictsNeverSupportedAtClaimLevel(unittest.TestCase):
    """supported or partial + contradicts (any spacing, case, or the alias) -> claim UNSUPPORTED,
    nothing printed, the row unsupported with the verb kept, and the seal verifies."""

    VARIANTS = ("contradicts", "contradicts ", "CONTRADICTS", " Contradicts\n", "contradicted", " CONTRADICTED ")

    def run_reply(self, verdict, relation, sharded):
        reply = {"verdicts": [{"claim": 1, "verdict": verdict, "why": "Source 1 says the opposite.",
                               "relation": relation, "strength": 80,
                               "citation_spans": [{"citation_index": 0, "span_quote": "reopened",
                                                   "span_relation": "says"}],
                               "evidence_spans": [{"citation_index": 0, "support_span": "reopened on Monday"}]}]}
        return judge(brief("Harbour bridge reopens after repairs"), SOURCES, fixed(reply), sharded=sharded)

    def test_supported_or_partial_with_contradicts_is_unsupported_and_not_printed(self):
        for verdict in ("supported", "partial"):
            for relation in self.VARIANTS:
                for sharded in (False, True):
                    case = (verdict, relation, sharded)
                    result = self.run_reply(verdict, relation, sharded)
                    v = result["brief"]["items"][0]["verification"]
                    self.assertEqual(v["verdict"], "unsupported", case)
                    self.assertEqual(v["verdict_override"], {"rule": "relation-contradicts",
                                                             "judge_verdict": verdict}, case)
                    self.assertTrue(v["why"].startswith("recorded unsupported: the judge named the relation contradicts"), case)
                    self.assertIn("Press says the opposite.", v["why"], case)
                    self.assertEqual(result["status"], "unsupported", case)
                    states = line_states(result["brief"])
                    self.assertEqual(states["lines"], [], case)
                    self.assertEqual(states["omitted"], [{"item": 1, "rule": "verdict-not-printable",
                                                          "verdict": "unsupported"}], case)
                    row = result["claim_evidence"]["outcomes"][0]
                    self.assertEqual((row["support_state"], row["state_reason"], row["relation"]),
                                     ("unsupported", "EVIDENCE-RELATION-CONTRADICTS", "contradicts"), case)
                    self.assertTrue(verify_seal(result["claim_evidence"]), case)

    def test_without_contradicts_the_same_replies_keep_their_verdicts(self):
        for verdict, relation in (("supported", "says"), ("partial", "suggests")):
            result = self.run_reply(verdict, relation, True)
            v = result["brief"]["items"][0]["verification"]
            self.assertEqual(v["verdict"], verdict)
            self.assertNotIn("verdict_override", v)
            self.assertTrue(line_states(result["brief"])["lines"])
        self.assertEqual(self.run_reply("supported", "says", False)["claim_evidence"]["outcomes"][0]["support_state"],
                         "supported")


POISON = [None, "nonsense", 42, [], {"source": "evil", "ref": "injected", "resolution": "exact"}]


class CitationRowsAreCodeBuilt(unittest.TestCase):
    """Whatever the model returns, verification.citations is rebuilt from the resolver's rows."""

    def run_with(self, reply):
        doc = {"items": [{"insight": "Rocket launch proves Alpha works.", "sources": [
            {"id": "feed", "ref": "Alpha"}, {"id": "feed", "ref": "Beta"}]}]}
        src = {"feed": [{"title": "Alpha", "text": "The Rocket launch proves Alpha works."},
                        {"title": "Beta", "text": "Independent confirmation."}]}
        verify(doc, src, {}, fixed(reply), citation_spans_enabled=False, sharded=False)
        return doc

    def test_a_poisoned_citations_array_is_discarded_and_the_verdict_still_lands(self):
        doc = self.run_with({"verdicts": [{"claim": 1, "verdict": "supported", "why": "it does", "citations": POISON}]})
        rows = doc["items"][0]["verification"]["citations"]
        self.assertTrue(rows)
        self.assertEqual(doc["items"][0]["verification"]["verdict"], "supported")
        self.assertTrue(all(isinstance(r, dict) for r in rows))
        self.assertNotIn("evil", json.dumps(rows))

    def test_rows_carry_exactly_the_resolver_fields_on_every_path(self):
        for reply in ({"verdicts": [{"claim": 1, "verdict": "partial", "why": "only half of it"}]},
                      {"garbage": True, "citations": POISON}):
            doc = self.run_with(reply)
            for r in doc["items"][0]["verification"]["citations"]:
                self.assertEqual(set(r), {"source", "ref", "resolution", "match_score", "url", "matched_title"})
        doc = self.run_with({"verdicts": [{"claim": 1, "verdict": "partial", "why": "only half of it"}]})
        self.assertIn("only half", doc["items"][0]["verification"]["why"])

    def test_source_ordinals_in_why_become_source_labels(self):
        doc = {"items": [{"insight": "x", "sources": [{"id": "press", "ref": "Ferry timetable"}]}]}
        verify(doc, SOURCES, {}, fixed({"verdicts": [{"claim": 1, "verdict": "supported", "why": "Source 1 says so"}]}),
               citation_spans_enabled=False, sharded=False)
        self.assertEqual(doc["items"][0]["verification"]["why"], "Press says so")


def partial_brief(claim="Alpha establishes the first half and Beta establishes the missing half."):
    return {"items": [{"insight": claim, "sources": [{"id": "feed", "ref": "Alpha"}],
                       "verification": {"verdict": "partial", "why": "Alpha alone is incomplete", "citations": []}}]}


def partial_sources(with_beta=True):
    rows = [{"title": "Alpha", "text": "Alpha establishes the first half."}]
    if with_beta:
        rows.append({"title": "Beta", "text": "Beta establishes the missing half."})
    return {"feed": rows}


class EvidenceCompletionTests(unittest.TestCase):
    def test_split_support_completes_and_appends_one_resolved_source(self):
        doc, calls = partial_brief(), []

        def affirmative(prompt):
            calls.append(prompt)
            return {"combined_verdict": "completed", "candidate_adds_material_support": True,
                    "why": "Beta supplies the missing half."}
        rec = complete_partials(doc, partial_sources(), {}, enabled=True, decision_call=affirmative)
        item = doc["items"][0]
        self.assertEqual(item["verification"]["verdict"], "supported-completed")
        self.assertEqual([s["ref"] for s in item["sources"]], ["Alpha", "Beta"])
        self.assertEqual((len(calls), rec["attempted"], rec["completed"]), (1, 1, 1))
        self.assertNotIn("text", item["verification"]["completion"]["candidate"])
        self.assertNotIn("Alpha establishes", str(rec))

    def test_without_a_candidate_it_stays_partial_and_makes_no_call(self):
        doc = partial_brief()
        rec = complete_partials(doc, partial_sources(with_beta=False), {}, enabled=True,
                                decision_call=lambda p: self.fail("must not call the model"))
        self.assertEqual((doc["items"][0]["verification"]["verdict"], rec["attempted"],
                          rec["events"][0]["status"]), ("partial", 0, "skipped"))

    def test_the_cap_counts_failed_calls_and_later_claims_are_budget_skipped(self):
        doc = partial_brief()
        doc["items"] *= 7
        doc = copy.deepcopy(doc)
        calls = []

        def invalid(prompt):
            calls.append(prompt)
            return {"combined_verdict": "wrong", "candidate_adds_material_support": False, "why": "invalid"}
        rec = complete_partials(doc, partial_sources(), {}, enabled=True, cap=6, decision_call=invalid)
        self.assertEqual((len(calls), rec["attempted"], rec["invalid"], rec["budget_skipped"]), (6, 6, 6, 1))
        self.assertTrue(all(i["verification"]["verdict"] == "partial" for i in doc["items"]))

    def test_disabled_is_inert(self):
        doc = partial_brief()
        before = copy.deepcopy(doc)
        rec = complete_partials(doc, partial_sources(), {}, enabled=False,
                                decision_call=lambda p: self.fail("must not call the model"))
        self.assertEqual((doc, rec["status"]), (before, "off"))

    def test_through_a_model_adapter(self):
        from claim_seal.adapters import ReplayJudge
        claim = partial_brief()["items"][0]["insight"]
        model = ReplayJudge({}, {claim: {"combined_verdict": "completed",
                                         "candidate_adds_material_support": True, "why": "w"}})
        doc = partial_brief()
        rec = complete_partials(doc, partial_sources(), {}, enabled=True, model=model)
        self.assertEqual((rec["completed"], doc["items"][0]["verification"]["verdict"]), (1, "supported-completed"))


class SourceLineageTests(unittest.TestCase):
    def test_digests_are_recorded_once_and_rechecked(self):
        captured = capture_sources(SOURCES)
        rows = source_lineage(captured)
        self.assertEqual([r["capture_check"] for r in rows], ["ok", "ok"])
        self.assertEqual(rows[0]["response_id"], "press-1")
        self.assertIsNone(rows[0]["raw_artifact"])
        self.assertNotIn("reopened on Monday", json.dumps(rows))
        captured["press"]["items"][0]["body"] = "edited after capture"
        self.assertEqual(source_lineage(captured)[0]["capture_check"], "LINEAGE-RAW-DIGEST")
        captured = capture_sources(SOURCES)
        captured["press"]["items"][1]["normalized_sha256"] = "0" * 64
        self.assertEqual(source_lineage(captured)[1]["capture_check"], "LINEAGE-NORMALIZED-DIGEST")

    def test_capture_does_not_repin_digests_an_item_already_carries(self):
        captured = capture_sources(SOURCES)
        captured["press"]["items"][0]["text"] = "changed"
        again = capture_sources(captured)
        self.assertEqual(source_lineage(again)[0]["capture_check"], "LINEAGE-RAW-DIGEST")

    def test_undecodable_bytes_are_refused(self):
        raw = b"\xff not utf-8"
        got = digests_for(raw)
        self.assertIsNone(got["normalized_sha256"])
        self.assertEqual(got["normalize_error"], "UnicodeDecodeError")
        rows = source_lineage({"s": [{"title": "t", "text": "x", "body": raw, "raw_sha256": got["raw_sha256"]}]})
        self.assertEqual(rows[0]["capture_check"], "LINEAGE-NORMALIZE-FAILED")

    def test_norm_v1_normalizes_line_endings_and_composes_unicode(self):
        self.assertEqual(normalize(NORMALIZER_ID, "a\r\nb\rc".encode()), "a\nb\nc")
        self.assertEqual(normalize(NORMALIZER_ID, "Café".encode()), "Café")
        with self.assertRaises(KeyError):
            normalize("norm/v9", b"x")


if __name__ == "__main__":
    unittest.main()

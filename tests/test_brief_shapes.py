"""A lone value where a list is expected, at every entry point.

A writer (often a model) sometimes puts the detail paragraph in a string instead of a list of
strings, or one citation object where a list of citations belongs. Iterating a string walks its
characters, so a string detail used to become one "line" per character ("detail[0]" = "T"), and
a lone citation object crashed pass 1 or was dropped. The rule (claim_seal.resolve.shape_brief):
a string detail is ONE detail sentence, [detail]; a lone object in items, sources or analysis is
a one-element list; shapes with no safe reading raise BriefError (one reason for all of them:
content would be invented, or dropped without being seen). Every call site of the rule is pinned
by a direct call here, so deleting any one of them fails a test. Every source text here is
synthetic, written for these tests.
"""
from __future__ import annotations

import contextlib
import copy
import importlib
import io
import json
import os
import tempfile
import unittest

import helpers  # noqa: F401
from claim_seal import judge, line_states, resolve, resolve_only
from claim_seal.adapters import ReplayJudge
from claim_seal import build_claim_evidence, capture_sources
from claim_seal.attribution import attribution_checks, check_span
from claim_seal.cli import main as cli_main
from claim_seal.evidence import evidence_candidates
from claim_seal.relations import item_lines
from claim_seal.spans import span_fields

# The package re-exports the function resolve(), which hides the submodule of the same name.
resolve_mod = importlib.import_module("claim_seal.resolve")

INSIGHT = "The island ferry will run a second evening crossing from June."
WHY = "Commuters on the late shift have had no way home after eight in the evening."
DETAIL = ("The harbour board approved the extra crossing at its May meeting. "
          "Fares stay the same as on the daytime boats.")
TITLE = "Board approves second evening crossing"
CITE = {"id": "harbour", "ref": TITLE}
STATEMENT = "A second crossing suggests the board expects evening demand to grow."

SOURCES = {"harbour": {"label": "Harbour notices", "items": [
    {"title": TITLE,
     "text": "The harbour board approved a second evening ferry crossing at its May meeting, "
             "running from June. Fares are unchanged from the daytime service."}]}}

ANSWERS = {
    INSIGHT: {"verdict": "supported", "why": "the notice states the second crossing from June",
              "relation": "says", "strength": 90,
              "citation_spans": [{"citation_index": 0,
                                  "span_quote": "The harbour board approved the extra crossing",
                                  "span_relation": "says"}],
              "evidence_spans": [{"citation_index": 0,
                                  "support_span": "approved a second evening ferry crossing"}]},
    STATEMENT: {"verdict": "partial", "why": "growth is an inference from the approval",
                "relation": "suggests", "strength": 40},
}


def brief(detail=DETAIL, sources=None, analysis=None):
    item = {"insight": INSIGHT, "why_it_matters": WHY, "detail": detail,
            "sources": copy.deepcopy([CITE]) if sources is None else sources}
    if analysis is not None:
        item["analysis"] = analysis
    return {"items": [item]}


def run(doc):
    return judge(doc, copy.deepcopy(SOURCES), ReplayJudge(ANSWERS))


# One verdict for claim 1, as the judge's parsed entry, for a direct build_claim_evidence call.
VERDICTS = {0: {"verdict": "supported", "relation": "says", "strength": 90,
                "evidence_spans": [{"citation_index": 0,
                                    "support_span": "approved a second evening ferry crossing"}]}}


def badly_shaped():
    """A lone item (not in a list) with a string detail and a lone citation object."""
    return {"items": {"insight": INSIGHT, "why_it_matters": WHY, "detail": DETAIL,
                      "sources": dict(CITE)}}


def spanned(doc):
    """doc with a judged citation whose span sits in detail[0], as judge() leaves it."""
    items = doc["items"] if isinstance(doc["items"], list) else [doc["items"]]
    items[0]["verification"] = {"verdict": "supported", "citations": [
        {"source": "harbour", "ref": TITLE, "resolution": "exact",
         "span": {"field": "detail[0]", "start": 70, "end": 80, "quote": "Fares stay",
                  "sentence_relation": "says"}}]}
    return doc


class StringDetailIsOneSentenceTests(unittest.TestCase):
    def test_line_states_prints_the_string_detail_as_one_line_not_one_per_character(self):
        states = line_states(run(brief())["brief"])
        detail_lines = [line for line in states["lines"] if line["field"].startswith("detail")]
        self.assertEqual([(line["field"], line["text"]) for line in detail_lines],
                         [("detail[0]", DETAIL)])
        self.assertFalse([line for line in states["lines"] if len(line["text"] or "") < 2])

    def test_judge_gives_the_same_answer_for_a_string_and_for_a_one_element_list(self):
        as_string, as_list = run(brief(DETAIL)), run(brief([DETAIL]))
        self.assertEqual(as_string["brief"], as_list["brief"])
        self.assertEqual(as_string["brief"]["items"][0]["detail"], [DETAIL])
        self.assertEqual(as_string["claim_evidence"], as_list["claim_evidence"])
        self.assertEqual(as_string["attribution"], as_list["attribution"])
        self.assertEqual(line_states(as_string["brief"]), line_states(as_list["brief"]))

    def test_a_span_quoted_from_the_string_detail_anchors_in_detail_0(self):
        result = run(brief())
        cite = result["brief"]["items"][0]["verification"]["citations"][0]
        self.assertEqual(cite["span"]["field"], "detail[0]")
        detail = result["brief"]["items"][0]["detail"][0]
        self.assertEqual(detail[cite["span"]["start"]:cite["span"]["end"]], cite["span"]["quote"])

    def test_span_fields_reads_a_string_detail_as_one_field(self):
        claim = {"kind": "item", "index": 0, "claim": INSIGHT}
        self.assertEqual(span_fields(brief(), claim),
                         {"insight": INSIGHT, "why_it_matters": WHY, "detail[0]": DETAIL})

    def test_attribution_reads_the_whole_sentence_of_a_string_detail(self):
        doc = brief("The harbour board reported that the crossing is approved.")
        doc["items"][0]["verification"] = {"verdict": "supported", "citations": [
            {"source": "harbour", "ref": TITLE, "resolution": "exact",
             "span": {"field": "detail[0]", "start": 0, "end": 10, "quote": "The harbou",
                      "sentence_relation": "says"}}]}
        rows = attribution_checks(doc, {"harbour": "Harbour notices"})
        self.assertEqual([r["sentence"] for r in rows],
                         ["The harbour board reported that the crossing is approved."])

    def test_read_only_entry_points_do_not_modify_the_callers_brief(self):
        for doc in (brief(), spanned(badly_shaped())):
            before = copy.deepcopy(doc)
            claim = {"kind": "item", "index": 0, "claim": INSIGHT}
            resolve(doc, SOURCES)
            line_states(doc)
            attribution_checks(doc)
            span_fields(doc, claim)
            evidence_candidates(doc, capture_sources(SOURCES))
            build_claim_evidence(doc, capture_sources(SOURCES), VERDICTS)
            judge(doc, copy.deepcopy(SOURCES), ReplayJudge(ANSWERS))
            self.assertEqual(doc, before)

    def test_resolve_only_annotates_the_brief_it_is_given_in_the_list_shape(self):
        doc = brief()
        resolve_only(doc, SOURCES, {})
        self.assertEqual(doc["items"][0]["detail"], [DETAIL])
        self.assertEqual(doc["items"][0]["verification"]["verdict"], "cites-real")

    def test_a_well_shaped_brief_is_returned_as_the_same_object(self):
        doc = brief([DETAIL])
        self.assertIs(resolve_mod.shape_brief(doc), doc)
        self.assertEqual(resolve_mod.shape_brief({"items": [{"insight": "x", "detail": "",
                                                             "sources": [], "analysis": None}]}),
                         {"items": [{"insight": "x", "detail": "", "sources": [], "analysis": None}]})


class LoneObjectIsAOneElementListTests(unittest.TestCase):
    def test_a_lone_citation_object_resolves_like_a_one_element_list(self):
        report = resolve(brief(sources=dict(CITE)), SOURCES)
        self.assertEqual(report["totals"], {"citations": 1, "exact": 1, "fuzzy": 0, "unresolved": 0})
        self.assertEqual(report, resolve(brief(sources=[dict(CITE)]), SOURCES))

    def test_a_lone_analysis_statement_is_checked_not_dropped(self):
        statement = {"text": STATEMENT, "sources": dict(CITE)}
        report = resolve(brief(analysis=statement), SOURCES)
        self.assertEqual([c["kind"] for c in report["claims"]], ["item", "analysis"])
        self.assertEqual(report["totals"]["exact"], 2)
        result = run(brief(analysis=statement))
        self.assertEqual(result["brief"]["items"][0]["analysis"][0]["verification"]["verdict"],
                         "partial")
        self.assertEqual(result["brief"], run(brief(analysis=[{"text": STATEMENT,
                                                               "sources": [dict(CITE)]}]))["brief"])

    def test_a_lone_item_and_a_lone_connection_citation_resolve(self):
        doc = {"items": brief()["items"][0],
               "connection": {"text": "The board and the timetable agree.", "sources": dict(CITE)}}
        report = resolve(doc, SOURCES)
        self.assertEqual([c["kind"] for c in report["claims"]], ["item", "connection"])
        self.assertEqual(report["totals"]["exact"], 2)


class EveryShapeCallIsPinnedTests(unittest.TestCase):
    """A direct call to each reader that shapes its own input, with a badly shaped brief.

    These readers are usually reached with a brief an entry point already shaped, so the
    end-to-end tests above stay green if one of their own shape calls is deleted. Each test here
    calls one reader directly and compares it with the same call on the well-shaped brief.
    """

    def well_shaped(self):
        return {"items": [{"insight": INSIGHT, "why_it_matters": WHY, "detail": [DETAIL],
                           "sources": [dict(CITE)]}]}

    def test_build_claim_evidence_reads_a_badly_shaped_brief_like_the_well_shaped_one(self):
        captured = capture_sources(SOURCES)
        record = build_claim_evidence(badly_shaped(), captured, VERDICTS)
        self.assertEqual(record, build_claim_evidence(self.well_shaped(), captured, VERDICTS))
        self.assertEqual([o["support_state"] for o in record["outcomes"]], ["supported"])

    def test_build_claim_evidence_refuses_an_unreadable_brief(self):
        captured = capture_sources(SOURCES)
        for doc, where in ((brief({"a": "Fares stay the same."}), "items[0].detail"),
                           (brief(analysis=[STATEMENT]), "items[0].analysis[0]"),
                           ({"items": brief()["items"], "connection": [{"text": STATEMENT}]},
                            "connection")):
            with self.subTest(where=where):
                with self.assertRaises(resolve_mod.BriefError) as caught:
                    build_claim_evidence(doc, captured, VERDICTS)
                self.assertIn(where + " must be", str(caught.exception))

    def test_evidence_candidates_reads_a_lone_citation_like_a_list_of_one(self):
        captured = capture_sources(SOURCES)
        candidates = evidence_candidates(badly_shaped(), captured)
        self.assertEqual(list(candidates), [(0, 0)])
        self.assertEqual(candidates, evidence_candidates(self.well_shaped(), captured))

    def test_span_fields_reads_a_lone_item(self):
        claim = {"kind": "item", "index": 0, "claim": INSIGHT}
        self.assertEqual(span_fields(badly_shaped(), claim),
                         {"insight": INSIGHT, "why_it_matters": WHY, "detail[0]": DETAIL})

    def test_attribution_checks_reads_a_lone_item(self):
        rows = attribution_checks(spanned(badly_shaped()))
        self.assertEqual(rows, attribution_checks(spanned(self.well_shaped())))
        self.assertEqual([r["sentence"] for r in rows],
                         ["Fares stay the same as on the daytime boats."])

    def test_check_span_called_directly_reads_a_string_detail_as_one_sentence(self):
        item = {"insight": INSIGHT, "detail": DETAIL}
        citation = spanned({"items": [item]})["items"][0]["verification"]["citations"][0]
        self.assertEqual(check_span(item, citation)["sentence"],
                         "Fares stay the same as on the daytime boats.")

    def test_item_lines_called_directly_reads_a_string_detail_as_one_line(self):
        self.assertEqual(item_lines({"insight": INSIGHT, "detail": DETAIL}),
                         [("insight", "insight", INSIGHT), ("detail", "detail[0]", DETAIL)])


class UnreadableShapesAreRefusedTests(unittest.TestCase):
    BAD = {
        "detail as an object": (brief({"a": "Fares stay the same."}), "items[0].detail"),
        "detail as a number": (brief(3), "items[0].detail"),
        "sources as a string": (brief(sources=TITLE), "items[0].sources"),
        "a citation that is a string": (brief(sources=[TITLE]), "items[0].sources[0]"),
        "analysis as a string": (brief(analysis=STATEMENT), "items[0].analysis"),
        "an item that is a string": ({"items": [INSIGHT]}, "items[0]"),
        "items as a string": ({"items": INSIGHT}, "items"),
        "an analysis citation as a string": (brief(analysis=[{"text": STATEMENT, "sources": TITLE}]),
                                             "items[0].analysis[0].sources"),
        # The same loss as a string analysis, so the same refusal: it has no sources and would
        # be dropped unseen, whether it stands alone or sits inside a list.
        "an analysis statement that is a string": (brief(analysis=[STATEMENT]),
                                                   "items[0].analysis[0]"),
        "an analysis statement that is a number": (brief(analysis=[{"text": STATEMENT}, 7]),
                                                   "items[0].analysis[1]"),
        # One connection per brief: a list is refused at any length, a list of one included.
        "connection as a list of one": ({"items": brief()["items"],
                                         "connection": [{"text": STATEMENT, "sources": [CITE]}]},
                                        "connection"),
        "connection as a string": ({"items": brief()["items"], "connection": STATEMENT},
                                   "connection"),
        "connection as a number": ({"items": brief()["items"], "connection": 0}, "connection"),
        # 0, False and {} are not "no detail": they follow the rule for their type.
        "detail as 0": (brief(0), "items[0].detail"),
        "detail as False": (brief(False), "items[0].detail"),
        "detail as an empty object": (brief({}), "items[0].detail"),
        "a detail entry that is not a string": (brief(["Fares stay the same.", 3]),
                                                "items[0].detail[1]"),
        "a connection detail as an empty object": (
            {"items": brief()["items"], "connection": {"text": STATEMENT, "detail": {}}},
            "connection.detail"),
        "sources as 0": (brief(sources=0), "items[0].sources"),
        "sources as an empty string": (brief(sources=""), "items[0].sources"),
    }

    def test_every_entry_point_refuses_with_the_field_named(self):
        claim = {"kind": "item", "index": 0, "claim": INSIGHT}
        for name, (doc, where) in self.BAD.items():
            for entry, call in (("resolve", lambda d: resolve(d, SOURCES)),
                                ("resolve_only", lambda d: resolve_only(d, SOURCES, {})),
                                ("judge", lambda d: judge(d, SOURCES, ReplayJudge(ANSWERS))),
                                ("line_states", line_states),
                                ("attribution_checks", attribution_checks),
                                ("span_fields", lambda d: span_fields(d, claim)),
                                ("evidence_candidates",
                                 lambda d: evidence_candidates(d, capture_sources(SOURCES))),
                                ("build_claim_evidence",
                                 lambda d: build_claim_evidence(d, capture_sources(SOURCES),
                                                                VERDICTS))):
                with self.subTest(shape=name, entry=entry):
                    with self.assertRaises(resolve_mod.BriefError) as caught:
                        call(copy.deepcopy(doc))
                    self.assertIn(where + " must be", str(caught.exception))

    def test_none_and_an_empty_list_are_still_no_value(self):
        plain = resolve(brief([]), SOURCES)
        for name, doc in (
                ("detail None", brief(None)), ("detail []", brief([])), ("detail ''", brief("")),
                ("analysis None", brief([], analysis=None)), ("analysis []", brief([], analysis=[])),
                ("connection None", {"items": brief([])["items"], "connection": None}),
                ("connection []", {"items": brief([])["items"], "connection": []})):
            with self.subTest(shape=name):
                self.assertEqual(resolve(doc, SOURCES), plain)
                self.assertIs(resolve_mod.shape_brief(doc), doc)
        for empty in (None, []):
            doc = {"items": [{"insight": INSIGHT, "sources": empty}]}
            with self.subTest(sources=empty):
                self.assertIs(resolve_mod.shape_brief(doc), doc)
                self.assertEqual(resolve(doc, SOURCES)["totals"]["citations"], 0)

    def test_the_cli_refuses_an_unreadable_brief_with_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for name, doc in (("brief", brief({"a": "Fares stay the same."})), ("sources", SOURCES)):
                paths[name] = os.path.join(tmp, name + ".json")
                with open(paths[name], "w", encoding="utf-8") as handle:
                    json.dump(doc, handle)
            err, out = io.StringIO(), io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
                code = cli_main(["check", paths["brief"], paths["sources"]])
            self.assertEqual(code, 2)
            self.assertIn("refused: items[0].detail must be a list of strings or one string",
                          err.getvalue())


if __name__ == "__main__":
    unittest.main()

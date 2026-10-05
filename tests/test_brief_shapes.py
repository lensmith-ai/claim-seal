"""A lone value where a list is expected, at every entry point.

A writer (often a model) sometimes puts the detail paragraph in a string instead of a list of
strings, or one citation object where a list of citations belongs. Iterating a string walks its
characters, so a string detail used to become one "line" per character ("detail[0]" = "T"), and
a lone citation object crashed pass 1 or was dropped. The rule (claim_seal.resolve.shape_brief):
a string detail is ONE detail sentence, [detail]; a lone object in items, sources or analysis is
a one-element list; shapes with no safe reading raise BriefError. Every source text here is
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
from claim_seal.attribution import attribution_checks
from claim_seal.cli import main as cli_main
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
        doc = brief()
        before = copy.deepcopy(doc)
        resolve(doc, SOURCES)
        line_states(doc)
        attribution_checks(doc)
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
    }

    def test_every_entry_point_refuses_with_the_field_named(self):
        for name, (doc, where) in self.BAD.items():
            for entry, call in (("resolve", lambda d: resolve(d, SOURCES)),
                                ("resolve_only", lambda d: resolve_only(d, SOURCES, {})),
                                ("judge", lambda d: judge(d, SOURCES, ReplayJudge(ANSWERS))),
                                ("line_states", line_states)):
                with self.subTest(shape=name, entry=entry):
                    with self.assertRaises(resolve_mod.BriefError) as caught:
                        call(copy.deepcopy(doc))
                    self.assertIn(where + " must be", str(caught.exception))

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

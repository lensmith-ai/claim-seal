"""Forgiving relation words (one synonym list, unknown = can't tell) and the attribution-overclaim
heuristic (informational, never changes a verdict, not sealed)."""
from __future__ import annotations

import copy
import json
import unittest

import helpers  # noqa: F401
from helpers import fixed

from claim_seal import judge, line_states, verify_seal
from claim_seal.attribution import HEDGED, STRONG, find_attribution, judge_attribution
from claim_seal.evidence import build_claim_evidence, capture_sources
from claim_seal.prompts import build_prompt
from claim_seal.relations import RELATION_SYNONYMS, SPAN_RELATIONS, read_relation, through_line
from claim_seal.resolve import resolve

SAYS = ("states", "stated", "reports", "reported", "notes", "noted", "describes", "described", "writes",
        "wrote", "announces", "announced", "confirms", "confirmed", "says", "said")
STEP = {"infers": "implies", "inferred": "implies", "argues": "implies", "argued": "implies",
        **{v: v for v in ("implies", "suggests", "connotes", "follows", "explains", "posits", "illustrates")}}
DISAGREES = ("contradicts", "contradicted", "refutes", "refuted", "denies", "denied")


class RelationWordTests(unittest.TestCase):
    def test_states_it_words_read_as_says(self):
        for word in SAYS:
            for form in (word, word.upper(), "  " + word.title() + "\n"):
                self.assertEqual(read_relation(form)[0], "says", repr(form))
        self.assertEqual(read_relation("described"), ("says", "synonym"))
        self.assertEqual(read_relation("says"), ("says", "verb"))

    def test_step_beyond_words_keep_a_canonical_step_verb(self):
        for word, verb in STEP.items():
            for form in (word, " " + word.upper() + " "):
                self.assertEqual(read_relation(form)[0], verb, repr(form))
                self.assertEqual(through_line(form), "step-beyond")
        for verb in ("implies", "suggests", "connotes", "follows", "explains", "posits", "illustrates"):
            self.assertEqual(read_relation(verb), (verb, "verb"))

    def test_disagreeing_words_read_as_contradicts(self):
        for word in DISAGREES:
            for form in (word, word.capitalize() + "  "):
                self.assertEqual(read_relation(form)[0], "contradicts", repr(form))

    def test_anything_else_is_undetermined_and_marked_unknown(self):
        for word in ("pondered", "Hallucinates ", "supports", "agrees"):
            self.assertEqual(read_relation(word), ("undetermined", "unknown"), word)
        self.assertEqual(read_relation(""), (None, "empty"))

    def test_the_list_lives_in_one_place_and_only_maps_into_the_vocabulary(self):
        self.assertTrue(set(RELATION_SYNONYMS.values()) <= set(SPAN_RELATIONS))
        self.assertFalse(set(RELATION_SYNONYMS) & set(SPAN_RELATIONS))

    def test_the_judges_prompt_menu_is_still_the_ten_words(self):
        report = resolve({"items": [{"insight": "x", "sources": [{"id": "s", "ref": "t"}]}]},
                         {"s": [{"title": "t", "text": "body"}]})
        prompt, _ = build_prompt(report, citation_spans_enabled=True, sharded=True, claim_evidence_candidates={})
        menu = ('"says"|"implies"|"suggests"|"connotes"|"follows"|"explains"|"posits"|"illustrates"|'
                '"contradicts"|"undetermined"')
        self.assertEqual(prompt.count(menu), 2)  # span_relation and relation
        for synonym in ("described", "refutes", "argued"):
            self.assertNotIn('"' + synonym + '"', prompt)

    def test_evidence_rows_keep_the_raw_word_inside_the_seal(self):
        doc = {"items": [{"insight": "c", "sources": [{"id": "s", "ref": "T"}]}]}
        src = capture_sources({"s": [{"title": "T", "text": "The body says it."}]})
        for word, verb, read in (("described", "says", "synonym"), ("pondered", "undetermined", "unknown")):
            rec = build_claim_evidence(doc, src, {0: {"verdict": "supported", "relation": word, "evidence_spans": [
                {"citation_index": 0, "support_span": "The body says it."}]}})
            row = rec["outcomes"][0]
            self.assertEqual((row["relation"], row["relation_raw"], row["relation_read"]), (verb, word, read))
            self.assertEqual(row["relation_source"], "compat-default")
            self.assertTrue(verify_seal(rec))
            tampered = copy.deepcopy(rec)
            tampered["outcomes"][0]["relation_raw"] = "says"
            self.assertFalse(verify_seal(tampered))
        exact = build_claim_evidence(doc, src, {0: {"verdict": "supported", "relation": "says"}})
        self.assertNotIn("relation_raw", exact["outcomes"][0])


class AttributionFinderTests(unittest.TestCase):
    def test_strong_hedged_and_none(self):
        self.assertEqual(find_attribution("The council says the bridge reopened."), ("says", "strong"))
        self.assertEqual(find_attribution("The council argues the bridge reopened."), ("argues", "hedged"))
        self.assertEqual(find_attribution("The bridge reopened on Monday."), (None, None))

    def test_past_tense(self):
        self.assertEqual(find_attribution("The council reported that the bridge reopened."), ("reported", "strong"))
        self.assertEqual(find_attribution("Engineers predicted a June opening."), ("predicted", "hedged"))
        self.assertEqual(find_attribution("A survey found strong support."), ("found", "strong"))

    def test_two_verbs_take_the_first_after_the_source_name_else_the_first(self):
        sentence = "The mayor claimed a delay, but the council reported the bridge reopened."
        self.assertEqual(find_attribution(sentence), ("claimed", "hedged"))
        self.assertEqual(find_attribution(sentence, ["Council"]), ("reported", "strong"))

    def test_a_source_name_before_the_verb(self):
        self.assertEqual(find_attribution("Officials claimed delays; Harbour Press notes the reopening.",
                                          ["Harbour Press"]), ("notes", "strong"))

    def test_a_name_with_no_verb_after_it_falls_back_to_the_first_verb(self):
        self.assertEqual(find_attribution("Officials claimed delays, as Harbour Press did.", ["Harbour Press"]),
                         ("claimed", "hedged"))

    def test_false_positive_guards(self):
        for sentence in ("The report was late.", "The reports were late.", "Its notes were brief.",
                         "The foundation stone was laid.", "These findings matter.", "Their claims were old.",
                         "A statement followed.",
                         # no determiner in front: only the word boundary keeps these out
                         "Foundation work began in May.", "Reportedly, the bridge reopened.",
                         "Unconfirmed figures circulated.", "Showroom traffic rose."):
            self.assertEqual(find_attribution(sentence), (None, None), sentence)
        self.assertEqual(find_attribution("The reports say little, but the council said yes."), ("said", "strong"))

    def test_the_lists_are_disjoint_and_include_past_tense(self):
        self.assertFalse(set(STRONG) & set(HEDGED))
        for word in ("said", "stated", "reported", "described", "wrote", "noted", "announced",
                     "confirmed", "showed", "found"):
            self.assertIn(word, STRONG)


class AttributionFlagTests(unittest.TestCase):
    def test_each_flag(self):
        self.assertEqual(judge_attribution("says", "strong", "implies")["flag"], "overclaim")
        self.assertIn("only implies it", judge_attribution("says", "strong", "implies")["message"])
        self.assertEqual(judge_attribution("reported", "strong", "contradicts")["flag"], "misattribution")
        self.assertEqual(judge_attribution("argues", "hedged", "says")["flag"], "understated")

    def test_no_flag_cases(self):
        for args in (("says", "strong", "says"), ("argues", "hedged", "implies"),
                     ("argues", "hedged", "contradicts"), ("says", "strong", "undetermined"),
                     ("says", "strong", None), (None, None, "implies")):
            out = judge_attribution(*args)
            self.assertIsNone(out["flag"], args)
            self.assertIsNone(out["message"], args)
        self.assertEqual(set(judge_attribution("says", "strong", "implies")),
                         {"attribution_verb", "attribution_strength", "relation", "flag", "message"})


SOURCES = {"press": {"label": "Harbour Press", "items": [
    {"title": "Bridge reopens", "text": "The harbour bridge reopened on Monday after six weeks of repairs."}]}}


def run(detail, span_quote, span_relation, verdict="supported"):
    doc = {"items": [{"insight": "The bridge is open again.", "detail": [detail],
                      "sources": [{"id": "press", "ref": "Bridge reopens"}]}]}
    reply = {"verdicts": [{"claim": 1, "verdict": verdict, "why": "w", "relation": "says",
                           "citation_spans": [{"citation_index": 0, "span_quote": span_quote,
                                               "span_relation": span_relation}]}]}
    return judge(doc, SOURCES, fixed(reply), local_placement=False)


class AttributionInPipelineTests(unittest.TestCase):
    def test_overclaim_is_surfaced_and_changes_nothing_else(self):
        result = run("Harbour Press says traffic will double.", "traffic will double", "implies")
        flagged = [r for r in result["attribution"] if r["flag"]]
        self.assertEqual([(r["flag"], r["attribution_verb"], r["relation"]) for r in flagged],
                         [("overclaim", "says", "implies")])
        states = line_states(result["brief"], labels={"press": "Harbour Press"})
        line = next(l for l in states["lines"] if l["field"] == "detail[0]")
        self.assertEqual(line["state"], "reading")
        self.assertEqual(line["attribution"][0]["flag"], "overclaim")
        quiet = run("Harbour Press argues traffic will double.", "traffic will double", "implies")
        self.assertEqual([l["state"] for l in line_states(quiet["brief"])["lines"]],
                         [l["state"] for l in states["lines"]])
        self.assertEqual(result["brief"]["items"][0]["verification"]["verdict"],
                         quiet["brief"]["items"][0]["verification"]["verdict"])

    def test_misattribution_on_a_left_out_line(self):
        result = run("Harbour Press reported the bridge stays shut.", "the bridge stays shut", "contradicts")
        states = line_states(result["brief"])
        self.assertEqual(states["left_out"][0]["attribution"][0]["flag"], "misattribution")

    def test_understated_is_informational(self):
        result = run("Harbour Press argued that repairs took six weeks.", "repairs took six weeks", "says")
        row = next(r for r in result["attribution"] if r["field"] == "detail[0]")
        self.assertEqual(row["flag"], "understated")
        line = next(l for l in line_states(result["brief"])["lines"] if l["field"] == "detail[0]")
        self.assertEqual(line["state"], "supported")

    def test_attribution_is_not_inside_the_seal(self):
        result = run("Harbour Press says traffic will double.", "traffic will double", "implies")
        encoded = json.dumps(result["claim_evidence"])
        for key in ('"attribution', '"flag"', '"overclaim"'):
            self.assertNotIn(key, encoded)
        before = result["claim_evidence"]["verification"]["seal"]["value"]
        result["attribution"][0]["flag"] = None
        self.assertTrue(verify_seal(result["claim_evidence"]))
        self.assertEqual(result["claim_evidence"]["verification"]["seal"]["value"], before)


if __name__ == "__main__":
    unittest.main()

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
DISAGREES = ("contradicts", "contradicted", "refutes", "refuted", "denies", "denied", "disputes", "disputed",
             "disagrees", "disagreed", "conflicts", "rejects", "rejected")


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


class Round6AttributionFixes(unittest.TestCase):
    NAMES = ["Harbour Press", "Bridge reopens", "press"]

    def find(self, sentence, names=None):
        return find_attribution(sentence, self.NAMES if names is None else names)

    def test_negation_within_three_words_skips_the_verb(self):
        for sentence in ("Harbour Press never said traffic will double.",
                         "Harbour Press has not confirmed the date.",
                         "Harbour Press did not report it, and never stated it.",
                         "Harbour Press didn't say so; it hasn't noted a date.",
                         "Harbour Press doesn’t claim a figure."):
            self.assertEqual(self.find(sentence), (None, None), sentence)
        # a negation further away is a documented misfire, pinned so the README stays honest
        self.assertEqual(self.find("Nothing in Harbour Press states that tolls will rise."), ("states", "strong"))

    def test_quoted_speech_is_skipped(self):
        self.assertEqual(self.find('"We said it would double," the mayor told Harbour Press.'), (None, None))
        self.assertEqual(self.find("“We confirmed it,” an official told Harbour Press."), (None, None))
        self.assertEqual(self.find('The mayor said "traffic will double", Harbour Press reports.'),
                         ("reports", "strong"))

    def test_questions_are_not_checked(self):
        for sentence in ("Who said traffic will double?", "Does Harbour Press say it will double? ",
                         "Why does the council claim delays?"):
            self.assertEqual(self.find(sentence), (None, None), sentence)

    def test_sentence_initial_plural_nouns_and_known_compounds(self):
        for sentence in ("Reports show traffic doubled.", "Notes from May mention it.",
                         "The United States sent engineers.", "Member states agreed on funding.",
                         "TV shows covered the reopening.", "Lost-and-found desks reopened."):
            self.assertEqual(self.find(sentence), (None, None), sentence)
        # documented misfires that remain
        self.assertEqual(self.find("Recent reports show traffic doubled."), ("reports", "strong"))
        self.assertEqual(self.find("It was found that the deck was unsafe."), ("found", "strong"))

    def test_a_listed_word_inside_the_source_name_is_skipped(self):
        names = ["Council Notes"]
        self.assertEqual(find_attribution("Council Notes: traffic will double.", names), (None, None))
        self.assertEqual(find_attribution("Council Notes reported a delay.", names), ("reported", "strong"))
        self.assertEqual(find_attribution("Council Notes: traffic will double.", []), ("Notes", "strong"))

    def test_hedged_past_forms(self):
        self.assertEqual(self.find("Harbour Press speculated that traffic would double."), ("speculated", "hedged"))
        self.assertEqual(self.find("Harbour Press thought traffic would double."), ("thought", "hedged"))
        self.assertEqual(self.find("The thought was idle."), (None, None))
        for present, past in (("argues", "argued"), ("claims", "claimed"), ("suggests", "suggested"),
                              ("believes", "believed"), ("predicts", "predicted"), ("proposes", "proposed"),
                              ("expects", "expected"), ("thinks", "thought"), ("speculates", "speculated")):
            self.assertIn(present, HEDGED)
            self.assertIn(past, HEDGED)

    def test_messages_say_possible_and_carry_the_reread_note(self):
        over = judge_attribution("says", "strong", "implies")["message"]
        mis = judge_attribution("said", "strong", "contradicts")["message"]
        self.assertTrue(over.startswith("possible overclaim:"))
        self.assertTrue(mis.startswith("possible misattribution:"))
        for message in (over, mis):
            self.assertIn("treat a flag as a prompt to reread the sentence, not as a finding", message)
        self.assertTrue(judge_attribution("argued", "hedged", "says")["message"].startswith("understated"))

    def test_line_states_never_crashes_on_non_string_text(self):
        for insight in (5, {"t": "x"}, None, ["a"]):
            brief = {"items": [{"insight": insight, "verification": {"verdict": "supported", "citations": [
                {"source": "s", "resolution": "exact",
                 "span": {"field": "insight", "start": 0, "sentence_relation": "says"}}]}}]}
            out = line_states(brief)
            self.assertEqual([l["state"] for l in out["lines"]], ["supported"], repr(insight))
            self.assertEqual([r["flag"] for r in out["lines"][0]["attribution"] if r["flag"]], [], repr(insight))

    def test_check_span_itself_returns_nothing_for_non_string_text(self):
        from claim_seal.attribution import check_span
        for text in (5, {"t": "x"}, ["a"]):
            citation = {"source": "s", "span": {"field": "insight", "start": 0, "sentence_relation": "says"}}
            self.assertIsNone(check_span({"insight": text}, citation), repr(text))
            citation = {"source": "s", "span": {"field": "detail[0]", "start": 0, "sentence_relation": "says"}}
            self.assertIsNone(check_span({"insight": "x", "detail": [text]}, citation), repr(text))

    def test_line_states_survives_a_heuristic_that_raises(self):
        import claim_seal.attribution as attribution_module
        original = attribution_module.check_span
        attribution_module.check_span = lambda *a, **k: 1 / 0
        try:
            out = line_states({"items": [{"insight": "Harbour Press says so.", "verification": {
                "verdict": "supported", "citations": [{"source": "s", "resolution": "exact", "span": {
                    "field": "insight", "start": 0, "sentence_relation": "implies"}}]}}]})
        finally:
            attribution_module.check_span = original
        self.assertEqual([l["state"] for l in out["lines"]], ["reading"])


class Round6ClaimLevelTests(unittest.TestCase):
    SRC = {"press": {"label": "Harbour Press", "items": [
        {"title": "Bridge reopens", "text": "The harbour bridge reopened on Monday after six weeks of repairs."}]}}

    def run_reply(self, verdict, relation, **kw):
        doc = {"items": [{"insight": "The bridge reopened on Monday.", "detail": ["Repairs took six weeks."],
                          "sources": [{"id": "press", "ref": "Bridge reopens"}]}]}
        reply = {"verdicts": [{"claim": 1, "verdict": verdict, "why": "w", "relation": relation, "strength": 70,
                               "citation_spans": [{"citation_index": 0, "span_quote": "Repairs took six weeks",
                                                   "span_relation": "says"}],
                               "evidence_spans": [{"citation_index": 0, "support_span": "reopened on Monday"}]}]}
        return judge(doc, self.SRC, fixed(reply), local_placement=False, **kw)

    def test_new_disagreement_synonyms_override_at_claim_level(self):
        for word in ("disputes", "Disputed ", "disagrees", "disagreed", "CONFLICTS", "rejects", "rejected"):
            self.assertEqual(read_relation(word), ("contradicts", "synonym"), word)
            for verdict in ("supported", "partial"):
                result = self.run_reply(verdict, word)
                self.assertEqual(result["brief"]["items"][0]["verification"]["verdict"], "unsupported", (word, verdict))
                self.assertEqual(line_states(result["brief"])["lines"], [])

    def test_an_unknown_claim_level_word_is_recorded_and_the_verdict_word_decides(self):
        result = self.run_reply("supported", "pondered")
        self.assertEqual(result["brief"]["items"][0]["verification"]["verdict"], "supported")
        row = result["claim_evidence"]["outcomes"][0]
        self.assertEqual((row["support_state"], row["relation"], row["relation_raw"], row["relation_read"]),
                         ("supported", "undetermined", "pondered", "unknown"))
        self.assertTrue(verify_seal(result["claim_evidence"]))

    def test_the_override_works_without_the_evidence_pass(self):
        for verdict in ("supported", "partial"):
            for sharded in (True, False):
                result = self.run_reply(verdict, "contradicts", evidence=False, sharded=sharded)
                self.assertNotIn("claim_evidence", result)
                self.assertEqual(result["brief"]["items"][0]["verification"]["verdict"], "unsupported",
                                 (verdict, sharded))
                self.assertEqual(line_states(result["brief"])["lines"], [])
        plain = self.run_reply("partial", "suggests", evidence=False)
        self.assertEqual(plain["brief"]["items"][0]["verification"]["verdict"], "partial")

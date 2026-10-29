"""The optional Jev first pass (claim_seal.jev). Every Jev reply here comes from a fake transport
or a loopback server: no test reaches the network."""
from __future__ import annotations

import copy
import http.server
import json
import os
import threading
import unittest
from unittest.mock import patch

import helpers
from helpers import corpus, per_claim_model, fixed, FROZEN_WHOLE_BRIEF_REPLY

from claim_seal import judge, verify_seal
from claim_seal.jev import (CHOICES, DEFAULT_THRESHOLD, FIRST_PASS_CONTRACT, JevError,
                            JevFirstPass, resolve_first_pass)
from claim_seal.seal import chain_value
from claim_seal.support import located_claims

KEY_ENV = "CLAIM_SEAL_TEST_JEV_KEY"
KEY = "test-key-value-never-in-output"

RELAY = "The relay handles ten thousand messages an hour after the queue rewrite."
LABS = "Two labs report the same memory saving from the compaction pass."
AUDIT = "The audit trail records the audit trail twice when the audit trail rotates."
TRADE = "Both the queue rewrite and the compaction pass trade peak speed for steadier tails."


def reply(choice, confidence, model="typesafe/jev-test", cost=0.00002):
    return json.dumps({"answers": {"relation": {"choice": choice, "confidence": confidence,
                                                "type": "choice"}},
                       "id": "gen-dec-test", "model": model, "provider": "Fake",
                       "usage": {"cost": cost, "input_tokens": 500, "output_tokens": 44}}).encode()


class FakeJev:
    """transport(endpoint, body, headers, timeout): answers from rule(claim, section) which returns
    (choice, confidence), raw bytes, or an exception to raise. Records every call."""

    def __init__(self, rule):
        self.rule = rule
        self.calls = []
        self._lock = threading.Lock()

    def __call__(self, endpoint, body, headers, timeout):
        req = json.loads(body)
        with self._lock:
            self.calls.append({"endpoint": endpoint, "request": req, "headers": dict(headers),
                               "timeout": timeout})
        out = self.rule(req["state"]["claim"], req["state"]["section"])
        if isinstance(out, BaseException):
            raise out
        if isinstance(out, bytes):
            return out
        return reply(*out)


def table(by_claim, default=("says_nothing", 0.5)):
    """rule from {claim text: (choice, conf) | {section substring: (choice, conf)}}."""
    def rule(claim, section):
        entry = by_claim.get(claim, default)
        if isinstance(entry, dict):
            for needle, out in entry.items():
                if needle in section:
                    return out
            return default
        return entry
    return rule


def fp(rule, **kw):
    fake = FakeJev(rule)
    kw.setdefault("api_key_env", KEY_ENV)
    return JevFirstPass(transport=fake, **kw), fake


def claims_of(result):
    brief = result["brief"]
    return {c.get("insight") or c.get("text"): c["verification"]
            for c in list(brief["items"]) + [brief["connection"]]}


def rows_for(result, n_item):
    return [r for r in result["claim_evidence"]["outcomes"]
            if r["claim_ref"]["kind"] == "item" and r["claim_ref"]["item_index"] == n_item]


def strip_first_pass(value):
    """A result with every first-pass field removed, for comparing against an off run."""
    if isinstance(value, dict):
        return {k: strip_first_pass(v) for k, v in value.items()
                if k not in ("decided_by", "decider", "first_pass", "wall_s", "queue_wait_s", "seal")}
    if isinstance(value, list):
        return [strip_first_pass(v) for v in value]
    return value


@patch.dict(os.environ, {KEY_ENV: KEY})
class DefaultOff(unittest.TestCase):
    def test_none_false_and_env_off_are_the_same_run(self):
        b, s = corpus()
        base = judge(b, s, per_claim_model()[0])
        for arg in (None, False):
            with self.subTest(first_pass=arg):
                again = judge(copy.deepcopy(b), copy.deepcopy(s), per_claim_model()[0], first_pass=arg)
                self.assertEqual(strip_first_pass(again), strip_first_pass(base))
                self.assertEqual(json.dumps(again["claim_evidence"]),
                                 json.dumps(base["claim_evidence"]))
        for value in (None, "", "off", "0"):
            env = {} if value is None else {"CLAIM_SEAL_JEV": value}
            with self.subTest(env=env), patch.dict(os.environ, env, clear=False):
                if value is None:
                    os.environ.pop("CLAIM_SEAL_JEV", None)
                self.assertIsNone(resolve_first_pass("env"))
                again = judge(copy.deepcopy(b), copy.deepcopy(s), per_claim_model()[0], first_pass="env")
                self.assertEqual(again["claim_evidence"], base["claim_evidence"])

    def test_off_adds_no_first_pass_fields(self):
        b, s = corpus()
        text = json.dumps(judge(b, s, per_claim_model()[0]))
        for word in ("decided_by", "decider", "first_pass", "jev"):
            self.assertNotIn(word, text)


@patch.dict(os.environ, {KEY_ENV: KEY})
class Threshold(unittest.TestCase):
    def run_relay(self, confidence, **kw):
        first, fake = fp(table({RELAY: ("supports", confidence)}), **kw)
        model, calls = per_claim_model()
        b, s = corpus()
        result = judge(b, s, model, first_pass=first)
        return result, fake, calls

    def test_at_the_threshold_jev_decides_and_the_judge_is_not_asked(self):
        result, fake, calls = self.run_relay(0.80)
        v = claims_of(result)[RELAY]
        self.assertEqual(v["decided_by"]["by"], "jev")
        self.assertEqual(v["verdict"], "supported")
        self.assertNotIn(RELAY, calls)               # the judge never saw it
        self.assertIn(LABS, calls)                   # the rest were judged
        self.assertEqual(result["verification"]["first_pass"]["decided_by_jev"], 1)

    def test_just_below_the_threshold_the_judge_decides(self):
        result, fake, calls = self.run_relay(0.79)
        v = claims_of(result)[RELAY]
        self.assertEqual(v["decided_by"]["by"], "judge")
        self.assertEqual(v["decided_by"]["reason"], "below-threshold")
        self.assertEqual(v["decided_by"]["answers"][0]["choice"], "supports")
        self.assertEqual(v["decided_by"]["answers"][0]["confidence"], 0.79)
        self.assertEqual(calls.get(RELAY), 1)
        b, s = corpus()
        off = claims_of(judge(b, s, per_claim_model()[0]))[RELAY]
        self.assertEqual(v["verdict"], off["verdict"])   # the judge's own answer, gated as always

    def test_the_threshold_is_configurable(self):
        result, _, calls = self.run_relay(0.85, threshold=0.90)
        self.assertEqual(claims_of(result)[RELAY]["decided_by"]["reason"], "below-threshold")
        self.assertEqual(result["verification"]["first_pass"]["settings"]["threshold"], 0.90)
        result, _, calls = self.run_relay(0.85, threshold=0.85)
        self.assertEqual(claims_of(result)[RELAY]["decided_by"]["by"], "jev")

    def test_the_default_threshold_is_0_80(self):
        self.assertEqual(DEFAULT_THRESHOLD, 0.80)
        self.assertEqual(JevFirstPass(api_key_env=KEY_ENV).threshold, 0.80)


@patch.dict(os.environ, {KEY_ENV: KEY})
class Mapping(unittest.TestCase):
    def decide(self, choice, claim=AUDIT):
        first, _ = fp(table({claim: (choice, 0.97)}))
        b, s = corpus()
        model, calls = per_claim_model()
        return judge(b, s, model, first_pass=first), calls

    def test_supports_is_supported_with_the_window_located_and_sealed(self):
        result, calls = self.decide("supports")
        v = claims_of(result)[AUDIT]
        self.assertEqual((v["verdict"], v["decided_by"]["by"], v["decided_by"]["choice"]),
                         ("supported", "jev", "supports"))
        self.assertNotIn("verdict_override", v)
        self.assertTrue(verify_seal(result["claim_evidence"]))
        (row,) = rows_for(result, 3)
        self.assertEqual((row["support_state"], row["evidence"], row["relation"], row["relation_source"]),
                         ("supported", "present", "undetermined", "jev-first-pass"))
        self.assertEqual(row["decider"]["section"], "evidence-window")
        self.assertEqual(row["decider"]["answers"], [{"citation_index": 0, "choice": "supports",
                                                      "confidence": 0.97, "model": "typesafe/jev-test"}])
        span = row["evidence_spans"][0]
        self.assertTrue(span["start"] >= 0 and span["end"] > span["start"])
        self.assertNotIn(AUDIT, calls)

    def test_contradicts_is_unsupported_with_relation_contradicts(self):
        result, _ = self.decide("contradicts")
        v = claims_of(result)[AUDIT]
        self.assertEqual(v["verdict"], "unsupported")
        (row,) = rows_for(result, 3)
        self.assertEqual((row["support_state"], row["relation"], row["evidence_spans"]),
                         ("unsupported", "contradicts", []))
        self.assertEqual(result["status"], "unsupported")

    def test_says_nothing_is_unsupported_with_no_relation(self):
        result, _ = self.decide("says_nothing")
        v = claims_of(result)[AUDIT]
        self.assertEqual(v["verdict"], "unsupported")
        (row,) = rows_for(result, 3)
        self.assertEqual((row["support_state"], row["relation"]), ("unsupported", "undetermined"))

    def test_jev_never_produces_partial(self):
        for choice in CHOICES:
            with self.subTest(choice=choice):
                result, _ = self.decide(choice)
                self.assertNotEqual(claims_of(result)[AUDIT]["verdict"], "partial")

    def test_supported_by_jev_passes_the_support_gate_only_through_a_located_row(self):
        result, _ = self.decide("supports")
        located = located_claims({"claims": [
            {"kind": "item", "index": i, "analysis_index": None} for i in range(4)]
            + [{"kind": "connection", "index": 0, "analysis_index": None}]},
            result["brief"], result["claim_evidence"])
        self.assertIn(3, located)


@patch.dict(os.environ, {KEY_ENV: KEY})
class SeveralCitations(unittest.TestCase):
    def run_labs(self, alpha, beta):
        first, fake = fp(table({LABS: {"resident memory": alpha, "second lab": beta}}))
        model, calls = per_claim_model()
        b, s = corpus()
        return judge(b, s, model, first_pass=first), calls, fake

    def test_one_call_per_cited_source(self):
        _, _, fake = self.run_labs(("supports", 0.9), ("supports", 0.9))
        sections = [c["request"]["state"]["section"] for c in fake.calls
                    if c["request"]["state"]["claim"] == LABS]
        self.assertEqual(len(sections), 2)

    def test_supports_and_says_nothing_is_supported_with_one_located_row(self):
        result, calls, _ = self.run_labs(("supports", 0.9), ("says_nothing", 0.95))
        self.assertEqual(claims_of(result)[LABS]["verdict"], "supported")
        rows = rows_for(result, 1)
        self.assertEqual([r["support_state"] for r in rows], ["supported", "unresolved"])
        self.assertNotIn(LABS, calls)

    def test_supports_and_contradicts_goes_to_the_judge(self):
        result, calls, _ = self.run_labs(("supports", 0.99), ("contradicts", 0.99))
        v = claims_of(result)[LABS]
        self.assertEqual((v["decided_by"]["by"], v["decided_by"]["reason"]), ("judge", "citations-disagree"))
        self.assertEqual(calls.get(LABS), 1)

    def test_one_source_below_the_threshold_sends_the_claim_to_the_judge(self):
        result, calls, _ = self.run_labs(("supports", 0.99), ("supports", 0.5))
        self.assertEqual(claims_of(result)[LABS]["decided_by"]["reason"], "below-threshold")
        self.assertEqual(calls.get(LABS), 1)


@patch.dict(os.environ, {KEY_ENV: KEY})
class Fallback(unittest.TestCase):
    def assert_same_as_off(self, result, base):
        """Everything but the first-pass fields is the off run's; and the off run's seal value
        recomputes from these rows once their decider blocks are taken out."""
        self.assertEqual(strip_first_pass(result), strip_first_pass(base))
        rows = [{k: v for k, v in r.items() if k != "decider"} for r in result["claim_evidence"]["outcomes"]]
        self.assertEqual(chain_value(rows), base["claim_evidence"]["verification"]["seal"]["value"])
        self.assertTrue(verify_seal(result["claim_evidence"]))

    def off(self):
        b, s = corpus()
        return judge(b, s, per_claim_model()[0])

    def test_missing_key_judges_everything_and_counts_the_fallbacks(self):
        first, fake = fp(table({}), api_key_env="CLAIM_SEAL_TEST_JEV_KEY_UNSET")
        os.environ.pop("CLAIM_SEAL_TEST_JEV_KEY_UNSET", None)
        b, s = corpus()
        result = judge(b, s, per_claim_model()[0], first_pass=first)
        rec = result["verification"]["first_pass"]
        self.assertEqual(fake.calls, [])
        self.assertEqual((rec["status"], rec["unavailable"], rec["fallbacks"]), ("unavailable", "missing-key", 4))
        self.assertEqual(rec["fallback_reasons"], {"missing-key": 4})
        self.assert_same_as_off(result, self.off())

    def test_each_call_failure_falls_back_to_the_judge(self):
        cases = {
            "timeout": JevError("timeout"),
            "http": JevError("http", "HTTP 503"),
            "transport": RuntimeError("socket closed"),
            "invalid-response": b'{"answers": {"relation": {"choice": "maybe", "confidence": 0.99}}}',
        }
        for kind, failure in cases.items():
            with self.subTest(kind=kind):
                first, _ = fp(lambda claim, section, failure=failure: failure)
                b, s = corpus()
                result = judge(b, s, per_claim_model()[0], first_pass=first)
                rec = result["verification"]["first_pass"]
                self.assertEqual(rec["fallback_reasons"], {kind: 4})
                self.assertEqual(rec["decided_by_jev"], 0)
                self.assertEqual(claims_of(result)[RELAY]["decided_by"]["reason"], "fallback:" + kind)
                self.assert_same_as_off(result, self.off())

    def test_a_reply_that_is_not_json_falls_back(self):
        first, _ = fp(lambda c, s: b"<html>busy</html>")
        b, s = corpus()
        result = judge(b, s, per_claim_model()[0], first_pass=first)
        self.assertEqual(result["verification"]["first_pass"]["fallback_reasons"], {"invalid-response": 4})

    def test_out_of_range_confidence_falls_back(self):
        first, _ = fp(lambda c, s: reply("supports", 1.5))
        b, s = corpus()
        result = judge(b, s, per_claim_model()[0], first_pass=first)
        self.assertEqual(result["verification"]["first_pass"]["decided_by_jev"], 0)

    def test_jev_failure_and_judge_failure_leave_claims_unchecked_never_supported(self):
        first, _ = fp(lambda c, s: JevError("timeout"))
        b, s = corpus()
        broken = helpers.Scripted(lambda prompt: "not json at all")
        result = judge(b, s, broken, first_pass=first)
        verdicts = {k: v["verdict"] for k, v in claims_of(result).items()}
        self.assertNotIn("supported", verdicts.values())
        self.assertEqual(result["status"], "verifier-error")

    def test_one_failed_source_sends_only_its_claim_to_the_judge(self):
        def rule(claim, section):
            if claim == LABS and "second lab" in section:
                return JevError("timeout")
            return ("supports", 0.95)
        first, _ = fp(rule)
        b, s = corpus()
        model, calls = per_claim_model()
        result = judge(b, s, model, first_pass=first)
        self.assertEqual(claims_of(result)[LABS]["decided_by"]["reason"], "fallback:timeout")
        self.assertEqual(set(calls), {LABS})
        self.assertEqual(result["verification"]["first_pass"]["fallbacks"], 1)


@patch.dict(os.environ, {KEY_ENV: KEY})
class TheRecord(unittest.TestCase):
    def result(self):
        first, fake = fp(table({RELAY: ("supports", 0.93), AUDIT: ("contradicts", 0.88)}))
        b, s = corpus()
        return judge(b, s, per_claim_model()[0], first_pass=first), fake

    def test_every_row_carries_a_decider_inside_the_seal(self):
        result, _ = self.result()
        rows = result["claim_evidence"]["outcomes"]
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row["decider"]["first_pass"], FIRST_PASS_CONTRACT)
            self.assertIn(row["decider"]["by"], ("jev", "judge", "none"))
        self.assertTrue(verify_seal(result["claim_evidence"]))

    def test_editing_the_decider_breaks_the_seal(self):
        result, _ = self.result()
        for edit in ("by", "confidence"):
            with self.subTest(edit=edit):
                tampered = copy.deepcopy(result["claim_evidence"])
                row = next(r for r in tampered["outcomes"] if r["decider"]["by"] == "judge"
                           and r["decider"]["answers"])
                if edit == "by":
                    row["decider"]["by"] = "jev"
                else:
                    row["decider"]["answers"][0]["confidence"] = 0.99
                self.assertFalse(verify_seal(tampered))

    def test_the_claim_records_choice_confidence_model_cost_and_latency(self):
        result, _ = self.result()
        answer = claims_of(result)[RELAY]["decided_by"]["answers"][0]
        self.assertEqual((answer["choice"], answer["confidence"], answer["model"], answer["cost_usd"]),
                         ("supports", 0.93, "typesafe/jev-test", 0.00002))
        self.assertIsInstance(answer["latency_s"], float)

    def test_the_run_record_counts_and_costs(self):
        result, fake = self.result()
        rec = result["verification"]["first_pass"]
        self.assertEqual(rec["calls"], len(fake.calls))
        self.assertEqual(rec["decided_by_jev"], 2)
        self.assertEqual(rec["passed_to_judge"], 2)
        self.assertAlmostEqual(rec["cost_usd"], 0.00002 * len(fake.calls))
        self.assertEqual(rec["models_returned"], ["typesafe/jev-test"])
        self.assertEqual(rec["settings"]["api_key_env"], KEY_ENV)

    def test_the_key_is_sent_in_one_header_and_never_written(self):
        result, fake = self.result()
        self.assertTrue(all(c["headers"] == {"Authorization": "Bearer " + KEY} for c in fake.calls))
        self.assertNotIn(KEY, json.dumps(result))
        self.assertTrue(all(KEY not in json.dumps(c["request"]) for c in fake.calls))

    def test_jev_sees_the_claim_and_the_judges_evidence_window(self):
        _, fake = self.result()
        req = next(c["request"] for c in fake.calls if c["request"]["state"]["claim"] == RELAY)
        self.assertIn("ten thousand messages an hour", req["state"]["section"])
        self.assertEqual(req["model"], "typesafe/jev-1.14-20261014")
        self.assertEqual(set(req["questions"]["relation"]["criteria"]), set(CHOICES))

    def test_not_checkable_claims_are_not_asked(self):
        _, fake = self.result()
        asked = {c["request"]["state"]["claim"] for c in fake.calls}
        self.assertNotIn("A vendor promises a faster planner next spring.", asked)


@patch.dict(os.environ, {KEY_ENV: KEY})
class Paths(unittest.TestCase):
    def test_whole_brief_path_numbers_only_the_claims_left_for_the_judge(self):
        first, _ = fp(table({RELAY: ("supports", 0.95), LABS: ("supports", 0.95)}))
        reply_obj = {"verdicts": [
            {"claim": 1, "verdict": "partial", "why": "the cited text names one duplicate record, not two"},
            {"claim": 2, "verdict": "partial", "why": "the trade-off is implied, not stated"}]}
        model = fixed(reply_obj)
        b, s = corpus()
        result = judge(b, s, model, sharded=False, first_pass=first)
        self.assertEqual(len(model.prompts), 1)
        self.assertNotIn(RELAY, model.prompts[0])
        self.assertIn("CLAIM 1:\n  " + AUDIT, model.prompts[0])
        self.assertIn("CLAIM 2:\n  " + TRADE, model.prompts[0])
        v = claims_of(result)
        self.assertEqual((v[AUDIT]["verdict"], v[TRADE]["verdict"]), ("partial", "partial"))
        self.assertEqual(v[RELAY]["decided_by"]["by"], "jev")

    def test_whole_brief_off_still_sends_every_claim(self):
        model = fixed(FROZEN_WHOLE_BRIEF_REPLY)
        b, s = corpus()
        judge(b, s, model, sharded=False)
        self.assertIn(RELAY, model.prompts[0])

    def test_when_jev_decides_everything_the_judge_is_not_called(self):
        first, _ = fp(lambda c, s: ("says_nothing", 0.99))
        model, calls = per_claim_model()
        b, s = corpus()
        result = judge(b, s, model, first_pass=first)
        self.assertEqual(calls, {})
        self.assertEqual(result["verification"]["status"], "ok")
        self.assertEqual(result["verification"]["attempts"], 0)
        self.assertTrue(verify_seal(result["claim_evidence"]))

    def test_decide_can_leave_supports_to_the_judge(self):
        first, _ = fp(lambda c, s: ("supports", 0.99), decide=("contradicts", "says_nothing"))
        model, calls = per_claim_model()
        b, s = corpus()
        result = judge(b, s, model, first_pass=first)
        self.assertEqual(claims_of(result)[RELAY]["decided_by"]["reason"], "class-not-deciding")
        self.assertEqual(result["verification"]["first_pass"]["decided_by_jev"], 0)

    def test_a_source_with_no_captured_body_is_not_put_to_jev(self):
        b, s = corpus()
        s["beta"]["items"][1]["body"] = None        # the audit claim's only source
        first, fake = fp(lambda c, s_: ("supports", 0.99))
        model, calls = per_claim_model()
        result = judge(b, s, model, first_pass=first)
        self.assertEqual(claims_of(result)[AUDIT]["decided_by"]["reason"], "no-window")
        self.assertNotIn(AUDIT, {c["request"]["state"]["claim"] for c in fake.calls})
        self.assertEqual(calls.get(AUDIT), 1)

    def test_evidence_off_is_refused(self):
        first, _ = fp(table({}))
        b, s = corpus()
        with self.assertRaises(ValueError):
            judge(b, s, per_claim_model()[0], evidence=False, first_pass=first)


class Settings(unittest.TestCase):
    def test_bad_settings_are_refused(self):
        for kw in ({"threshold": 0}, {"threshold": 1.5}, {"threshold": True}, {"threshold": "0.8"},
                   {"decide": ()}, {"decide": ("supports", "maybe")}, {"timeout_s": 0},
                   {"model": ""}, {"endpoint": "http://example.test/decisions"},
                   {"endpoint": "https://user:pw@example.test/decisions"}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                JevFirstPass(**kw)

    def test_from_env_is_off_unless_switched_on(self):
        self.assertIsNone(JevFirstPass.from_env({}))
        self.assertIsNone(JevFirstPass.from_env({"CLAIM_SEAL_JEV": "off", "CLAIM_SEAL_JEV_THRESHOLD": "0.9"}))
        got = JevFirstPass.from_env({"CLAIM_SEAL_JEV": "on", "CLAIM_SEAL_JEV_THRESHOLD": "0.9",
                                     "CLAIM_SEAL_JEV_MODEL": "typesafe/other",
                                     "CLAIM_SEAL_JEV_KEY_ENV": "CLAIM_SEAL_OTHER_KEY",
                                     "CLAIM_SEAL_JEV_TIMEOUT": "3",
                                     "CLAIM_SEAL_JEV_DECIDE": "contradicts, says_nothing"})
        self.assertEqual((got.threshold, got.model, got.api_key_env, got.timeout_s, got.decide),
                         (0.9, "typesafe/other", "CLAIM_SEAL_OTHER_KEY", 3.0, ("contradicts", "says_nothing")))

    def test_from_env_refuses_an_unreadable_value(self):
        for env in ({"CLAIM_SEAL_JEV": "maybe"}, {"CLAIM_SEAL_JEV": "on", "CLAIM_SEAL_JEV_THRESHOLD": "high"},
                    {"CLAIM_SEAL_JEV": "on", "CLAIM_SEAL_JEV_THRESHOLD": "1.2"}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                JevFirstPass.from_env(env)

    def test_first_pass_argument_shapes(self):
        self.assertIsNone(resolve_first_pass(None))
        self.assertIsNone(resolve_first_pass(False))
        with self.assertRaises(ValueError):
            resolve_first_pass(0.8)
        with self.assertRaises(ValueError):
            resolve_first_pass(True)

    def test_settings_never_hold_the_key(self):
        with patch.dict(os.environ, {KEY_ENV: KEY}):
            first = JevFirstPass(api_key_env=KEY_ENV)
            self.assertNotIn(KEY, json.dumps(first.settings()))
            self.assertNotIn(KEY, repr(vars(first)))


class _Server:
    def __init__(self, respond):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                respond(self)

            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/decisions"
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02},
                         daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def _send(code, body=b"", headers=None):
    def respond(handler):
        handler.send_response(code)
        for k, v in (headers or {}).items():
            handler.send_header(k, v)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
    return respond


@patch.dict(os.environ, {KEY_ENV: KEY})
class RealTransportOnLoopback(unittest.TestCase):
    def ask(self, respond, timeout_s=5):
        server = _Server(respond)
        try:
            first = JevFirstPass(endpoint=server.url, allow_insecure_loopback=True,
                                 api_key_env=KEY_ENV, timeout_s=timeout_s)
            return first.ask(KEY, "a claim", "a section")
        finally:
            server.close()

    def test_a_reply_is_read(self):
        got = self.ask(_send(200, reply("contradicts", 1)))
        self.assertEqual((got["choice"], got["confidence"], got["model"]), ("contradicts", 1.0, "typesafe/jev-test"))

    def test_http_error_redirect_and_timeout_are_named(self):
        cases = [("http", _send(503, b"busy")),
                 ("redirect", _send(302, b"", {"Location": "http://127.0.0.1:9/elsewhere"}))]
        for kind, respond in cases:
            with self.subTest(kind=kind), self.assertRaises(JevError) as caught:
                self.ask(respond)
            self.assertEqual(caught.exception.kind, kind)
        release = threading.Event()

        def slow(handler):
            release.wait(2)
        try:
            with self.assertRaises(JevError) as caught:
                self.ask(slow, timeout_s=0.2)
            self.assertEqual(caught.exception.kind, "timeout")
        finally:
            release.set()

    def test_plain_http_off_loopback_is_refused_before_any_call(self):
        with self.assertRaises(ValueError):
            JevFirstPass(endpoint="http://example.test/decisions", allow_insecure_loopback=True)


@patch.dict(os.environ, {KEY_ENV: KEY})
class Cli(unittest.TestCase):
    def test_jev_without_a_key_runs_and_reports_the_fallbacks(self):
        from test_adapters_cli_scan import ANSWERS, BRIEF, SOURCES, run_cli
        with patch.dict(os.environ, {"CLAIM_SEAL_JEV_KEY_ENV": "CLAIM_SEAL_TEST_NO_SUCH_KEY"}):
            os.environ.pop("CLAIM_SEAL_TEST_NO_SUCH_KEY", None)
            code, out = run_cli("check", BRIEF, SOURCES, "--judge", "replay:" + ANSWERS, "--jev")
        self.assertEqual(code, 0)
        self.assertIn("jev first pass: threshold 0.80, decided 0", out)
        self.assertIn('"missing-key": 4', out)
        self.assertIn("decided by: judge (fallback:missing-key)", out)

    def test_jev_with_no_evidence_or_a_bad_threshold_is_refused(self):
        from test_adapters_cli_scan import ANSWERS, BRIEF, SOURCES, run_cli
        import contextlib
        import io
        for extra in (["--jev", "--no-evidence"], ["--jev-threshold", "2"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                code, _ = run_cli("check", BRIEF, SOURCES, "--judge", "replay:" + ANSWERS, *extra)
                self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()

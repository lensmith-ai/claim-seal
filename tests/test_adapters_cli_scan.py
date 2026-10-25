"""Adapters (no network), the CLI end to end, the demo's golden output, and the private-string scan."""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import helpers
from helpers import ROOT

from claim_seal import judge, line_states, verify_seal
from claim_seal.adapters import AdapterError, OpenAICompatible, ReplayJudge
from claim_seal.cli import main as cli_main

EXAMPLES = os.path.join(ROOT, "examples")
BRIEF = os.path.join(EXAMPLES, "brief.json")
SOURCES = os.path.join(EXAMPLES, "sources.json")
ANSWERS = os.path.join(EXAMPLES, "replay-answers.json")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import scan_private  # noqa: E402


def run_cli(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cli_main(list(argv))
    return code, out.getvalue()


class ReplayJudgeTests(unittest.TestCase):
    def test_answers_both_prompt_shapes_and_omits_unknown_claims(self):
        model = ReplayJudge({"Claim A.": {"verdict": "supported", "why": "w"}})
        prompt = "intro\n\nCLAIM 1:\n  Claim B.\n  SOURCE 1 (x): t\n\nCLAIM 2:\n  Claim A.\n  SOURCE 1 (x): t"
        reply = json.loads(model(prompt))
        self.assertEqual(reply, {"verdicts": [{"verdict": "supported", "why": "w", "claim": 2}]})
        self.assertEqual(model(prompt).model, "replay-judge/v1")

    def test_a_claim_missing_from_the_table_comes_back_unchecked(self):
        brief, sources = helpers.corpus()
        result = judge(brief, sources, ReplayJudge({}))
        self.assertEqual(result["brief"]["items"][0]["verification"]["verdict"], "unchecked")
        self.assertEqual(result["status"], "verifier-error")


class OpenAIAdapterTests(unittest.TestCase):
    """Never touches the network: urlopen is replaced."""

    def test_refuses_without_a_model_name_or_a_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(AdapterError):
                OpenAICompatible()
            adapter = OpenAICompatible(model="m")
            with self.assertRaises(AdapterError) as ctx:
                adapter("prompt")
            self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    def test_builds_the_request_from_the_environment_and_reads_the_reply(self):
        seen = {}

        class Reply(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(req, timeout=None):
            seen.update(url=req.full_url, headers=dict(req.header_items()),
                        body=json.loads(req.data), timeout=timeout)
            return Reply(json.dumps({"model": "served-model-2",
                                     "choices": [{"message": {"content": '{"verdicts": []}'}}]}).encode())
        env = {"MY_KEY": "sk-test-not-real", "CLAIM_SEAL_BASE_URL": "https://llm.example.test/v1/"}
        with patch.dict(os.environ, env, clear=True), \
                patch("claim_seal.adapters.urllib.request.urlopen", fake_urlopen):
            adapter = OpenAICompatible(model="judge-model", api_key_env="MY_KEY")
            out = adapter("the prompt", timeout=12.5)
        self.assertEqual(out, '{"verdicts": []}')
        self.assertEqual(out.model, "served-model-2")
        self.assertEqual(seen["url"], "https://llm.example.test/v1/chat/completions")
        self.assertEqual(seen["headers"]["Authorization"], "Bearer sk-test-not-real")
        self.assertEqual(seen["body"]["messages"], [{"role": "user", "content": "the prompt"}])
        self.assertEqual(seen["body"]["model"], "judge-model")
        self.assertEqual(seen["timeout"], 12.5)
        self.assertNotIn("sk-test-not-real", json.dumps(vars(adapter)))

    def test_an_http_429_is_a_rate_limit_to_the_per_claim_runner(self):
        import urllib.error

        def fake_urlopen(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, None)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}, clear=True), \
                patch("claim_seal.adapters.urllib.request.urlopen", fake_urlopen):
            brief, sources = helpers.corpus()
            result = judge(brief, sources, OpenAICompatible(model="m"), evidence=False)
        reasons = {s["reason"] for s in result["verification"]["sharding"]["shards"]}
        self.assertEqual(reasons, {"rate-limit"})
        self.assertEqual(max(s["attempts"] for s in result["verification"]["sharding"]["shards"]), 1)


class DemoTests(unittest.TestCase):
    def test_the_demo_output_matches_its_golden_file(self):
        code, out = run_cli("check", BRIEF, SOURCES, "--judge", "replay:" + ANSWERS)
        self.assertEqual(code, 0)
        with open(os.path.join(EXAMPLES, "demo-output.txt"), encoding="utf-8") as handle:
            self.assertEqual(out, handle.read())

    def test_the_demo_has_every_kind_of_claim(self):
        with open(BRIEF, encoding="utf-8") as h:
            brief = json.load(h)
        with open(SOURCES, encoding="utf-8") as h:
            sources = json.load(h)
        result = judge(brief, sources, ReplayJudge.from_file(ANSWERS))
        verdicts = [i["verification"]["verdict"] for i in result["brief"]["items"]]
        self.assertEqual(verdicts, ["supported", "partial", "unsupported", "unresolved", "partial"])
        flags = [(r["item"], r["flag"]) for r in result["attribution"] if r["flag"]]
        self.assertEqual(flags, [(5, "overclaim")])
        relations = {c["span"].get("sentence_relation") for i in result["brief"]["items"]
                     for c in i["verification"]["citations"] if c.get("span")}
        self.assertTrue({"says", "implies", "contradicts"} <= relations)
        states = line_states(result["brief"])
        self.assertIn("reading", {l["state"] for l in states["lines"]})
        self.assertTrue(states["left_out"])
        self.assertTrue(verify_seal(result["claim_evidence"]))

    def test_check_writes_a_result_that_verify_seal_accepts_and_tampering_breaks(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "result.json")
            code, _ = run_cli("check", BRIEF, SOURCES, "--judge", "replay:" + ANSWERS, "--out", out)
            self.assertEqual(code, 0)
            code, text = run_cli("verify-seal", out)
            self.assertEqual(code, 0, text)
            self.assertIn("OK", text)
            for how in ("upgrade", "append", "relabel", "edit"):
                bad = os.path.join(tmp, f"{how}.json")
                subprocess.run([sys.executable, os.path.join(EXAMPLES, "tamper.py"), out, bad, how],
                               check=True, capture_output=True)
                code, text = run_cli("verify-seal", bad)
                self.assertEqual(code, 1, how)
                self.assertIn("CHAIN FAIL", text)

    def test_verify_seal_names_its_scope_and_does_not_cover_verdicts_or_sentence_labels(self):
        """Documented limit: only claim_evidence.outcomes is sealed. Upgrading a claim verdict or a
        sentence relation in the brief leaves the seal valid, and the OK line says so."""
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "result.json")
            run_cli("check", BRIEF, SOURCES, "--judge", "replay:" + ANSWERS, "--out", out)
            with open(out, encoding="utf-8") as h:
                doc = json.load(h)
            doc["brief"]["items"][2]["verification"]["verdict"] = "supported"
            doc["brief"]["items"][1]["verification"]["citations"][1]["span"]["sentence_relation"] = "says"
            doc["claim_evidence"]["verification"]["rule_ids"] = []
            doc["brief"]["items"][0]["insight"] = "Edited claim text: the row's claim digest is not re-checked."
            with open(out, "w", encoding="utf-8") as h:
                json.dump(doc, h)
            code, text = run_cli("verify-seal", out)
        self.assertEqual(code, 0)
        self.assertIn("claim_evidence.outcomes: 7 rows recompute", text)
        self.assertIn("(outcome, span, verb, judge-chose flag, binding)", text)
        self.assertIn("Not sealed: claim verdicts, per-sentence relations and goes-further labels, rule ids", text)

    def test_a_sealed_record_with_its_rows_removed_or_nulled_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = os.path.join(tmp, "good.json")
            run_cli("check", BRIEF, SOURCES, "--judge", "replay:" + ANSWERS, "--out", good)
            with open(good, encoding="utf-8") as h:
                doc = json.load(h)
            emptied = json.loads(json.dumps(doc))
            emptied["claim_evidence"]["outcomes"] = []
            dropped = json.loads(json.dumps(doc))
            del dropped["claim_evidence"]["outcomes"]
            nulled = dict(doc, claim_evidence=None)
            bare = json.loads(json.dumps(doc["claim_evidence"]))
            bare["outcomes"] = []
            for name, variant in (("emptied", emptied), ("dropped", dropped), ("nulled", nulled),
                                  ("bare record", bare)):
                path = os.path.join(tmp, name + ".json")
                with open(path, "w", encoding="utf-8") as h:
                    json.dump(variant, h)
                code, text = run_cli("verify-seal", path)
                self.assertEqual(code, 1, name)
                self.assertIn("CHAIN FAIL", text, name)
                code, text = run_cli("verify-seal", good, path)
                self.assertEqual(code, 1, name + " beside a good file")

    def test_verify_seal_with_nothing_to_check_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = os.path.join(tmp, "empty.json")
            with open(empty, "w") as h:
                json.dump({"outcomes": []}, h)
            self.assertEqual(run_cli("verify-seal", empty)[0], 2)

    def test_check_without_a_judge_runs_pass_one_only(self):
        code, out = run_cli("check", BRIEF, SOURCES)
        self.assertEqual(code, 0)
        self.assertIn("pass 1 only", out)
        self.assertIn("[item 4] unresolved", out)
        self.assertNotIn("supported", out.replace("not for support", ""))

    def test_check_refuses_bad_sources_with_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "bad.json")
            with open(bad, "w") as h:
                json.dump({"s": [{"title": "no text"}]}, h)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(run_cli("check", BRIEF, bad, "--judge", "replay:" + ANSWERS)[0], 2)


class PrivateStringScanTests(unittest.TestCase):
    """Arms use neutral planted strings only. Project-specific terms live in a deny-list file
    outside the repository (CLAIM_SEAL_DENYLIST), never in this test."""

    # Split only so this file stays clean under its own scan; the strings are neutral.
    PLANTED = {
        "absolute server path": "/op" + "t/example-app/config",
        "home directory path": "/ho" + "me/someone/notes.txt",
        "api key shape": "sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2",
        "token shape": "ghp_" + "abcdefghijklmnop1234",
        "cloud key shape": "AKIA" + "ABCDEFGHIJKLMNOP",
        "bot token shape": "123456789:" + "A" * 35,
        "private key block": "-----BEGIN " + "RSA PRIVATE KEY-----",
        "job id shape": "12345678T123456Z-" + "abcdef",
        "run id shape": "1234-56-78-" + "9012",
        "personal email": "someone" + "@mail.invalid",
        "unexpected environment variable": "os.environ.get(" + '"OTHER_TOOL_FLAG")',
    }

    def test_the_repository_is_clean(self):
        hits, scanned = scan_private.scan(ROOT)
        self.assertEqual(hits, [])
        self.assertGreater(scanned, 20)

    def test_every_generic_pattern_has_an_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            for n, text in enumerate(self.PLANTED.values()):
                with open(os.path.join(tmp, f"f{n}.py"), "w") as h:
                    h.write("x = " + repr(text) + "\n")
            hits, _ = scan_private.scan(tmp)
        self.assertEqual(sorted({h[2] for h in hits}), sorted(self.PLANTED))
        self.assertEqual(len({h[0] for h in hits}), len(self.PLANTED))

    def test_allowed_names_and_addresses_do_not_trip_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "ok.py"), "w") as h:
                h.write('os.environ.get("OPENAI_API_KEY"); os.environ["CLAIM_SEAL_MODEL"]\n'
                        "# noreply@anthropic.com, someone@example.org, https://example.org/a/opt/b\n"
                        "key = 'sk-test-not-real'\n")
            self.assertEqual(scan_private.scan(tmp)[0], [])

    def test_a_binary_passes_only_when_listed_with_its_exact_sha256(self):
        png = b"\x89PNG\r\n\x1a\n" + bytes(range(256))
        digest = __import__("hashlib").sha256(png).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "docs"))
            with open(os.path.join(tmp, "docs", "fig.png"), "wb") as h:
                h.write(png)
            unlisted, _ = scan_private.scan(tmp)
            self.assertEqual([(h[0], h[2]) for h in unlisted], [("docs/fig.png", "unscannable file")])
            with patch.dict(scan_private.REVIEWED_BINARIES, {"docs/fig.png": digest}, clear=True):
                self.assertEqual(scan_private.scan(tmp)[0], [])
            with patch.dict(scan_private.REVIEWED_BINARIES, {"docs/fig.png": "0" * 64}, clear=True):
                changed, _ = scan_private.scan(tmp)
                self.assertEqual([(h[0], h[2]) for h in changed], [("docs/fig.png", "reviewed binary changed")])

    def test_an_extra_denylist_file_adds_patterns_with_per_file_allowances(self):
        with tempfile.TemporaryDirectory() as tmp:
            deny = os.path.join(tmp, "deny.txt")
            with open(deny, "w") as h:
                h.write("# comment\n\nquokka-[0-9]+\nzebra\tNOTICE,docs/ok.md\n")
            repo = os.path.join(tmp, "repo")
            os.makedirs(os.path.join(repo, "docs"))
            for rel, text in (("a.py", "QUOKKA-42"), ("NOTICE", "zebra"), ("docs/ok.md", "Zebra"),
                              ("b.md", "a zebra crossing")):
                with open(os.path.join(repo, rel), "w") as h:
                    h.write(text + "\n")
            extra = scan_private.load_denylist(deny)
            hits, _ = scan_private.scan(repo, extra)
            self.assertEqual(sorted((h[0], h[3]) for h in hits), [("a.py", "QUOKKA-42"), ("b.md", "zebra")])
            self.assertEqual(scan_private.scan(repo)[0], [])
            with patch.dict(os.environ, {scan_private.DENYLIST_ENV: deny}), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(scan_private.main(["scan", repo]), 1)
            self.assertEqual(scan_private.load_denylist(None), [])

    def test_history_mode_sees_a_term_that_only_an_old_commit_contained(self):
        with tempfile.TemporaryDirectory() as tmp:
            git = lambda *a: subprocess.run(["git", "-C", tmp, *a], check=True, capture_output=True)  # noqa: E731
            git("init", "-q")
            git("config", "user.name", "Test")
            git("config", "user.email", "test@example.org")
            with open(os.path.join(tmp, "a.txt"), "w") as h:
                h.write("quokka-7\n")
            git("add", "a.txt")
            git("commit", "-qm", "one")
            with open(os.path.join(tmp, "a.txt"), "w") as h:
                h.write("clean\n")
            git("commit", "-qam", "two")
            extra = [("deny", __import__("re").compile("quokka-[0-9]", 2), set())]
            self.assertEqual(scan_private.scan(tmp, extra)[0], [])
            hits, lines = scan_private.scan_history(tmp, extra)
            # seen where the first commit added it and where the second removed it
            self.assertEqual([(h[0].split(":", 1)[1], h[3]) for h in hits], [("a.txt", "quokka-7")] * 2)
            self.assertEqual(len({h[0] for h in hits}), 2)
            self.assertGreater(lines, 0)

    def test_binary_files_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "blob.bin"), "wb") as h:
                h.write(b"\xff\xfe\x00")
            hits, _ = scan_private.scan(tmp)
        self.assertEqual([(h[0], h[2]) for h in hits], [("blob.bin", "unscannable file")])


if __name__ == "__main__":
    unittest.main()

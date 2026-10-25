"""The public-release security review's findings, one class per finding, each built from the
review's own probe. Every test here failed on 88ca3ea (the reviewed revision) before its fix.

No test touches the network: endpoints are loopback servers on 127.0.0.1 or mocked openers, and
the only key is a planted test value.
"""
from __future__ import annotations

import http.server
import json
import os
import threading
import unittest
from unittest.mock import patch

import helpers  # noqa: F401 - puts the repository root on sys.path

from claim_seal.adapters import AdapterError, OpenAICompatible

TEST_KEY = "TEST_ONLY_KEY"


class _RecordingEnviron(dict):
    """An os.environ stand-in that records every name looked up."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.looked_up: list[str] = []

    def get(self, key, default=None):
        self.looked_up.append(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.looked_up.append(key)
        return super().__getitem__(key)


class Finding1HttpEndpointRefused(unittest.TestCase):
    """HIGH 1: an http CLAIM_SEAL_BASE_URL sent the bearer key in cleartext."""

    def test_an_http_base_url_from_the_environment_is_refused_before_any_key_is_read(self):
        env = _RecordingEnviron(OPENAI_API_KEY=TEST_KEY, CLAIM_SEAL_BASE_URL="http://example.test/v1")
        with patch.object(os, "environ", env):
            with self.assertRaises(AdapterError) as ctx:
                OpenAICompatible(model="m")
        self.assertIn("https", str(ctx.exception))
        self.assertNotIn(TEST_KEY, str(ctx.exception))
        self.assertNotIn("OPENAI_API_KEY", env.looked_up)

    def test_an_http_base_url_passed_in_code_is_refused(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": TEST_KEY}, clear=True):
            for url in ("http://example.test/v1", "HTTP://example.test/v1", "ftp://example.test/v1",
                        "file:///etc/passwd", "example.test/v1", "https:///v1",
                        "https://user:secret@example.test/v1"):
                with self.subTest(url=url), self.assertRaises(AdapterError):
                    OpenAICompatible(model="m", base_url=url)

    def test_https_is_accepted(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": TEST_KEY}, clear=True):
            adapter = OpenAICompatible(model="m", base_url="https://example.test/v1/")
            self.assertEqual(adapter.request("p").full_url, "https://example.test/v1/chat/completions")
            self.assertEqual(OpenAICompatible(model="m").base_url, "https://api.openai.com/v1")

    def test_loopback_http_needs_the_constructor_opt_in_and_the_environment_cannot_give_it(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": TEST_KEY,
                                     "CLAIM_SEAL_BASE_URL": "http://127.0.0.1:9/v1"}, clear=True):
            with self.assertRaises(AdapterError):
                OpenAICompatible(model="m")
            adapter = OpenAICompatible(model="m", allow_insecure_loopback=True)
            self.assertEqual(adapter.base_url, "http://127.0.0.1:9/v1")
            for url in ("http://example.test/v1", "http://10.0.0.1/v1", "http://localhost.example.test/v1"):
                with self.subTest(url=url), self.assertRaises(AdapterError):
                    OpenAICompatible(model="m", base_url=url, allow_insecure_loopback=True)


class _Server:
    """A loopback HTTP server on its own port (its own origin) that records every request and
    answers with `respond(handler)`."""

    def __init__(self, respond):
        seen = self.seen = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def _any(self):
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                seen.append((self.command, self.headers.get("Authorization"), self.path))
                respond(self)

            do_GET = do_POST = _any

            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02},
                                       daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def _redirect_to(location, code=302):
    def respond(handler):
        handler.send_response(code)
        handler.send_header("Location", location)
        handler.send_header("Content-Length", "0")
        handler.end_headers()
    return respond


def _reply_ok(handler):
    body = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class Finding2NoAuthenticatedRedirects(unittest.TestCase):
    """HIGH 2: a 302 from the endpoint forwarded Authorization to another origin."""

    def setUp(self):
        self.target = _Server(_reply_ok)
        self.addCleanup(self.target.close)

    def _call_through(self, first_respond):
        first = _Server(first_respond)
        self.addCleanup(first.close)
        with patch.dict(os.environ, {"OPENAI_API_KEY": TEST_KEY}, clear=True):
            adapter = OpenAICompatible(model="m", base_url=first.url + "/v1",
                                       allow_insecure_loopback=True)
            with self.assertRaises(AdapterError) as ctx:
                adapter("prompt", timeout=10)
        return first, ctx.exception

    def test_a_cross_origin_redirect_is_refused_and_the_key_never_reaches_the_target(self):
        # The review's probe: two loopback servers on different ports; the first answers 302.
        first, exc = self._call_through(_redirect_to(self.target.url + "/different-origin"))
        self.assertEqual(self.target.seen, [])
        self.assertEqual(first.seen, [("POST", "Bearer " + TEST_KEY, "/v1/chat/completions")])
        self.assertIn("redirect", str(exc))
        self.assertNotIn(TEST_KEY, str(exc))

    def test_a_same_origin_redirect_is_refused_too(self):
        holder = {}

        def respond(handler):
            _redirect_to(holder["url"] + "/elsewhere", 307)(handler)
        first = _Server(respond)
        self.addCleanup(first.close)
        holder["url"] = first.url
        with patch.dict(os.environ, {"OPENAI_API_KEY": TEST_KEY}, clear=True):
            adapter = OpenAICompatible(model="m", base_url=first.url + "/v1", allow_insecure_loopback=True)
            with self.assertRaises(AdapterError):
                adapter("prompt", timeout=10)
        self.assertEqual([p for _, _, p in first.seen], ["/v1/chat/completions"])

    def test_every_redirect_code_is_refused_without_a_follow_up_request(self):
        for code in (301, 302, 303, 307, 308):
            with self.subTest(code=code):
                self.target.seen.clear()
                self._call_through(_redirect_to(self.target.url + "/x", code))
                self.assertEqual(self.target.seen, [])

    def test_an_https_to_http_redirect_is_refused_by_the_handler_itself(self):
        import urllib.request
        from claim_seal import adapters
        with patch.dict(os.environ, {"OPENAI_API_KEY": TEST_KEY}, clear=True):
            req = OpenAICompatible(model="m", base_url="https://example.test/v1").request("p")
        handler = adapters._RefuseRedirects()
        with self.assertRaises(AdapterError):
            handler.redirect_request(req, None, 302, "Found", {}, "http://example.test/steal")
        # Defense in depth: the key is an unredirected header, which urllib never copies onto a
        # redirected request even if some other handler followed one.
        self.assertIn("Authorization", req.unredirected_hdrs)
        self.assertNotIn("Authorization", req.headers)
        follow = urllib.request.HTTPRedirectHandler().redirect_request(
            req, None, 302, "Found", {}, "http://example.test/steal")
        self.assertNotIn("Authorization", dict(follow.header_items()))


if __name__ == "__main__":
    unittest.main()

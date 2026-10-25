"""The public-release security review's findings, one class per finding, each built from the
review's own probe. Every test here failed on 88ca3ea (the reviewed revision) before its fix.

No test touches the network: endpoints are loopback servers on 127.0.0.1 or mocked openers, and
the only key is a planted test value.
"""
from __future__ import annotations

import os
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


if __name__ == "__main__":
    unittest.main()

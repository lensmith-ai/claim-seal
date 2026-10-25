"""Model adapters for pass 2. A model is any callable: prompt (str) -> reply text (str).

ReplayJudge          deterministic and offline: answers from a table of recorded verdicts keyed
                     by claim text. It does NOT judge anything; it replays what you give it.
                     Used by the tests and the demo.
OpenAICompatible     one chat-completions call per prompt to an OpenAI-compatible endpoint.
                     The API key is read from an environment variable at call time and never
                     stored, logged or written into a receipt.

Both return ModelText, a str that also carries the name of the model that answered.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

from .prompts import SOURCE_DATA_RULE, strip_source_data
from .support import ModelText

_CLAIM_BLOCK = re.compile(r"^CLAIM (\d+):\n  (.*)$", re.M)


class ReplayJudge:
    """Replays recorded judge answers. `answers` maps a claim's exact text to the verdict
    object the judge would return for it (verdict, why, and any of relation, strength,
    citation_spans, evidence_spans, citation_bears).

    Works for both prompt shapes: it finds every "CLAIM n:" block, looks up that claim's text,
    and returns {"verdicts": [...]} numbered as the prompt numbered them. A claim missing from
    the table is left out of the reply, so it comes back unchecked, exactly as a real judge's
    omission would. Completion prompts get `completion_answers` keyed by claim text, if given.
    """

    name = "replay-judge/v1"

    def __init__(self, answers: dict, completion_answers: dict | None = None):
        self.answers = dict(answers or {})
        self.completion_answers = dict(completion_answers or {})
        self.prompts: list[str] = []

    @classmethod
    def from_file(cls, path: str) -> "ReplayJudge":
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
        return cls(doc.get("answers") or {}, doc.get("completion_answers") or {})

    def __call__(self, prompt: str, timeout: float | None = None) -> ModelText:
        self.prompts.append(prompt)
        # Only the instruction part is read: a "CLAIM n:" line inside source data is source text.
        instructions = strip_source_data(prompt)
        if "Judge only the delimited fetched excerpts." in instructions:
            claim = instructions.split("CLAIM:\n", 1)[-1].split("\n", 1)[0]
            return ModelText(json.dumps(self.completion_answers.get(claim) or {}), model=self.name)
        verdicts = []
        for number, text in _CLAIM_BLOCK.findall(instructions):
            entry = self.answers.get(text)
            if isinstance(entry, dict):
                verdicts.append(dict(entry, claim=int(number)))
        return ModelText(json.dumps({"verdicts": verdicts}), model=self.name)


class AdapterError(RuntimeError):
    """The endpoint could not be called, or answered with an HTTP error."""


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _checked_base_url(url: str, *, allow_insecure_loopback: bool) -> str:
    """The endpoint the bearer key may be sent to, or an AdapterError. The URL is named in the
    error only by its scheme and host, so a mistyped key pasted into it is never echoed."""
    try:
        parts = urllib.parse.urlsplit(url)
        host = parts.hostname or ""
        parts.port  # noqa: B018 - raises ValueError on a malformed port
    except ValueError:
        raise AdapterError("the model endpoint URL could not be parsed") from None
    scheme = parts.scheme.lower()
    if parts.username is not None or parts.password is not None:
        raise AdapterError("the model endpoint URL must not carry a user name or password")
    if not host:
        raise AdapterError("the model endpoint URL has no host; use https://HOST/PATH")
    if scheme == "https":
        return url
    if scheme == "http" and allow_insecure_loopback and host.lower() in _LOOPBACK_HOSTS:
        return url
    raise AdapterError(f"refused model endpoint {scheme or '(no scheme)'}://{host}: the API key is "
                       "only sent over https (plain http is allowed only to a loopback address, "
                       "and only with OpenAICompatible(allow_insecure_loopback=True))")


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Every redirect on an authenticated call is refused, same origin or not: following one
    would hand the bearer key to wherever the endpoint (or anyone who controls its responses)
    points, including plain http."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise AdapterError(f"the model endpoint answered with a redirect (HTTP {code}); "
                           "claim-seal never follows redirects on calls that carry the API key. "
                           "Set the final endpoint URL directly.")


_OPENER = urllib.request.build_opener(_RefuseRedirects())


def _open(req: urllib.request.Request, timeout: float):
    """The only network call in claim-seal: one request, redirects refused."""
    return _OPENER.open(req, timeout=timeout)


class OpenAICompatible:
    """POST {base_url}/chat/completions with one user message; return the first choice's text.

    model      model name sent to the endpoint (default: $CLAIM_SEAL_MODEL)
    base_url   default: $CLAIM_SEAL_BASE_URL, else https://api.openai.com/v1. Must be https
               with a host and no user:password part; anything else is refused here, before
               the key is ever read.
    api_key_env  name of the environment variable holding the key (default OPENAI_API_KEY)
    allow_insecure_loopback  constructor-only switch for local testing: also accept
               http://127.0.0.1, http://[::1] or http://localhost. No environment variable can
               turn it on, and it never allows plain http to any other host.

    send_rule  also send claim_seal.prompts.SOURCE_DATA_RULE as a system message (default
               True; every prompt already opens with it, this adds the instruction channel)
    json_mode  ask for a JSON object reply (response_format json_object; default True). An
               endpoint that rejects it answers HTTP 400: pass json_mode=False.

    An HTTP 429 raises an AdapterError whose message contains "429", which the per-claim
    runner treats as a rate limit (fails closed, no retry).
    """

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", temperature: float = 0.0, *,
                 allow_insecure_loopback: bool = False, send_rule: bool = True,
                 json_mode: bool = True):
        self.model = model or os.environ.get("CLAIM_SEAL_MODEL") or ""
        self.base_url = _checked_base_url(
            (base_url or os.environ.get("CLAIM_SEAL_BASE_URL") or "https://api.openai.com/v1").rstrip("/"),
            allow_insecure_loopback=allow_insecure_loopback is True)
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.send_rule = send_rule
        self.json_mode = json_mode
        self.name = self.model or None
        if not self.model:
            raise AdapterError("no model name: pass model= or set CLAIM_SEAL_MODEL")

    def request(self, prompt: str) -> urllib.request.Request:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise AdapterError(f"no API key: set the {self.api_key_env} environment variable")
        messages = ([{"role": "system", "content": SOURCE_DATA_RULE.strip()}] if self.send_rule else [])
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "temperature": self.temperature, "messages": messages}
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=body, method="POST",
            headers={"Content-Type": "application/json"})
        # Unredirected: urllib never copies it onto a redirected request. Redirects are refused
        # outright anyway (_RefuseRedirects); this is the second lock on the same door.
        req.add_unredirected_header("Authorization", "Bearer " + key)
        return req

    def __call__(self, prompt: str, timeout: float | None = None) -> ModelText:
        req = self.request(prompt)
        try:
            with _open(req, timeout or 180) as resp:
                doc = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            hint = (" (if the endpoint does not accept response_format, use "
                    "OpenAICompatible(json_mode=False))" if exc.code == 400 and self.json_mode else "")
            raise AdapterError(f"HTTP {exc.code} from the model endpoint{hint}") from None
        except urllib.error.URLError as exc:
            raise AdapterError(f"could not reach the model endpoint: {exc.reason}") from None
        try:
            text = doc["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise AdapterError("the endpoint's reply had no choices[0].message.content") from None
        return ModelText(text, model=doc.get("model") or None)

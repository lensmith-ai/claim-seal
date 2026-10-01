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
import urllib.request

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
        if prompt.startswith("Judge only the delimited fetched excerpts."):
            claim = prompt.split("CLAIM:\n", 1)[-1].split("\n", 1)[0]
            return ModelText(json.dumps(self.completion_answers.get(claim) or {}), model=self.name)
        verdicts = []
        for number, text in _CLAIM_BLOCK.findall(prompt):
            entry = self.answers.get(text)
            if isinstance(entry, dict):
                verdicts.append(dict(entry, claim=int(number)))
        return ModelText(json.dumps({"verdicts": verdicts}), model=self.name)


class AdapterError(RuntimeError):
    """The endpoint could not be called, or answered with an HTTP error."""


class OpenAICompatible:
    """POST {base_url}/chat/completions with one user message; return the first choice's text.

    model      model name sent to the endpoint (default: $CLAIM_SEAL_MODEL)
    base_url   default: $CLAIM_SEAL_BASE_URL, else https://api.openai.com/v1
    api_key_env  name of the environment variable holding the key (default OPENAI_API_KEY)

    An HTTP 429 raises an AdapterError whose message contains "429", which the per-claim
    runner treats as a rate limit (fails closed, no retry).
    """

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", temperature: float = 0.0):
        self.model = model or os.environ.get("CLAIM_SEAL_MODEL") or ""
        self.base_url = (base_url or os.environ.get("CLAIM_SEAL_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.name = self.model or None
        if not self.model:
            raise AdapterError("no model name: pass model= or set CLAIM_SEAL_MODEL")

    def request(self, prompt: str) -> urllib.request.Request:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise AdapterError(f"no API key: set the {self.api_key_env} environment variable")
        body = json.dumps({"model": self.model, "temperature": self.temperature,
                           "messages": [{"role": "user", "content": prompt}]}).encode("utf-8")
        return urllib.request.Request(
            self.base_url + "/chat/completions", data=body, method="POST",
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})

    def __call__(self, prompt: str, timeout: float | None = None) -> ModelText:
        req = self.request(prompt)
        try:
            with urllib.request.urlopen(req, timeout=timeout or 180) as resp:
                doc = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise AdapterError(f"HTTP {exc.code} from the model endpoint") from None
        except urllib.error.URLError as exc:
            raise AdapterError(f"could not reach the model endpoint: {exc.reason}") from None
        try:
            text = doc["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise AdapterError("the endpoint's reply had no choices[0].message.content") from None
        return ModelText(text, model=doc.get("model") or None)

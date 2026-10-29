"""Optional Jev first pass for pass 2. OFF unless a caller turns it on; SHADOW when turned on.

With a first pass, each checkable claim is first put to Jev, a small classifier served on
OpenRouter's decisions endpoint, once per cited source: does this source section support the
claim, contradict it, or say nothing about it? Jev answers with one of the three and a
confidence.

SHADOW (the default once on, and the recommended use): Jev's answers are recorded beside the
judge's, the judge decides every claim exactly as without a first pass, and the run record counts
where Jev would have decided and whether it agreed. Jev has no "partial" answer; on real composed
briefs, where the judge calls most claims partial, Jev said "supports" to many of them.

DECIDING ("cascade", opt in with decide=...): when every cited source of a claim is answered at or
above the threshold, the answers do not disagree with each other, and the claim-level answer is in
`decide`, Jev's answer decides the claim and the judge is not called for it. Every other claim
goes to the judge, exactly as without a first pass.

    from claim_seal import judge
    from claim_seal.jev import JevFirstPass
    result = judge(brief, sources, model, first_pass=JevFirstPass())              # shadow
    result = judge(brief, sources, model, first_pass=JevFirstPass(decide=("supports",)))
    result = judge(brief, sources, model, first_pass="env")   # on only if CLAIM_SEAL_JEV=on

WHAT JEV IS SHOWN. The claim text and, per citation, the same evidence window the judge must quote
from: a slice of the digest-pinned, normalized source document (claim_seal.evidence). A claim with
a resolved citation that has no window (no captured body, a drifted digest, too many citations for
the window budget) is not put to Jev; it goes to the judge.

THE MAPPING to claim-seal's words:
  supports      -> verdict "supported". Jev quotes nothing, so the support passage recorded for
                   that citation is the whole window Jev read, located in the source like any
                   judge quote. It is marked in the sealed row (decider.section = "evidence-window")
                   so nobody mistakes it for a quoted sentence. The claim's relation is
                   "undetermined": Jev does not say whether the source states the claim or implies
                   it. Jev has no "partial": a claim the judge would call partial can come back
                   "supported" from Jev.
  contradicts   -> verdict "unsupported", relation "contradicts".
  says_nothing  -> verdict "unsupported", relation "undetermined".
Several citations: all says_nothing -> unsupported; contradicts with no supports -> unsupported
(contradicts); supports with no contradicts -> supported; supports AND contradicts -> the judge.

FAILURE. A missing key, an HTTP error, a timeout, a redirect, an oversized or malformed reply: that
claim goes to the normal judge, and the fallback is counted by reason in the run record. A Jev
failure never fails a claim and never marks one supported.

THE RECORD. Every claim checked with a first pass on carries `decided_by` in its verification
block (path, reason, threshold, Jev's claim-level answer when it was confident, per-citation
choice, confidence, resolved model, cost and latency, and claim-seal's own reading), and every
claim-evidence row carries a `decider` block inside the seal (path, reason, mode, threshold, Jev's
claim-level answer, and per citation the choice, confidence and resolved model). The run record
gains `first_pass`: mode, settings (never the key), counts, pass-up and fallback reasons, total
returned cost, latency, and the comparison with the judge on the claims the judge decided.

THE KEY is read from the environment variable named by api_key_env (default OPENROUTER_API_KEY)
once per check, sent only over https (redirects refused, as for OpenAICompatible), and never
stored, logged or written into a result.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import time
import urllib.error
import urllib.request

from . import limits

FIRST_PASS_CONTRACT = "jev-first-pass/v1"
DEFAULT_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
# The snapshot the published numbers were measured on. "~typesafe/jev-latest" is a moving alias.
DEFAULT_MODEL = "typesafe/jev-1.14-20261014"
DEFAULT_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_THRESHOLD = 0.80
DEFAULT_TIMEOUT_S = 10.0
DEFAULT_POOL = 4
# Which Jev answers may decide a claim. Empty = shadow: Jev decides nothing, its answers are kept.
DEFAULT_DECIDE: tuple = ()
# Sent with every request: no fallback to another provider, no data collection, zero retention,
# and a price ceiling (USD per million tokens) the provider must stay under.
DEFAULT_PROVIDER = {"allow_fallbacks": False, "data_collection": "deny", "zdr": True,
                    "max_price": {"prompt": "10.0", "completion": "0"}}

CHOICES = ("supports", "contradicts", "says_nothing")
# The question, word for word as it was measured. Changing it changes the answers.
QUESTION_ID = "relation"
QUESTION = {
    "type": "choice",
    "instructions": "How does the evidence section relate to the claim?",
    "criteria": {
        "supports": "The evidence section states the claim or directly implies that it is true.",
        "contradicts": "The evidence section states the opposite of the claim or implies it is false.",
        "says_nothing": "The evidence section does not address what the claim asserts, either way.",
    },
}
# Jev choice -> (claim-seal verdict, claim relation)
VERDICT_BY_CHOICE = {"supports": ("supported", "undetermined"),
                     "contradicts": ("unsupported", "contradicts"),
                     "says_nothing": ("unsupported", "undetermined")}

ENV_SWITCH = "CLAIM_SEAL_JEV"
_ON_WORDS = frozenset({"1", "on", "true", "yes"})
_OFF_WORDS = frozenset({"", "0", "off", "false", "no"})


class JevError(RuntimeError):
    """One first-pass call failed. `kind` is the counted reason: missing-key, http, timeout,
    transport, redirect, too-large, invalid-response."""

    def __init__(self, kind: str, message: str = ""):
        super().__init__(message or kind)
        self.kind = kind


def _check_threshold(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"the Jev threshold must be a number, got {type(value).__name__}")
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"the Jev threshold must be above 0 and at most 1, got {value}")
    return value


def _https_transport(endpoint: str, body: bytes, headers: dict, timeout: float) -> bytes:
    """POST once, redirects refused (the same opener as OpenAICompatible), reply size bounded."""
    from .adapters import AdapterError, _open  # local: adapters imports support

    req = urllib.request.Request(endpoint, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    for name, value in headers.items():
        req.add_unredirected_header(name, value)
    try:
        with _open(req, timeout) as resp:
            raw = resp.read(limits.MAX_MODEL_RESPONSE_BYTES + 1)
    except AdapterError:
        raise JevError("redirect", "the endpoint answered with a redirect") from None
    except urllib.error.HTTPError as exc:
        raise JevError("http", f"HTTP {exc.code}") from None
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise JevError("timeout", "timed out") from None
        raise JevError("transport", f"could not reach the endpoint: {type(exc.reason).__name__}") from None
    except TimeoutError:
        raise JevError("timeout", "timed out") from None
    except OSError as exc:
        raise JevError("transport", type(exc).__name__) from None
    if len(raw) > limits.MAX_MODEL_RESPONSE_BYTES:
        raise JevError("too-large", f"reply larger than {limits.MAX_MODEL_RESPONSE_BYTES} bytes")
    return raw


class JevFirstPass:
    """Settings for the first pass. Pass an instance to judge(first_pass=...).

    threshold      a Jev answer decides only at or above this confidence (default 0.80)
    model          model id sent to the endpoint (default typesafe/jev-1.14-20261014, the
                   measured snapshot; the id that actually answered is recorded on every answer)
    endpoint       default https://openrouter.ai/api/alpha/decisions; https only (plain http only
                   to a loopback address, and only with allow_insecure_loopback=True)
    api_key_env    name of the environment variable holding the key (default OPENROUTER_API_KEY)
    timeout_s      per call (default 10)
    pool_size      parallel calls (default 4)
    decide         which Jev answers may decide a claim: any of supports, contradicts,
                   says_nothing. Default () = shadow: Jev decides nothing, the judge decides every
                   claim, Jev's answers are recorded beside it.
    provider       the provider routing block sent with each request (default DEFAULT_PROVIDER)
    transport      transport(endpoint, body_bytes, headers, timeout) -> reply bytes; for tests.
                   Default: one https POST, redirects refused.
    """

    def __init__(self, *, threshold: float = DEFAULT_THRESHOLD, model: str = DEFAULT_MODEL,
                 endpoint: str = DEFAULT_ENDPOINT, api_key_env: str = DEFAULT_KEY_ENV,
                 timeout_s: float = DEFAULT_TIMEOUT_S, pool_size: int = DEFAULT_POOL,
                 decide=DEFAULT_DECIDE, provider: dict | None = None, transport=None,
                 allow_insecure_loopback: bool = False):
        from .adapters import AdapterError, _checked_base_url  # local: adapters imports support

        self.threshold = _check_threshold(threshold)
        if not isinstance(model, str) or not model.strip():
            raise ValueError("the Jev model id must be a non-empty string")
        self.model = model.strip()
        try:
            self.endpoint = _checked_base_url(str(endpoint),
                                              allow_insecure_loopback=allow_insecure_loopback is True)
        except AdapterError as exc:
            raise ValueError(str(exc).replace("model endpoint", "Jev endpoint")) from None
        if not isinstance(api_key_env, str) or not api_key_env:
            raise ValueError("api_key_env must name an environment variable")
        self.api_key_env = api_key_env
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise ValueError("timeout_s must be a positive number")
        self.timeout_s = float(timeout_s)
        self.pool_size = max(1, int(pool_size))
        if isinstance(decide, str):
            raise ValueError("decide must be a list or tuple of answers, not one string")
        decide = tuple(decide)
        unknown = [c for c in decide if c not in CHOICES]
        if unknown:
            raise ValueError(f"decide must be a subset of {CHOICES}, got {decide}")
        self.decide = tuple(c for c in CHOICES if c in decide)
        self.provider = dict(DEFAULT_PROVIDER if provider is None else provider)
        self.transport = transport or _https_transport

    # ---- configuration from the environment ------------------------------------------------
    @classmethod
    def from_env(cls, environ=None, *, require_switch: bool = True, **overrides):
        """Settings from CLAIM_SEAL_JEV_* variables. With require_switch (the default) it returns
        None unless CLAIM_SEAL_JEV is on (1/on/true/yes); an unreadable switch value is an error,
        never a silent off or on.

          CLAIM_SEAL_JEV            on | off (default off)
          CLAIM_SEAL_JEV_THRESHOLD  e.g. 0.80
          CLAIM_SEAL_JEV_MODEL      model id
          CLAIM_SEAL_JEV_ENDPOINT   https URL
          CLAIM_SEAL_JEV_KEY_ENV    name of the variable that holds the key
          CLAIM_SEAL_JEV_TIMEOUT    seconds per call
          CLAIM_SEAL_JEV_DECIDE     comma list from supports,contradicts,says_nothing;
                                    unset, empty or "none" = shadow (Jev decides nothing)
        """
        env = os.environ if environ is None else environ
        switch = str(env.get("CLAIM_SEAL_JEV", "")).strip().lower()
        if switch not in _ON_WORDS | _OFF_WORDS:
            raise ValueError(f"{ENV_SWITCH} must be on or off, got {switch!r}")
        if require_switch and switch not in _ON_WORDS:
            return None
        kw: dict = {}
        if env.get("CLAIM_SEAL_JEV_THRESHOLD"):
            try:
                kw["threshold"] = float(env["CLAIM_SEAL_JEV_THRESHOLD"])
            except ValueError:
                raise ValueError("CLAIM_SEAL_JEV_THRESHOLD must be a number") from None
        if env.get("CLAIM_SEAL_JEV_MODEL"):
            kw["model"] = env["CLAIM_SEAL_JEV_MODEL"]
        if env.get("CLAIM_SEAL_JEV_ENDPOINT"):
            kw["endpoint"] = env["CLAIM_SEAL_JEV_ENDPOINT"]
        if env.get("CLAIM_SEAL_JEV_KEY_ENV"):
            kw["api_key_env"] = env["CLAIM_SEAL_JEV_KEY_ENV"]
        if env.get("CLAIM_SEAL_JEV_TIMEOUT"):
            try:
                kw["timeout_s"] = float(env["CLAIM_SEAL_JEV_TIMEOUT"])
            except ValueError:
                raise ValueError("CLAIM_SEAL_JEV_TIMEOUT must be a number") from None
        decide = str(env.get("CLAIM_SEAL_JEV_DECIDE") or "").strip().lower()
        if decide and decide != "none":
            kw["decide"] = [w.strip() for w in decide.split(",") if w.strip()]
        kw.update(overrides)
        return cls(**kw)

    @property
    def mode(self) -> str:
        return "decide" if self.decide else "shadow"

    def settings(self) -> dict:
        """What the run record keeps: never the key, only the name of its variable."""
        return {"mode": self.mode, "endpoint": self.endpoint, "model_requested": self.model,
                "threshold": self.threshold, "decide": list(self.decide),
                "timeout_s": self.timeout_s, "api_key_env": self.api_key_env}

    # ---- one call ---------------------------------------------------------------------------
    def request_body(self, claim: str, section: str) -> bytes:
        body = {"model": self.model, "provider": self.provider,
                "questions": {QUESTION_ID: QUESTION},
                "state": {"claim": claim, "section": section}}
        return json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")

    def ask(self, key: str, claim: str, section: str) -> dict:
        """One Jev answer: {choice, confidence, model, cost_usd, latency_s, request_id}, or a
        JevError. The key goes into one header and nowhere else."""
        started = time.monotonic()
        raw = self.transport(self.endpoint, self.request_body(claim, section),
                             {"Authorization": "Bearer " + key}, self.timeout_s)
        latency = round(time.monotonic() - started, 3)
        try:
            doc = json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)
        except (UnicodeDecodeError, ValueError, RecursionError, TypeError):
            raise JevError("invalid-response", "reply is not JSON") from None
        answers = doc.get("answers") if isinstance(doc, dict) else None
        answer = answers.get(QUESTION_ID) if isinstance(answers, dict) else None
        choice = answer.get("choice") if isinstance(answer, dict) else None
        confidence = answer.get("confidence") if isinstance(answer, dict) else None
        if (choice not in CHOICES or isinstance(confidence, bool)
                or not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0):
            raise JevError("invalid-response", "choice or confidence missing or out of range")
        usage = doc.get("usage") if isinstance(doc.get("usage"), dict) else {}
        cost = usage.get("cost")
        cost = float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None
        model = doc.get("model")
        request_id = doc.get("id")
        return {"choice": choice, "confidence": float(confidence),
                "model": model if isinstance(model, str) else None,
                "cost_usd": cost, "latency_s": latency,
                "request_id": request_id if isinstance(request_id, str) else None}

    # ---- the pass over one brief ------------------------------------------------------------
    def run(self, report: dict, checkable: list, windows: dict | None) -> dict:
        """Ask Jev about every checkable claim. Returns {"decisions": {n: decision},
        "record": run record}. A decision is {"by": "jev"|"judge", "reason", "answers",
        "verdict"(when by jev)}. Never raises for a Jev failure."""
        windows = windows or {}
        key = os.environ.get(self.api_key_env) or ""
        unavailable = None if key else "missing-key"
        jobs: dict = {}            # (n, citation_index) -> (claim text, window)
        plans: dict = {}           # n -> list of citation indexes asked, or a pass-up reason
        for n in checkable:
            claim = report["claims"][n]
            resolved = [i for i, x in enumerate(claim["citations"])
                        if x["resolution"] != "unresolved" and x.get("text")]
            if not resolved or any(not windows.get((n, i)) for i in resolved):
                plans[n] = "no-window"
                continue
            plans[n] = resolved
            if unavailable is None:
                for i in resolved:
                    jobs[(n, i)] = (claim["claim"], windows[(n, i)])
        results: dict = {}
        if jobs:
            def one(job):
                text, section = jobs[job]
                try:
                    return job, self.ask(key, text, section)
                except JevError as exc:
                    return job, {"error": exc.kind}
                except Exception as exc:  # noqa: BLE001 - a broken transport is a fallback, never a crash
                    return job, {"error": "transport", "detail": type(exc).__name__}
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.pool_size) as pool:
                for job, answer in pool.map(one, sorted(jobs)):
                    results[job] = answer
        key = None  # noqa: F841 - drop the reference as soon as the calls are done
        decisions: dict = {}
        for n in checkable:
            plan = plans[n]
            if plan == "no-window":
                decisions[n] = {"by": "judge", "reason": "no-window", "answers": []}
                continue
            if unavailable is not None:
                decisions[n] = {"by": "judge", "reason": "fallback:" + unavailable, "answers": []}
                continue
            answers = [dict(results[(n, i)], citation_index=i) for i in plan]
            decisions[n] = self._decide(answers)
        return {"decisions": decisions, "record": self._record(decisions, unavailable)}

    def _decide(self, answers: list) -> dict:
        errors = sorted({a["error"] for a in answers if "error" in a})
        if errors:
            return {"by": "judge", "reason": "fallback:" + errors[0], "answers": answers}
        if any(a["confidence"] < self.threshold for a in answers):
            return {"by": "judge", "reason": "below-threshold", "answers": answers}
        choices = {a["choice"] for a in answers}
        if "supports" in choices and "contradicts" in choices:
            return {"by": "judge", "reason": "citations-disagree", "answers": answers}
        choice = ("supports" if "supports" in choices else
                  "contradicts" if "contradicts" in choices else "says_nothing")
        if choice not in self.decide:
            # Jev was confident and consistent but may not decide this answer: the judge decides,
            # and Jev's claim-level answer is kept beside it for comparison.
            return {"by": "judge", "reason": "shadow" if not self.decide else "class-not-deciding",
                    "jev_choice": choice, "answers": answers}
        return {"by": "jev", "reason": None, "choice": choice, "jev_choice": choice,
                "answers": answers}

    def _record(self, decisions: dict, unavailable) -> dict:
        answers = [a for d in decisions.values() for a in d["answers"]]
        ok = [a for a in answers if "error" not in a]
        latencies = sorted(a["latency_s"] for a in ok if isinstance(a.get("latency_s"), (int, float)))
        costs = [a["cost_usd"] for a in ok if a.get("cost_usd") is not None]
        reasons: dict = {}
        fallbacks: dict = {}
        for d in decisions.values():
            if d["by"] == "judge":
                reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1
                if d["reason"].startswith("fallback:"):
                    kind = d["reason"].split(":", 1)[1]
                    fallbacks[kind] = fallbacks.get(kind, 0) + 1
        errors: dict = {}
        for a in answers:
            if "error" in a:
                errors[a["error"]] = errors.get(a["error"], 0) + 1
        return {
            "contract": FIRST_PASS_CONTRACT,
            "mode": self.mode,
            "status": "unavailable" if unavailable else "ok",
            **({"unavailable": unavailable} if unavailable else {}),
            "settings": self.settings(),
            "claims_asked": sum(1 for d in decisions.values() if d["answers"]),
            "decided_by_jev": sum(1 for d in decisions.values() if d["by"] == "jev"),
            "passed_to_judge": sum(1 for d in decisions.values() if d["by"] == "judge"),
            "pass_up_reasons": reasons,
            "fallbacks": sum(fallbacks.values()),
            "fallback_reasons": fallbacks,
            "calls": len(answers),
            "call_errors": errors,
            "cost_usd": round(sum(costs), 9) if costs else (0.0 if not ok else None),
            "cost_unreported_calls": len(ok) - len(costs),
            "latency_s": ({"p50": latencies[len(latencies) // 2], "max": latencies[-1]}
                          if latencies else None),
            "models_returned": sorted({a["model"] for a in ok if a.get("model")}),
        }

    # ---- what a decision becomes -------------------------------------------------------------
    def verdict_entry(self, decision: dict, windows: dict, n: int, *, spans: bool) -> dict:
        """The judge-shaped verdict entry for a claim Jev decided."""
        verdict, relation = VERDICT_BY_CHOICE[decision["choice"]]
        answers = decision["answers"]
        lowest = min(a["confidence"] for a in answers)
        entry = {"verdict": verdict, "relation": relation,
                 "why": (f"decided by the Jev first pass: {decision['choice']} for "
                         + ("each of the " + str(len(answers)) + " cited sources"
                            if len(answers) > 1 else "the cited source")
                         + f" (lowest confidence {lowest:.2f}, threshold {self.threshold:.2f})")[:200]}
        if spans:
            entry["citation_spans"] = []
        if verdict == "supported":
            # The passage Jev read, located in the source by the same code that locates a judge
            # quote. Only for the citations Jev said support the claim.
            entry["evidence_spans"] = [{"citation_index": a["citation_index"],
                                        "support_span": windows[(n, a["citation_index"])]}
                                       for a in answers if a["choice"] == "supports"]
        else:
            entry["evidence_spans"] = []
        entry["strength"] = None
        entry["multi_span_required"] = None
        return entry

    def sealed_decider(self, decision: dict | None) -> dict:
        """The decider block kept inside every claim-evidence row of a claim (sealed)."""
        if decision is None:
            return {"by": "none", "reason": "not-checkable", "first_pass": FIRST_PASS_CONTRACT,
                    "mode": self.mode, "threshold": self.threshold, "answers": []}
        out = {"by": decision["by"], "reason": decision["reason"], "first_pass": FIRST_PASS_CONTRACT,
               "mode": self.mode, "threshold": self.threshold,
               "jev_choice": decision.get("jev_choice"),
               "answers": [{k: a.get(k) for k in ("citation_index", "choice", "confidence", "model")}
                           if "error" not in a else {"citation_index": a["citation_index"],
                                                     "error": a["error"]}
                           for a in decision["answers"]]}
        if decision["by"] == "jev":
            out["choice"] = decision["choice"]
            out["section"] = "evidence-window"
        return out

    def claim_decider(self, decision: dict | None, reading: str | None = None) -> dict:
        """The decided_by block on a claim's verification (not sealed): the sealed fields plus
        cost, latency and request id per answer, and claim-seal's own reading of the claim
        (see claim_seal_reading) when the judge decided it."""
        out = self.sealed_decider(decision)
        if decision is not None:
            out["answers"] = [dict(a) for a in decision["answers"]]
            if decision["by"] == "judge":
                out["claim_seal_reading"] = reading
        return out


# How Jev's claim-level answer compares with what claim-seal recorded for the same claim.
AGREES_WITH = {"supports": "supported", "contradicts": "unsupported-contradicts",
               "says_nothing": "unsupported"}


def claim_seal_reading(verification: dict, relation=None) -> str:
    """claim-seal's verdict for a claim in words comparable with Jev's: supported, partial,
    unsupported-contradicts (the judge named the relation contradicts), unsupported, unresolved,
    unchecked."""
    from .relations import read_relation  # local: keeps this module light
    verdict = verification.get("verdict")
    override = (verification.get("verdict_override") or {}).get("rule")
    if verdict == "unsupported" and (override == "relation-contradicts"
                                     or read_relation(relation)[0] == "contradicts"):
        return "unsupported-contradicts"
    return str(verdict)


def compare_with_judge(decided_by: list) -> dict:
    """Over the claims the judge decided where Jev gave a confident claim-level answer: how many
    there were, how many agreed, and every (Jev answer, claim-seal reading) pair counted."""
    pairs: dict = {}
    agree = compared = 0
    for block in decided_by:
        choice = block.get("jev_choice")
        if block.get("by") != "judge" or not choice:
            continue
        reading = block.get("claim_seal_reading")
        compared += 1
        agree += AGREES_WITH.get(choice) == reading
        key = f"{choice} vs {reading}"
        pairs[key] = pairs.get(key, 0) + 1
    return {"compared": compared, "agree": agree,
            "agreement": round(agree / compared, 3) if compared else None,
            "pairs": dict(sorted(pairs.items()))}


def resolve_first_pass(first_pass):
    """judge()'s first_pass argument -> a JevFirstPass or None. None and False are off; "env"
    reads JevFirstPass.from_env() (off unless CLAIM_SEAL_JEV is on)."""
    if first_pass is None or first_pass is False:
        return None
    if first_pass == "env":
        return JevFirstPass.from_env()
    if isinstance(first_pass, JevFirstPass):
        return first_pass
    raise ValueError("first_pass must be None, \"env\" or a claim_seal.jev.JevFirstPass")


__all__ = ["JevFirstPass", "JevError", "resolve_first_pass", "claim_seal_reading",
           "compare_with_judge", "AGREES_WITH", "DEFAULT_DECIDE", "CHOICES", "QUESTION",
           "VERDICT_BY_CHOICE", "FIRST_PASS_CONTRACT", "DEFAULT_ENDPOINT", "DEFAULT_MODEL",
           "DEFAULT_KEY_ENV", "DEFAULT_THRESHOLD"]

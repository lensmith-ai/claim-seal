"""claim-seal: check each claim in a piece of writing against the sources it cites, record where
the writing takes a step beyond its sources, and seal the result in a SHA-256 digest chain.

    from claim_seal import resolve, judge, seal, verify_seal, goes_further
    from claim_seal.adapters import ReplayJudge, OpenAICompatible

Pass 1, resolve(), is pure code: does every cited reference exist in the sources?
Pass 2, judge(), asks a model (any callable prompt -> text) whether each resolved citation
supports its claim, which sentence it bears on, and how that sentence relates to it.
The result's claim-evidence record is sealed; verify_seal() recomputes the chain.
Optional, off by default: judge(..., first_pass=JevFirstPass()) asks a small classifier first
and sends only the claims it is unsure of to the judge (claim_seal.jev).

It does not prove claims true. It checks support by the given sources only, the support judge is
a model and can be wrong, and the seal is an unkeyed digest chain, not a signature.
"""
from .evidence import build_claim_evidence, capture_sources, digests_for, source_lineage
from .support import ModelText, extract_json, judge, overall_status, verify
from .relations import (GOES_FURTHER, LINE_RULE, SPAN_RELATIONS, STEP_BEYOND, UnknownRelation,
                        derive_coarse_relation, goes_further, line_rule, line_states)
from .resolve import BriefError, SourceError, resolve, resolve_only, shape_brief
from .seal import seal, seal_problem, verify_seal
from .limits import InputTooLarge
from .completion import CompletionDisabled
from .jev import JevFirstPass
from . import limits  # noqa: F401 - the ceilings, as claim_seal.limits

__version__ = "0.2.0"

__all__ = [
    "resolve", "resolve_only", "judge", "verify", "overall_status",
    "seal", "verify_seal", "seal_problem", "build_claim_evidence", "capture_sources",
    "digests_for", "source_lineage",
    "SPAN_RELATIONS", "STEP_BEYOND", "GOES_FURTHER", "LINE_RULE", "UnknownRelation",
    "derive_coarse_relation", "goes_further", "line_rule", "line_states",
    "ModelText", "extract_json", "SourceError", "BriefError", "shape_brief", "__version__",
    "InputTooLarge", "CompletionDisabled", "limits", "JevFirstPass",
]

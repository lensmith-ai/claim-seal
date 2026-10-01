"""claim-seal: check each claim in a piece of writing against the sources it cites, record where
the writing takes a step beyond its sources, and seal the result in a SHA-256 digest chain.

    from claim_seal import resolve, judge, seal, verify_seal, through_line
    from claim_seal.adapters import ReplayJudge, OpenAICompatible

Pass 1, resolve(), is pure code: does every cited reference exist in the sources?
Pass 2, judge(), asks a model (any callable prompt -> text) whether each resolved citation
supports its claim, which sentence it bears on, and how that sentence relates to it.
The result's claim-evidence record is sealed; verify_seal() recomputes the chain.

It does not prove claims true. It checks support by the given sources only, the support judge is
a model and can be wrong, and the seal is an unkeyed digest chain, not a signature.
"""
from .evidence import build_claim_evidence, capture_sources, digests_for, source_lineage
from .support import ModelText, extract_json, judge, overall_status, verify
from .relations import (LINE_RULE, SPAN_RELATIONS, STEP_BEYOND, THROUGH_LINE, UnknownRelation,
                        derive_coarse_relation, line_rule, line_states, through_line)
from .resolve import SourceError, resolve, resolve_only
from .seal import seal, seal_problem, verify_seal

__version__ = "0.1.0"

__all__ = [
    "resolve", "resolve_only", "judge", "verify", "overall_status",
    "seal", "verify_seal", "seal_problem", "build_claim_evidence", "capture_sources",
    "digests_for", "source_lineage",
    "SPAN_RELATIONS", "STEP_BEYOND", "THROUGH_LINE", "LINE_RULE", "UnknownRelation",
    "derive_coarse_relation", "through_line", "line_rule", "line_states",
    "ModelText", "extract_json", "SourceError", "__version__",
]

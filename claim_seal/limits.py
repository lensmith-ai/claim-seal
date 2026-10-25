"""Size and depth ceilings for everything claim-seal reads: input files, JSON, source bodies,
model replies. Over a ceiling, the input is refused with InputTooLarge (the CLI exits 2), never
parsed partway or allowed to exhaust memory or the recursion limit.

The ceilings are generous for real briefs (a brief is a few kilobytes; a captured web page a few
megabytes) and are module attributes, so a caller with a larger, trusted workload can raise them
before calling judge(). They are read at call time.
"""
from __future__ import annotations

import json
import os

MAX_INPUT_FILE_BYTES = 64 * 1024 * 1024     # one brief, sources, answers or record file
MAX_JSON_DEPTH = 64                          # nesting of arrays and objects (a brief needs ~6)
MAX_JSON_VALUES = 2_000_000                  # values of any kind in one JSON document
MAX_CLAIMS = 5_000                           # items + analysis statements + connection
MAX_CITATIONS = 50_000                       # citations across all claims
MAX_SOURCE_ITEMS = 50_000                    # items across all sources
MAX_SOURCE_BODY_BYTES = 16 * 1024 * 1024     # one item's body (or text), UTF-8 bytes
MAX_TOTAL_SOURCE_BYTES = 64 * 1024 * 1024    # all items' bodies together
MAX_MODEL_RESPONSE_BYTES = 8 * 1024 * 1024   # one HTTP reply from the model endpoint
MAX_MODEL_TEXT_CHARS = 256_000               # one model answer handed to the JSON extractor


class InputTooLarge(ValueError):
    """An input is over one of the ceilings above, too deeply nested, or not decodable."""


def _shape_problem(doc) -> str | None:
    """Depth and value count of a decoded JSON value (or a dict/list from a Python caller),
    walked without recursion. bytes, str, numbers, booleans and None are leaves."""
    stack = [(doc, 1)]
    values = 0
    while stack:
        value, depth = stack.pop()
        values += 1
        if values > MAX_JSON_VALUES:
            return f"holds more than {MAX_JSON_VALUES} JSON values"
        if isinstance(value, dict):
            children = value.values()
        elif isinstance(value, (list, tuple)):
            children = value
        else:
            continue
        if depth > MAX_JSON_DEPTH:
            return f"is nested more than {MAX_JSON_DEPTH} levels deep"
        stack.extend((child, depth + 1) for child in children)
    return None


def check_json_shape(doc, what: str = "input") -> None:
    problem = _shape_problem(doc)
    if problem:
        raise InputTooLarge(f"{what} {problem}")


def load_json_file(path: str, what: str | None = None):
    """Read one JSON file under the ceilings: size checked before reading, strict UTF-8, depth and
    value count checked after decoding. Raises OSError for a missing file, InputTooLarge for the
    rest."""
    what = what or os.path.basename(path)
    with open(path, "rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        if size > MAX_INPUT_FILE_BYTES:
            raise InputTooLarge(f"{what} is larger than {MAX_INPUT_FILE_BYTES} bytes")
        raw = handle.read(MAX_INPUT_FILE_BYTES + 1)
    if len(raw) > MAX_INPUT_FILE_BYTES:
        raise InputTooLarge(f"{what} is larger than {MAX_INPUT_FILE_BYTES} bytes")
    try:
        doc = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise InputTooLarge(f"{what} is not valid UTF-8 (byte {exc.start})") from None
    except RecursionError:
        raise InputTooLarge(f"{what} is nested too deeply to read") from None
    except MemoryError:
        raise InputTooLarge(f"{what} is too large to read into memory") from None
    except json.JSONDecodeError as exc:
        raise InputTooLarge(f"{what} is not valid JSON: {exc.msg} at line {exc.lineno}") from None
    check_json_shape(doc, what)
    return doc


def _body_bytes(item: dict) -> int:
    body = item["body"] if "body" in item else item.get("text")
    if body is None:
        return 0
    if isinstance(body, (bytes, bytearray)):
        return len(body)
    text = str(body)
    # A str of n code points is at most 4n UTF-8 bytes: skip the encode for most bodies.
    return len(text) if text.isascii() else len(text.encode("utf-8", "surrogatepass"))


def check_inputs(brief, sources) -> None:
    """The ceilings on one check's inputs, applied before any model call."""
    from .resolve import claim_units, load_sources, shape_brief  # local: resolve is heavier
    check_json_shape(brief, "the brief")
    check_json_shape(sources, "the sources")
    claims = citations = 0
    for _, _, _, claim in claim_units(shape_brief(brief)):
        claims += 1
        citations += len(claim.get("sources") or [])
        if claims > MAX_CLAIMS:
            raise InputTooLarge(f"the brief has more than {MAX_CLAIMS} claims")
        if citations > MAX_CITATIONS:
            raise InputTooLarge(f"the brief has more than {MAX_CITATIONS} citations")
    items = total = 0
    for sid, entry in load_sources(sources).items():
        for n, item in enumerate(entry["items"]):
            items += 1
            if items > MAX_SOURCE_ITEMS:
                raise InputTooLarge(f"the sources hold more than {MAX_SOURCE_ITEMS} items")
            size = _body_bytes(item)
            if size > MAX_SOURCE_BODY_BYTES:
                raise InputTooLarge(f"source {sid!r} item {n + 1} is larger than "
                                    f"{MAX_SOURCE_BODY_BYTES} bytes")
            total += size
            if total > MAX_TOTAL_SOURCE_BYTES:
                raise InputTooLarge(f"the source bodies together are larger than "
                                    f"{MAX_TOTAL_SOURCE_BYTES} bytes")

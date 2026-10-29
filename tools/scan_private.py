#!/usr/bin/env python3
"""Fail if the repository carries a string that should not be published.

    python3 tools/scan_private.py [ROOT]              scan the working tree
    python3 tools/scan_private.py --history [ROOT]    scan every patch and message in git history
    exit 0 clean, 1 at least one hit

Built-in checks are generic only: absolute server and home paths, environment-variable names
this project does not use, credential shapes (API keys, tokens, private-key blocks), internal-id
shapes (timestamp job ids, dated run ids), and personal email addresses.

Project-specific terms (names, ids, anything private) do NOT belong in this file, which is
public. Put them in a deny-list file kept outside the repository and point CLAIM_SEAL_DENYLIST
at it. Its format: one Python regex per line, matched case-insensitively, optionally followed by
a TAB and comma-separated repo paths where that pattern is allowed; blank and '#' lines are
ignored.

Skips .git, .venv and caches; a file it cannot read as UTF-8 text is itself a hit.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys

SKIP_DIRS = {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache"}
# Binary files this scan cannot read as text. Each was reviewed by eye before release and is pinned
# by its exact sha256: any change to the bytes, or any binary not listed here, fails the scan.
REVIEWED_BINARIES = {
    "docs/readme/how-it-works.png": "727e245f6d902403c033f7f5aece5e9d678bd0655fc76f32110d8a51aff652a6",
}
DENYLIST_ENV = "CLAIM_SEAL_DENYLIST"

# Environment variables this project reads or documents. Any other name read from or exported
# to the environment is flagged: it may be a leftover from somewhere else.
ALLOWED_ENV = re.compile(r"^(CLAIM_SEAL_[A-Z0-9_]+|OPENAI_API_KEY|OPENROUTER_API_KEY)$")
_ENV_USE = re.compile(r"""(?:environ(?:\.get)?\(\s*|environ\[\s*|getenv\(\s*)["']([A-Z][A-Z0-9_]*)["']"""
                      r"""|\bexport\s+([A-Z][A-Z0-9_]*)=""")

GENERIC = [
    ("absolute server path", re.compile(r"(?<![\w.:/])/(?:srv|root|opt|var/www|var/lib|etc)/[\w.-]")),
    ("home directory path", re.compile(r"(?<![\w.:/])(?:/home/|/Users/|~/)[\w.-]+")),
    ("api key shape", re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_]{20,}")),
    ("token shape", re.compile(r"\b(?:ghp|gho|github_pat|glpat|xox[abpr])[-_][A-Za-z0-9_-]{16,}")),
    ("cloud key shape", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("bot token shape", re.compile(r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("job id shape", re.compile(r"\b\d{8}T\d{6}Z-[0-9a-f]{6}\b", re.I)),
    ("run id shape", re.compile(r"\b\d{4}-\d{2}-\d{2}-\d{4}\b")),
    ("personal email", re.compile(r"\b[\w.+-]+@(?!(?:[\w-]+\.)*(?:example\.(?:com|org|net|test)|anthropic\.com)\b)"
                                  r"(?![\w.-]*noreply)[\w-]+\.[\w.-]+\b")),
]


def load_denylist(path: str | None) -> list[tuple[str, re.Pattern, set]]:
    """[(label, pattern, allowed paths)] from a private deny-list file, or [] when none."""
    if not path:
        return []
    out = []
    with open(path, encoding="utf-8") as handle:
        for number, raw in enumerate(handle, 1):
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            pattern, _, allowed = line.partition("\t")
            out.append((f"deny-list entry {number}", re.compile(pattern, re.I),
                        {p.strip() for p in allowed.split(",") if p.strip()}))
    return out


def check_line(rel: str, line: str, extra) -> list[tuple[str, str]]:
    found = []
    for label, pattern in GENERIC:
        m = pattern.search(line)
        if m:
            found.append((label, m.group(0)))
    for m in _ENV_USE.finditer(line):
        name = m.group(1) or m.group(2)
        if not ALLOWED_ENV.match(name):
            found.append(("unexpected environment variable", name))
    for label, pattern, allowed in extra:
        m = pattern.search(line)
        if m and rel not in allowed:
            found.append((label, m.group(0)))
    return found


def scan(root: str, extra=()) -> tuple[list[tuple[str, int, str, str]], int]:
    """(hits, number of text files scanned) over the working tree under root."""
    hits, scanned = [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root)
            try:
                with open(path, encoding="utf-8") as handle:
                    lines = handle.read().splitlines()
            except (UnicodeDecodeError, OSError) as exc:
                # Fail closed: a file this scan cannot read is a file it cannot vouch for,
                # unless it is a reviewed binary whose bytes are unchanged.
                pinned = REVIEWED_BINARIES.get(rel.replace(os.sep, "/"))
                if pinned is None:
                    hits.append((rel, 0, "unscannable file", type(exc).__name__))
                    continue
                with open(path, "rb") as handle:
                    actual = hashlib.sha256(handle.read()).hexdigest()
                if actual != pinned:
                    hits.append((rel, 0, "reviewed binary changed", actual[:12]))
                continue
            scanned += 1
            for number, line in enumerate(lines, 1):
                for label, text in check_line(rel, line, extra):
                    hits.append((rel, number, label, text))
    return hits, scanned


def scan_history(root: str, extra=()) -> tuple[list[tuple[str, int, str, str]], int]:
    """Scan every commit message and every patch line in all refs. (hits, lines scanned)."""
    log = subprocess.run(["git", "-C", root, "log", "-p", "--all", "--format=COMMIT %H%n%B"],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    hits, rel, commit = [], "(message)", "?"
    for number, line in enumerate(log, 1):
        if line.startswith("COMMIT "):
            commit, rel = line[7:15], "(message)"
            continue
        if line.startswith("+++ b/"):
            rel = line[6:]
            continue
        if line.startswith(("+++ ", "--- ", "diff --git ", "index ", "@@ ")):
            continue
        for label, text in check_line(rel, line, extra):
            hits.append((f"{commit}:{rel}", number, label, text))
    return hits, len(log)


def main(argv: list[str]) -> int:
    args = argv[1:]
    history = "--history" in args
    args = [a for a in args if a != "--history"]
    root = args[0] if args else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    extra = load_denylist(os.environ.get(DENYLIST_ENV))
    hits, count = (scan_history if history else scan)(root, extra)
    for rel, number, label, text in hits:
        print(f"{rel}:{number}: {label}: {text!r}")
    what = f"{count} history line(s)" if history else f"{count} text file(s)"
    print(f"private-string scan: {len(hits)} hit(s) across {what}; "
          f"generic patterns + {len(extra)} deny-list pattern(s)")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

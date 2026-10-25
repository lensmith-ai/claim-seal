#!/bin/sh
# Clean build-and-install smoke gate: build the sdist and wheel with only the hash-pinned build
# requirement, install the wheel into a second, empty venv with no index and no dependencies, run
# the installed `claim-seal` command on the demo from outside the repository, compare its output with
# examples/demo-output.txt, check the seal it wrote, and print the artifacts' SHA256SUMS.
#   sh tools/build_check.sh [OUT_DIR]      (OUT_DIR defaults to a new temporary directory)
# Needs network access once, for pip to fetch the pinned setuptools wheel.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
OUT=${1:-$WORK/dist}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
PY=${PYTHON:-python3}

"$PY" -m venv "$WORK/build-env"
"$WORK/build-env/bin/python" -m pip install --quiet --disable-pip-version-check --no-deps \
    --require-hashes --only-binary=:all: -r "$ROOT/tools/build-requirements.txt"
# Build from a copy, so build/ and *.egg-info never land in the working tree.
mkdir "$WORK/src"
(cd "$ROOT" && tar -cf - --exclude=.git --exclude=__pycache__ --exclude=.venv --exclude=build \
    --exclude=dist --exclude='*.egg-info' .) | (cd "$WORK/src" && tar -xf -)
for hook in build_sdist build_wheel; do        # one process per hook, as a build frontend does
    (cd "$WORK/src" && "$WORK/build-env/bin/python" -I -c "
import sys
from setuptools import build_meta
print(getattr(build_meta, sys.argv[1])(sys.argv[2]))" "$hook" "$OUT")
done

"$PY" -m venv "$WORK/install-env"
"$WORK/install-env/bin/python" -m pip install --quiet --disable-pip-version-check --no-index \
    --no-deps "$OUT"/claim_seal-*.whl
mkdir "$WORK/run"
cd "$WORK/run"
"$WORK/install-env/bin/claim-seal" check "$ROOT/examples/brief.json" "$ROOT/examples/sources.json" \
    --judge "replay:$ROOT/examples/replay-answers.json" --out "$WORK/run/result.json" > "$WORK/run/out.txt"
# The installed command must print exactly the golden demo output (plus its "wrote" line).
head -n "$(wc -l < "$ROOT/examples/demo-output.txt")" "$WORK/run/out.txt" | diff - "$ROOT/examples/demo-output.txt"
"$WORK/install-env/bin/claim-seal" verify-seal "$WORK/run/result.json" > /dev/null
(cd "$OUT" && sha256sum claim_seal-* | tee SHA256SUMS)
echo "build-check: sdist and wheel built with the hash-pinned setuptools, installed clean, demo matched"

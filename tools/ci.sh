#!/bin/sh
# Everything a change must pass: the tests, the private-string scan, the offline demo.
set -eu
cd "$(dirname "$0")/.."
python3 -m unittest discover -s tests -t tests
python3 tools/scan_private.py
python3 -m claim_seal check examples/brief.json examples/sources.json \
    --judge replay:examples/replay-answers.json > /dev/null
echo "ci: tests, scan and demo all passed"

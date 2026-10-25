"""Make a tampered copy of a check result, to watch verify-seal refuse it.

    python3 examples/tamper.py RESULT.json TAMPERED.json [upgrade|append|relabel|edit]

upgrade  turns the first step-beyond verb into "says" (claims more support than was judged)
append   appends a copy of the first outcome row
relabel  marks the first defaulted verb as the judge's own choice
edit     raises the first row's strength to 100

TAMPERED.json must not exist yet: like `check --out`, this never overwrites a file and never
writes through a symlink.
"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from claim_seal.limits import load_json_file  # noqa: E402
from claim_seal.output import OutputRefused, write_text  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
how = sys.argv[3] if len(sys.argv) > 3 else "upgrade"
doc = load_json_file(src)
rows = doc["claim_evidence"]["outcomes"]
if how == "upgrade":
    row = next(r for r in rows if r["relation_coarse"] == "inferred")
    row["relation"], row["relation_coarse"] = "says", "stated"
elif how == "append":
    rows.append(copy.deepcopy(rows[0]))
elif how == "relabel":
    row = next((r for r in rows if r["relation_source"] == "compat-default"), rows[0])
    row["relation_source"] = "judge"
elif how == "edit":
    rows[0]["strength"] = 100
else:
    sys.exit(f"unknown tamper {how!r}")
try:
    write_text(dst, json.dumps(doc, ensure_ascii=False, indent=1))
except OutputRefused as exc:
    sys.exit(f"refused: {exc}")
print(f"wrote {dst} ({how}); the seal value beside the rows was left as it was")

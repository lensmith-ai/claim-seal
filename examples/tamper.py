"""Make a tampered copy of a check result, to watch verify-seal refuse it.

    python3 examples/tamper.py RESULT.json TAMPERED.json [upgrade|append|relabel|edit]

upgrade  turns the first step-beyond verb into "says" (claims more support than was judged)
append   appends a copy of the first outcome row
relabel  marks the first defaulted verb as the judge's own choice
edit     raises the first row's strength to 100
"""
import copy
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
how = sys.argv[3] if len(sys.argv) > 3 else "upgrade"
doc = json.load(open(src, encoding="utf-8"))
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
json.dump(doc, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"wrote {dst} ({how}); the seal value beside the rows was left as it was")

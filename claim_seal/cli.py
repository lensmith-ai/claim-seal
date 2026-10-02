"""claim-seal command line.

  claim-seal check BRIEF.json SOURCES.json [--judge replay:ANSWERS.json | --judge openai[:MODEL]]
                   [--out RESULT.json] [--whole-brief] [--no-spans] [--no-evidence] ...
  claim-seal verify-seal RECORD.json [RECORD.json ...]

check without --judge runs pass 1 only (are the cited references real?) and says so.
verify-seal accepts a claim-evidence record, or a check result that contains one.
Exit codes: check 0 ran, 2 bad input. verify-seal 0 all verify, 1 at least one fails (including a
record that has a seal but no rows, or a result whose claim_evidence is null), 2 nothing to check.
"""
from __future__ import annotations

import argparse
import json
import sys

from .adapters import AdapterError, OpenAICompatible, ReplayJudge
from .support import judge
from .relations import THROUGH_LINE, UnknownRelation, line_states
from .attribution import check_span
from .resolve import SourceError, load_sources, resolve_only
from .seal import seal_problem


def _load(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _model_from(spec: str):
    kind, _, arg = spec.partition(":")
    if kind == "replay":
        if not arg:
            raise SystemExit("--judge replay needs a file: --judge replay:ANSWERS.json")
        return ReplayJudge.from_file(arg)
    if kind == "openai":
        return OpenAICompatible(model=arg or None)
    raise SystemExit(f"unknown --judge {spec!r}: use replay:FILE or openai[:MODEL]")


def _claim_text(obj: dict) -> str:
    return obj.get("text") or obj.get("insight") or obj.get("headline") or ""


def _walk(brief: dict):
    for i, item in enumerate(brief.get("items") or []):
        yield f"item {i + 1}", item
        for k, statement in enumerate(item.get("analysis") or []):
            if isinstance(statement, dict):
                yield f"item {i + 1} analysis {k + 1}", statement
    if isinstance(brief.get("connection"), dict):
        yield "connection", brief["connection"]


def _attribution_text(row: dict) -> str:
    verb = row.get("attribution_verb")
    if not verb:
        return "attribution: none found"
    text = f'attribution: "{verb}" ({row["attribution_strength"]}) vs relation {row.get("relation") or "-"}'
    return text + (f' -> {row["flag"]}: {row["message"]}' if row.get("flag") else " -> no flag")


def _print_report(result: dict, out, labels: dict | None = None) -> None:
    brief = result["brief"]
    rec = result["verification"]
    w = out.write
    w(f"overall: {result.get('status')}    "
      f"citations: {rec['resolution']['citations']} "
      f"(exact {rec['resolution']['exact']}, fuzzy {rec['resolution']['fuzzy']}, "
      f"unresolved {rec['resolution']['unresolved']})\n")
    w(f"verdicts: {json.dumps(rec.get('verdicts'))}\n\n")
    for where, obj in _walk(brief):
        v = obj.get("verification") or {}
        w(f"[{where}] {v.get('verdict', '-').upper()}: {_claim_text(obj)}\n")
        w(f"    why: {v.get('why', '')}\n")
        for n, cite in enumerate(v.get("citations") or []):
            line = f"    cite {n}: {cite['source']} / {cite['ref']!r} -> {cite['resolution']}"
            span = cite.get("span")
            if span:
                rel = span.get("sentence_relation")
                label = THROUGH_LINE.get(rel, "-") if rel else "unrated (no relation)"
                line += (f"\n        span [{span['field']}] {span['quote']!r}"
                         f"\n        relation: {rel or '-'}  through line: {label}"
                         f"  (placed by {span['producer']})")
            if span:
                found = check_span(obj, cite, labels)
                if found is not None:
                    line += "\n        " + _attribution_text(found)
            if cite.get("bears"):
                line += f"\n        bears: {cite['bears']}"
            w(line + "\n")
        w("\n")
    try:
        states = line_states(brief, labels=labels)
    except UnknownRelation as exc:
        w(f"through-line page states refused: {exc}\n")
    else:
        w("what a reader-facing page may print:\n")
        for line in states["lines"]:
            w(f"  item {line['item']} [{line['field']}] {line['label']} "
              f"({line['state_scope']}): {line['text']}\n")
            for row in line.get("attribution") or []:
                if row.get("flag"):
                    w(f"      attribution {row['flag']}: {row['message']}\n")
        for row in states["omitted"]:
            w(f"  item {row['item']} omitted: {row['rule']}\n")
        for row in states["left_out"]:
            w(f"  item {row['item']} [{row['field']}] left out: {row['rule']}\n")
            for found in row.get("attribution") or []:
                if found.get("flag"):
                    w(f"      attribution {found['flag']}: {found['message']}\n")
    evidence = result.get("claim_evidence")
    if evidence:
        found = evidence["verification"]["seal"]
        w(f"\nclaim-evidence rows: {len(evidence['outcomes'])}  "
          f"states: {json.dumps(_tally(evidence['outcomes'], 'support_state'))}\n")
        w(f"seal: {found['algorithm']} {found['value']}  (unkeyed; tamper-evident only "
          f"alongside version history; not a signature)\n")
        w("seal covers: the claim-evidence rows (outcome, span, verb, judge-chose flag, binding); "
          "not the verdicts, sentence relations, through-line labels, attribution flags or rule "
          "ids above\n")
        problem = seal_problem(evidence)
        w(f"seal recomputes: {'yes' if problem is None else 'NO - ' + problem}\n")


def _tally(rows, key):
    out: dict = {}
    for row in rows:
        out[row[key]] = out.get(row[key], 0) + 1
    return out


def cmd_check(args) -> int:
    try:
        brief, sources = _load(args.brief), _load(args.sources)
    except (OSError, ValueError) as exc:
        print(f"cannot read input: {exc}", file=sys.stderr)
        return 2
    try:
        if not args.judge:
            receipts: dict = {}
            rec = resolve_only(brief, sources, receipts)
            print("pass 1 only (no --judge given): citations are checked for being real, "
                  "not for support")
            print(f"resolution: {json.dumps(rec['resolution'])}  verdicts: {json.dumps(rec['verdicts'])}")
            for where, obj in _walk(brief):
                v = obj.get("verification") or {}
                print(f"[{where}] {v.get('verdict')}: {_claim_text(obj)}")
            result = {"status": "resolve-only", "brief": brief, "verification": rec}
        else:
            model = _model_from(args.judge)
            result = judge(brief, sources, model, spans=not args.no_spans,
                           sharded=not args.whole_brief, local_placement=not args.no_local_placement,
                           evidence=not args.no_evidence, completion=args.completion,
                           pool_size=args.pool, claim_deadline_s=args.deadline)
            _print_report(result, sys.stdout,
                          {sid: e["label"] for sid, e in load_sources(sources).items()})
    except (SourceError, AdapterError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=1)
            handle.write("\n")
        print(f"\nwrote {args.out}")
    return 0


def cmd_verify_seal(args) -> int:
    checked = failed = passed = 0
    for path in args.records:
        try:
            doc = _load(path)
        except (OSError, ValueError) as exc:
            print(f"UNREADABLE {path}: {type(exc).__name__}")
            failed += 1
            continue
        is_result = isinstance(doc, dict) and "claim_evidence" in doc
        record = doc.get("claim_evidence") if is_result else doc
        seal_value = (((record.get("verification") or {}).get("seal") or {}).get("value")
                      if isinstance(record, dict) and isinstance(record.get("verification"), dict) else None)
        if is_result and not isinstance(record, dict):
            # A result that carries a claim_evidence slot with nothing in it lost its record.
            checked += 1
            failed += 1
            print(f"CHAIN FAIL {path}: claim_evidence is {type(record).__name__}, not a sealed record")
            continue
        if not isinstance(record, dict) or (not record.get("outcomes") and not seal_value):
            print(f"NOTHING    {path}: no claim-evidence outcome rows and no seal")
            continue
        checked += 1
        problem = seal_problem(record)
        if problem is None:
            passed += 1
            print(f"OK         {path}: claim_evidence.outcomes: {len(record['outcomes'])} rows recompute "
                  "(outcome, span, verb, judge-chose flag, binding). Not sealed: claim verdicts, "
                  "per-sentence relations and through-line labels, rule ids, other fields of this file")
        else:
            failed += 1
            print(f"CHAIN FAIL {path}: {problem}")
    print(f"\nsealed records checked {checked} · verified {passed} · failed {failed}")
    print("note: the chain is unkeyed. It shows a record changed after its seal value was "
          "witnessed (for example in version history); it is not a signature.")
    if failed:
        return 1
    if not checked:
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="claim-seal", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="check a brief's claims against its sources")
    check.add_argument("brief", help="the brief (claims) JSON")
    check.add_argument("sources", help="the sources JSON")
    check.add_argument("--judge", help="replay:ANSWERS.json (offline) or openai[:MODEL]")
    check.add_argument("--out", help="write the full result JSON here")
    check.add_argument("--whole-brief", action="store_true",
                       help="one model call for the whole brief instead of one per claim "
                            "(no per-sentence relations)")
    check.add_argument("--no-spans", action="store_true", help="do not ask for sentence spans")
    check.add_argument("--no-local-placement", action="store_true",
                       help="do not move thesis marks to the overlapping supporting sentence")
    check.add_argument("--no-evidence", action="store_true",
                       help="skip the evidence pass (no claim-evidence record, no seal)")
    check.add_argument("--completion", action="store_true",
                       help="try to complete partial claims with one more source item")
    check.add_argument("--pool", type=int, default=2, help="parallel per-claim calls (default 2)")
    check.add_argument("--deadline", type=float, default=120.0,
                       help="per-claim deadline in seconds, queue wait included (default 120)")
    check.set_defaults(func=cmd_check)
    verify = sub.add_parser("verify-seal", help="recompute the digest chain of sealed records")
    verify.add_argument("records", nargs="+")
    verify.set_defaults(func=cmd_verify_seal)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

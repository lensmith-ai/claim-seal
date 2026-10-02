# claim-seal

Check each claim in a piece of writing against the sources it cites. Record, sentence by
sentence, where the writing states what a source says and where it takes a step beyond it (its
"through line"). Seal the claim-evidence rows (per claim and citation: the outcome, the located
source passages, the relation verb, whether the judge chose that verb, and the source binding)
in a SHA-256 digest chain, so a later edit to those rows shows. The seal does NOT cover the claim
verdicts and their reasons, the per-sentence relations and through-line labels, or the record's
rule-id summary: see [what the seal covers](#the-seal-and-its-limits).

Python 3.10+, standard library only. MIT licensed.

claim-seal was extracted from a private composer that writes short sourced briefs and checks
every one before a reader sees it. The method, the judge prompts and the defaults are the ones
that composer runs in production; [what changed in extraction](#what-changed-in-extraction) is
listed below.

## What it does not do

- **It does not prove anything true.** It checks whether the *given sources* support a claim.
  A claim faithfully supported by a wrong source passes.
- **The support judge is a model and can be wrong.** Pass 2 is one model's reading. The library
  keeps that reading honest (a quote must be found exactly, a "supported" with no located
  passage is not recorded as support, a model failure is "unchecked", never a pass), but it
  cannot make the reading correct.
- **The seal is not a signature.** It is an unkeyed digest chain: tamper-evident only alongside
  version history (see [The seal](#the-seal-and-its-limits)).

## The two passes

**Pass 1, resolution (pure code, free).** Does each cited reference exist in the sources? Titles
are matched exactly (ignoring case and spacing) or fuzzily (token overlap of at least 0.6). A
fabricated citation, naming something that was never in the material, comes out
`unresolved`. An unresolved citation is never sent to pass 2: asking a model whether a source
nobody can find supports a claim would produce a confident answer about nothing.

**Pass 2, support (one model call per claim).** For claims with at least one resolved citation,
the judge reads the cited text and returns:

| field | meaning |
|---|---|
| `verdict` | `supported` (states or directly implies the claim), `partial` (related and consistent, but does not carry it all), `unsupported` |
| `why` | one sentence naming the deciding detail |
| `citation_spans` | for each citation, the phrase of OUR writing it bears on (`span_quote`) and how that sentence relates to it (`span_relation`) |
| `relation`, `strength` | the claim-level relation verb and a 0-100 grade (judgements, not evidence) |
| `evidence_spans` | the passage of the SOURCE that carries the claim (`support_span`) |
| `citation_bears` | one own-words sentence per source on how it bears on the claim |

Then everything the model said is checked by code, fail-closed:

- A span quote is kept only if it occurs **exactly once in exactly one field** of the
  writing: the claim, its why-it-matters, or one detail (a field may hold several sentences).
  Case, whitespace and smart punctuation are forgiven; paraphrase is not. The stored quote is
  sliced from our text, never copied from the model's. Every span records its producer.
- A span that lands on the thesis may move to the supporting sentence the source overlaps most
  (at least 3 shared content words and a quarter of the sentence, no ties). A moved span drops
  the judge's relation, because the judge never rated that sentence.
- An evidence passage must be found in the captured, digest-pinned source document. An
  ambiguous quote may grow by up to 1,536 characters of context to become unique; otherwise it
  is refused rather than guessed.
- A `supported` verdict with no located passage is recorded as `unresolved`, not supported. A
  `partial` is never recorded as `unsupported`, unless the judge also named the relation
  `contradicts` (see below). A missing or unreadable verdict is `unchecked` or `unknown`, never a
  pass.

Claim verdicts: `supported`, `partial`, `unsupported`, `unresolved` (pass 1 found nothing),
`unchecked` (no usable verdict). The brief's overall status is one of `ok`, `unsupported`,
`unresolved`, `partly-checked` (some claims checked, some not), `verifier-error`.

## Relations and the through line

| relation | meaning | through line | what a reader-facing page prints for that sentence |
|---|---|---|---|
| `says` | the source itself states it | states | no line state of its own: the sentence carries its claim's verdict; "supported" at line level only if a separate sentence-level check said so |
| `implies` `suggests` `connotes` `follows` `explains` `posits` `illustrates` | the writing takes a step beyond the source; the verb names the step | step-beyond | "our reading" (only when that citation resolved; otherwise the sentence is left out) |
| `contradicts` | the source disagrees | contradicts | left out, never printed; on the claim sentence itself the whole item is omitted |
| `undetermined` | the judge did not rate it | unrated | no line state of its own |
| anything else | | see below | see below |

Every relation word is read by one function (`read_relation`): trimmed, lower-cased, and the
aliases `stated`, `inferred`, `contradicted` read as `says`, `implies`, `contradicts` (a sentence
span records the alias it came from in `sentence_relation_alias`). Any other word is handled in
two ways, depending on where it appears:

- **From the judge** (a `span_relation` the judge returned): the word is **dropped** at intake.
  The span is kept without a relation, and that sentence then carries its claim's verdict like a
  `says` sentence. This is the composer's behaviour, kept on purpose: refusing would discard the
  location the judge did find.
- **In a brief handed to `line_states()`, `through_line()` or `line_rule()`** (a
  `sentence_relation` that did not come through intake): **refused** with `UnknownRelation`.

**A judge that names `contradicts` has said the source disagrees, and that wins over its verdict
word.** When the claim-level answer is `supported` or `partial` with the relation `contradicts`:

- the claim verdict is recorded **`unsupported`**; the verification block carries
  `verdict_override: {"rule": "relation-contradicts", "judge_verdict": "supported" | "partial"}`
  and its `why` begins "recorded unsupported: the judge named the relation contradicts";
- its evidence rows are `unsupported` with the rule id `EVIDENCE-RELATION-CONTRADICTS`, and keep
  the relation `contradicts` (sealed like every other row);
- no line of it prints: an `unsupported` claim is omitted from `line_states()`.

An `unsupported` verdict with `contradicts` stays as the judge gave it (rule
`EVIDENCE-MODEL-UNSUPPORTED`). Separately, a sentence whose own relation is `contradicts` is left
out whatever its claim's verdict. So `contradicts` never comes out supported or partly supported:
not as a printed line, not in the sealed record, not as a claim verdict. Because every guard uses
the same reader, `"contradicts "`, `"Contradicts"` and the alias `"contradicted"` all count.

This rule was measured, not assumed: in a preregistered agreement test on public refutation
items, 6 of 80 came back `partial` + `contradicts`. An earlier version recorded such claims as
`unchecked`, which hid real refutations.

## The seal and its limits

`judge()` returns a claim-evidence record: one outcome row per claim per citation, holding only
identities (offsets, SHA-256 digests, counts, verbs), never source text. The record is sealed:

```
chain_0 = sha256(canonical({"contract": "claim-evidence/v1", "outcome_count": N}))
chain_k = sha256(chain_{k-1} + canonical({"outcome": row_k, "stage_a_binding": binding_k}))
```

**What it covers:** the rows in `claim_evidence.outcomes`, each hashed whole: the claim it
belongs to (by a digest of the claim text, recorded in the row but NOT re-checked against the
brief by `verify_seal()`, so editing a claim's text in the brief still verifies), the citation,
the support state and its reason, every evidence span, the relation verb, `relation_source`
(whether the judge chose the verb or it was defaulted), strength, corroboration and the source
binding. Editing a row, upgrading a verb, appending, deleting or reordering rows, or relabelling
a defaulted verb as the judge's own all break the chain, and `verify_seal()` returns False.

**What it does not cover:** everything else in a `judge()` result. That includes the claim
verdicts and their `why` in `result["brief"]`, the per-sentence spans with their relations and
through-line labels, the line states a page derives from them, the run record in
`result["verification"]`, and the record's own `verification.rule_ids` and
`capture_time_result`. Changing any of those leaves `verify_seal()` True. Keep the whole result
under version history if you need those fields witnessed too.

**The limit, plainly:** the chain has no key. Anyone who can rewrite a row can recompute the
chain and write the new value beside it, and `verify_seal()` will then pass (a test pins this).
The seal is tamper-evident only alongside version history: commit the record, or publish its
seal value somewhere you do not control alone, and a later rewrite shows up as a change to a
value that was already witnessed. It is not authentication and says nothing about who wrote the
record. Sources are pinned the same way: their SHA-256 digests (raw, and after the `norm/v1`
normalizer) are taken when you hand them to `judge()`, and a source that drifts after an earlier
capture is refused. `normalizer_sha256` is the digest of this library's own normalizer source
file (`claim_seal/normalize.py`), so it differs from the digest the original composer records
for `norm/v1`, although the transformation is the same.

## Install

Nothing to install for the library, tests or demo: run from the repository root. To get the
`claim-seal` command: `python3 -m venv .venv && .venv/bin/pip install -e .`

## API

```python
from claim_seal import resolve, judge, seal, verify_seal, through_line, line_states
from claim_seal.adapters import ReplayJudge, OpenAICompatible

report = resolve(brief, sources)                  # pass 1 only, no model
result = judge(brief, sources, model=OpenAICompatible(model="your-model"))
result["status"]                                  # ok / unsupported / unresolved / partly-checked / verifier-error
result["brief"]                                   # annotated copy: every claim has "verification"
result["claim_evidence"]                          # the sealed record
verify_seal(result["claim_evidence"])             # True
through_line("suggests")                          # "step-beyond"
line_states(result["brief"])                      # what a page may print, line by line
```

`model` is any callable `prompt -> text`; if it accepts a `timeout` keyword it gets the remaining
budget. Return `claim_seal.ModelText(text, model="name")` to record which model actually answered.

`judge()` options (defaults are the composer's production settings): `sharded=True` (one call per
claim; required for per-sentence relations), `spans=True`, `local_placement=True`,
`evidence=True`, `completion=False` (try one more source item for `partial` claims, verdict
`supported-completed`), `locating_quote=False`, `pool_size=2`, `claim_deadline_s=120`.

Also: `resolve_only()` (pass 1 for writing that exaggerates on purpose: verdicts `cites-real` or
`unresolved`, never `supported`), `capture_sources()`, `source_lineage()`, `seal()`,
`seal_problem()`, `derive_coarse_relation()`, `line_rule()`.

**Inputs.** A brief: `{"items": [{"insight": "...", "why_it_matters": "...", "detail": ["..."],
"sources": [{"id": "source-id", "ref": "cited title"}], "analysis": [...]}], "connection": {...}}`.
Sources: `{"source-id": {"label": "...", "items": [{"title": "...", "text": "...", "url": "...",
"body": "...", "retrieved_at": "..."}]}}`. A citation names an item by title; pass 2 reads title
and text (first 700 characters); `body` is the full captured document evidence is quoted from
(defaults to `text`). See `examples/`.

## CLI

```
python3 -m claim_seal check claims.json sources.json --judge replay:answers.json --out result.json
python3 -m claim_seal check claims.json sources.json --judge openai:MODEL_NAME
python3 -m claim_seal check claims.json sources.json          # pass 1 only
python3 -m claim_seal verify-seal result.json                 # exit 0 ok, 1 fail, 2 nothing
```

Options: `--whole-brief`, `--no-spans`, `--no-local-placement`, `--no-evidence`, `--completion`,
`--pool N`, `--deadline S`.

## Demo (offline)

`examples/` holds three short self-written source texts about an invented town, a brief with one
claim of each kind (stated, step beyond, contradicted, fabricated citation) and recorded judge
answers. The replay adapter returns those answers; it does not judge. Everything after the
answers (resolution, span checks, evidence location, line states, the seal) runs for real.

```
python3 -m claim_seal check examples/brief.json examples/sources.json \
    --judge replay:examples/replay-answers.json --out /tmp/result.json
python3 -m claim_seal verify-seal /tmp/result.json                         # OK
python3 examples/tamper.py /tmp/result.json /tmp/tampered.json upgrade
python3 -m claim_seal verify-seal /tmp/tampered.json                       # CHAIN FAIL
```

The expected output is `examples/demo-output.txt`. With a real model, using any OpenAI-compatible
endpoint (the key is read from the environment at call time, never stored):

```
export OPENAI_API_KEY=...                          # or pass api_key_env= in code
export CLAIM_SEAL_BASE_URL=https://api.openai.com/v1   # optional; any compatible endpoint
python3 -m claim_seal check examples/brief.json examples/sources.json --judge openai:MODEL_NAME
```

A real model will not reproduce the recorded answers word for word.

## Tests

```
python3 -m unittest discover -s tests -t tests
python3 tools/scan_private.py        # generic checks: server paths, key/token and id shapes
CLAIM_SEAL_DENYLIST=/path/outside/the/repo.txt python3 tools/scan_private.py   # plus your own terms
python3 tools/scan_private.py --history                                       # every past commit too
sh tools/ci.sh                       # both, plus the demo
```

## Lenses

`lenses/raised-eyebrow.txt` is one of the composer's writing lenses, published as an example. A
lens shapes the writing, not the check; see `lenses/README.md`.

## What changed in extraction

The checking logic is the composer's. Its deployment flags became keyword arguments; its source
registry became the plain sources format above; private identifiers were removed. Behaviour
changes, each covered by a test: a `supported` or `partial` verdict paired with the relation
`contradicts` is recorded `unsupported` (claim and evidence rows, the override named);
every relation word is read by one shared reader (trimmed, with the `contradicted` alias) at the
claim guard, the evidence pass and the per-sentence intake; `judge()` returns an annotated copy
instead of editing its input; source digests are taken when sources are handed in rather than
by an HTTP fetcher. The composer's title-lineage checks for specific feed formats, its service
checkpoints and its per-call usage telemetry are not included.

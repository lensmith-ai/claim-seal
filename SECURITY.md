# Security policy

## Supported versions

claim-seal has one release line. Only the newest release gets security fixes.

| version | supported |
|---|---|
| 0.1.x (newest) | yes |
| anything older | no |

## Reporting a vulnerability

**Do not open a public issue for a security problem.** Report it privately instead, through
GitHub's private vulnerability reporting: on
[github.com/lensmith-ai/claim-seal](https://github.com/lensmith-ai/claim-seal), open the
**Security** tab and choose **Report a vulnerability**. Only the maintainers can read the report.

Please include:

- the claim-seal version or commit;
- what an attacker controls (the brief, the sources, the model endpoint's replies, the CLI
  arguments) and what they gain;
- the smallest input that shows it, written for the report.

Never put real credentials, API keys or private source material in a report or an issue. Use a
placeholder key (`TEST_ONLY_KEY`) and invented text, as the tests in this repository do.

## What to expect

This is a small project with one maintainer, so responses are best effort. The aim is to
acknowledge a report within 7 days, agree on the problem and a fix plan after that, and publish
the fix together with a GitHub security advisory that credits you, unless you ask not to be
named. Please give us a reasonable chance to ship the fix before you disclose the details.

## Known limits that are not vulnerabilities

The README lists these in "What it does not do"; reports about them are still welcome if you
find a way past the stated limit:

- The seal is an unkeyed SHA-256 chain: tamper-evident alongside version history, not a
  signature. Rewriting a record and recomputing its chain is expected to pass `verify_seal()`.
- The support judge is a language model. Source text is wrapped as data and the judge is told
  never to follow it, but a source can still try to steer the judge, and a steered judge that
  quotes a real passage of the source as support is not caught.
- claim-seal checks writing only against the sources it is given; it does not fetch or vet them.

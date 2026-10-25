# Releasing claim-seal

Some release protections are repository settings or signing steps, not files, so no commit can
switch them on. Each step below is the owner's, and each names the receipt that shows it is done:
paste that command's output into the release record. Do not open the repository to the public, or
publish a release, while a receipt is missing.

## Before the repository is public

1. **Private vulnerability reporting is on** (SECURITY.md sends reporters there).
   Settings, Security, "Private vulnerability reporting": Enable.
   Receipt: `gh api repos/lensmith-ai/claim-seal/private-vulnerability-reporting` prints
   `{"enabled":true}`.
2. **Secret scanning and push protection are on.**
   Settings, Security, "Secret protection": enable secret scanning and push protection.
   Receipt: `gh api repos/lensmith-ai/claim-seal --jq .security_and_analysis` shows
   `secret_scanning` and `secret_scanning_push_protection` with `"status": "enabled"`.
3. **CodeQL has run green on main.** `.github/workflows/codeql.yml` does this on every push.
   Receipt: `gh run list --workflow codeql.yml --branch main --limit 1` shows `completed success`,
   and `gh api repos/lensmith-ai/claim-seal/code-scanning/alerts --jq 'map(select(.state=="open"))|length'`
   prints `0`.
4. **CI is green on main**, both jobs (`check` on Python 3.10 and 3.12, and `package`).
   Receipt: `gh run list --workflow ci.yml --branch main --limit 1` shows `completed success`.

## For each release

5. **Build from a clean checkout and keep the checksums.**
   `sh tools/build_check.sh dist` builds the sdist and wheel with the hash-pinned setuptools,
   installs the wheel into an empty venv, runs the demo, and writes `dist/SHA256SUMS`.
   Receipt: its last line `build-check: ... demo matched` and the `SHA256SUMS` file.
6. **Sign an annotated tag on the released commit.**
   `git tag -s v0.1.0 -m "claim-seal 0.1.0"` and push the tag.
   Receipt: `git tag -v v0.1.0` reports a good signature from the owner's key.
7. **Publish the artifacts with their checksums and a provenance attestation.**
   Attach both files and `SHA256SUMS` to the GitHub release; create an attestation for each file
   (GitHub artifact attestations, `actions/attest-build-provenance` in a release workflow, or
   `gh attestation` tooling).
   Receipt: `gh attestation verify dist/claim_seal-0.1.0-py3-none-any.whl --repo lensmith-ai/claim-seal`
   succeeds, and `sha256sum -c SHA256SUMS` passes on the downloaded files.

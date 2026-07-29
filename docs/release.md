# Release procedure

This procedure releases the Python distribution. Use the
[artifact promotion and rollback playbooks](operations.md#playbook-qualify-and-promote-an-artifact)
for model artifacts.

1. Update `CHANGELOG.md` and the project version.
2. Regenerate committed schemas:

   ```bash
   PYTHONPATH=src python scripts/generate_schemas.py
   git diff --exit-code schema
   ```

3. Refresh the lock on the release platform with `python -m pip lock . -o
   pylock.toml`.
4. Run `scripts/quality.sh`.
5. Run the scheduled-equivalent five-seed benchmark with `--enforce` and
   compare the committed latency baseline. Active sampling must improve median
   reference-provider audit MAE by at least 10% over equal-budget LHS, every
   active run must stay below the latency ceiling, and pooled active audit
   coverage must remain within the configured 85–95% range. Median latency
   regression may not exceed 20%.

   ```bash
   pricing-mapper benchmark \
     --config config.example.toml \
     --output benchmark-results.json \
     --baseline benchmarks/baseline.json \
     --enforce
   ```
6. Build and inspect distributions:

   ```bash
   python -m build
   python -m twine check dist/*
   ```

7. Run `scripts/smoke.sh`; it creates an isolated environment, installs the
   wheel, validates config, maps, validates the artifact, predicts, and
   recomputes audit metrics.
8. Review dependency audit findings and document any accepted risk.
9. Tag `vX.Y.Z`. The release workflow rebuilds and verifies distributions.
10. Publish with trusted publishing only after the tag workflow passes. Never
    upload a locally modified artifact.

Rollback is a new patch release or restoration of a previously signed wheel
and model artifact. sklearn artifacts are not migrated across dependency
versions; retrain them. Follow the
[release and rollback playbook](operations.md#playbook-release-and-rollback)
to restore the wheel, runtime, and model artifact as one compatible unit.

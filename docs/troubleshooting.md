# Troubleshooting

For complete procedures, exit-code routing, and recovery criteria, see the
[operational playbooks](operations.md).

## “v1 configuration must be TOML”

JSON configuration was removed. Copy values into `config.example.toml` and
validate it. Unknown legacy options are intentionally not translated.

## “v0 pickle/state artifacts are intentionally rejected”

Do not rename or load the pickle. Pickle can execute code. If the old run has a
CSV of observations, validate and import it with `map run --seed-data`.

## Resume fingerprint mismatch

Resume requires the same complete configuration, provider identity, resolved
domain, Python version, and numerical/model dependency versions. Use the
original TOML and environment. Start a new run ID for a changed budget, model
policy, provider, domain, or runtime.

## Run already locked

Another live process holds the same run ID. OS locks are released when a
process exits; the small lock file may remain and is harmless. Do not delete a
lock to bypass a live owner.

## Provider unavailable/rejected

Only `ProviderUnavailable` is retried. Increase bounded retry settings if the
local provider is transiently unavailable. `ProviderRejected`, invalid output,
and unexpected exceptions stop the run with the pending row intact.

## No model meets the latency ceiling

Increase `model.max_p95_latency_ms`, lower tree/iteration counts, or run on the
deployment-class hardware. Candidate latency is measured warm, one row at a
time.

## Audit coverage gate failed

The artifact remains inspectable and includes a warning, but is not promotion
eligible. Increase calibration/audit budgets, examine distribution shift and
risk slices, and rerun under a new ID. Do not tune directly to audit outcomes.

## Artifact integrity/version mismatch

Restore an unmodified artifact directory. Every listed file is hashed and
unlisted files are rejected. sklearn cross-version loading is unsupported;
install the exact version in `dependencies.json` or retrain.

## Ollama model verification failed

Hybrid runs check the local model before any provider calls. Confirm the
container is reachable only at `127.0.0.1:11434`, then rerun
`./scripts/provision-ollama.sh`. Do not shorten or update the configured digest.
A missing model, wrong `Q4_K_M` quantization, changed Ollama version on resume,
or digest mismatch requires correction before the same run can continue.

## Advisor failed closed

The mapper retries a malformed, oversized, timed-out, or unavailable advisory
response twice. After the third failure it deliberately leaves no new mapping
batch and consumes no provider quotes. Preserve the journal, restore the exact
pinned runtime, and use `map resume`. Do not replace the advisor decision with a
handwritten policy.

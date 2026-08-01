# Security policy

## Supported versions

Security fixes are provided for the latest v1 minor/patch release.

## Security model

- Providers are trusted local code and execute with the caller’s permissions.
  The library does not sandbox them.
- v1 never loads pickle. `skops` files are hash checked, constrained to the
  expected sklearn estimator family, and loaded only with a fixed code-owned
  type allow-list.
- Manifest hashes provide integrity against corruption, not authenticity
  against an attacker able to replace the whole directory. Use external
  signing, access control, and trusted storage.
- Artifact paths reject traversal-like run IDs, publish by atomic rename, and
  refuse overwrite. SQLite and OS locks protect consistency.
- Telemetry omits quote payloads and provider exception messages. Dataset
  artifacts necessarily contain sampled quote inputs and premiums; protect
  them according to their sensitivity.
- Cross-version sklearn loading is rejected.
- The optional Ollama advisor is untrusted and cannot quote, read mapper
  outputs, or access provider credentials. It receives only strict aggregate
  diagnostics and can select only code-owned policies and existing bin IDs.
  Runtime and model digests are verified; invalid or unavailable responses fail
  closed before batch registration.

Keep dependencies patched, run `pip-audit`, review custom provider code, and
do not weaken hash, schema, or trust checks to recover a damaged artifact.

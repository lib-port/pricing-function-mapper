#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
COMPOSE_FILE="$PROJECT_DIR/deploy/ollama/compose.yaml"
MODEL_NAME="granite4.1:3b"
MODEL_DIGEST="sha256:6fd349357287c7ffc9e38189a93b48ea175d24fc566b38f09cfc564fb7f303eb"
PROVISION_ENDPOINT="http://127.0.0.1:11435"
RUNTIME_ENDPOINT="http://127.0.0.1:11434"

cleanup() {
  docker compose -f "$COMPOSE_FILE" --profile provision stop ollama-provision >/dev/null 2>&1 || true
  docker compose -f "$COMPOSE_FILE" --profile provision rm -f ollama-provision >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker compose -f "$COMPOSE_FILE" stop ollama >/dev/null 2>&1 || true
docker compose -f "$COMPOSE_FILE" --profile provision up -d ollama-provision

ready=0
for _attempt in $(seq 1 60); do
  if curl --fail --silent --show-error "$PROVISION_ENDPOINT/api/version" >/dev/null; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "$ready" != "1" ]]; then
  echo "Ollama provisioning endpoint did not become ready" >&2
  exit 1
fi

curl --fail --silent --show-error \
  --header "Content-Type: application/json" \
  --data "{\"model\":\"$MODEL_NAME\",\"stream\":false}" \
  "$PROVISION_ENDPOINT/api/pull" >/dev/null

TAGS_FILE=$(mktemp)
trap 'rm -f "$TAGS_FILE"; cleanup' EXIT
curl --fail --silent --show-error "$PROVISION_ENDPOINT/api/tags" >"$TAGS_FILE"
python3 - "$TAGS_FILE" "$MODEL_NAME" "$MODEL_DIGEST" <<'PY'
import json
import sys

path, expected_name, expected_digest = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)
matches = [
    model
    for model in payload.get("models", [])
    if model.get("name") == expected_name or model.get("model") == expected_name
]
if len(matches) != 1:
    raise SystemExit(f"expected exactly one installed {expected_name!r} model")
model = matches[0]
digest = model.get("digest", "")
if not digest.startswith("sha256:"):
    digest = f"sha256:{digest}"
if digest != expected_digest:
    raise SystemExit(f"model digest mismatch: expected {expected_digest}, got {digest}")
if model.get("details", {}).get("quantization_level") != "Q4_K_M":
    raise SystemExit("installed model is not Q4_K_M")
print(f"verified {expected_name} {expected_digest} Q4_K_M")
PY

cleanup
trap - EXIT
rm -f "$TAGS_FILE"
docker compose -f "$COMPOSE_FILE" up -d ollama

ready=0
for _attempt in $(seq 1 60); do
  if curl --fail --silent --show-error "$RUNTIME_ENDPOINT/api/version" >/dev/null; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "$ready" != "1" ]]; then
  echo "Offline Ollama runtime did not become ready" >&2
  exit 1
fi
echo "Offline Ollama runtime is ready at $RUNTIME_ENDPOINT"

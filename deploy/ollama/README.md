# Pinned local Ollama advisor

Provision the model once while registry access is available:

```bash
./scripts/provision-ollama.sh
```

The command uses the pinned Ollama `0.32.0` multi-platform image, pulls
`granite4.1:3b`, and refuses to continue unless the model-list API reports the
full digest
`sha256:6fd349357287c7ffc9e38189a93b48ea175d24fc566b38f09cfc564fb7f303eb`
and `Q4_K_M`. It then removes the provisioning container and starts the
runtime container on an internal Docker network with cloud features disabled.

The runtime is bound only to `127.0.0.1:11434`, limited to two CPUs, 8 GiB RAM,
and one parallel request. Its only persistent mount is the named model-cache
volume: mapper outputs, provider code, environment files, and credentials are
not mounted.

To stop or start the already-provisioned offline runtime:

```bash
docker compose -f deploy/ollama/compose.yaml stop ollama
docker compose -f deploy/ollama/compose.yaml up -d ollama
```

# Load Mimi Safetensors With Key Mapping

Status: ready-for-agent

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Load the prepared `mimi-pytorch-e351c8d8@125.safetensors` artifact into the repo-owned MLX Mimi Runtime by applying the key and tensor-layout mapping required by the Moshi Swift reference.

This issue is about turning a prepared Mimi artifact into populated MLX module parameters. It should not yet wire the runtime into the file-translation UI path.

## Acceptance criteria

- [ ] The loader reads the prepared `mimiWeights` artifact from `PreparedModelArtifacts`.
- [ ] The loader validates by artifact role, expected keys, and expected shapes rather than hardcoding the current artifact filename.
- [ ] Safetensors arrays are loaded using MLX Swift APIs.
- [ ] Key mapping covers the Moshi reference transformations for `encoder.model`, `decoder.model`, transformer projection keys, residual/downsample/upsample layers, and block indices.
- [ ] Tensor layout mapping covers convolution and transposed-convolution weights as needed by MLX.
- [ ] Missing, unexpected, or shape-incompatible weights surface as explicit user-visible Mimi load errors.
- [ ] Tests cover key mapping and tensor layout mapping using small fake arrays where possible.
- [ ] Porting notes reference `ref/moshi-swift/MoshiCLI/RunMimi.swift` `makeMimi(numCodebooks:)`.

## Notes

- Preserve the loaded q/codebook configuration from the runtime shell; do not infer random defaults from the artifact filename.
- The current manifest filename is an artifact locator, not the semantic compatibility check. Compatibility is determined by the Mimi Runtime contract, mapped keys, and tensor shapes.
- Keep deterministic encoder/decoder default until streaming encode/decode are implemented.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/14-port-mimi-module-structure-from-moshilib.md`

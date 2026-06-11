# Load Mimi Safetensors With Key Mapping

Status: done

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Load the prepared `mimi-pytorch-e351c8d8@125.safetensors` artifact into the repo-owned MLX Mimi Runtime by applying the key and tensor-layout mapping required by the Moshi Swift reference.

This issue is about turning a prepared Mimi artifact into populated MLX module parameters. It should not yet wire the runtime into the file-translation UI path.

## Acceptance criteria

- [x] The loader reads the prepared `mimiWeights` artifact from `PreparedModelArtifacts`.
- [x] The loader validates by artifact role, expected keys, and expected shapes rather than hardcoding the current artifact filename.
- [x] Safetensors arrays are loaded using MLX Swift APIs.
- [x] Key mapping covers the Moshi reference transformations for `encoder.model`, `decoder.model`, transformer projection keys, residual/downsample/upsample layers, and block indices.
- [x] Tensor layout mapping covers convolution and transposed-convolution weights as needed by MLX.
- [x] Missing, unexpected, or shape-incompatible weights surface as explicit user-visible Mimi load errors.
- [x] Tests cover key mapping and tensor layout mapping using small fake arrays where possible.
- [x] Porting notes reference `ref/moshi-swift/MoshiCLI/RunMimi.swift` `makeMimi(numCodebooks:)`.

## Verification

- `swift test` passes with 65 tests.
- `xcodebuild build -project S2STranslate.xcodeproj -scheme S2STranslate -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO` passes.

## Porting notes

- `MLXMimiWeightLoader` locates the prepared artifact by semantic role `mimiWeights`, not by the current `mimi-pytorch-e351c8d8@125.safetensors` filename.
- The default loader uses MLX Swift `loadArrays(url:)` to read safetensors, then maps flattened source keys into MoshiLib-compatible MLX parameter names.
- The key mapping follows `ref/moshi-swift/MoshiCLI/RunMimi.swift` `makeMimi(numCodebooks:)`: `encoder.model`/`decoder.model` prefix stripping, transformer projection and gating names, encoder residual/downsample indices, decoder upsample/residual indices, init/final conv names, and block-index compaction.
- Tensor layout mapping follows the same reference: convolution, `input_proj`, and `output_proj` weights swap their last two axes; transposed-convolution weights use axes `[1, 2, 0]`.
- Unit tests use shape-only fake tensors because constructing real `MLXArray` values in `swift test` requires the MLX Metal library to be available in the test process. Production loading still carries the real `MLXArray` payload and exposes `LoadedMLXMimiWeights.moduleParameters()` for handoff to MLX modules.
- Applying `ModuleParameters` directly to weight-bearing Mimi `MLXNN.Module` classes is intentionally deferred until the module shells from issue 14 become concrete MLX modules in the next runtime/model slice.

## Notes

- Preserve the loaded q/codebook configuration from the runtime shell; do not infer random defaults from the artifact filename.
- The current manifest filename is an artifact locator, not the semantic compatibility check. Compatibility is determined by the Mimi Runtime contract, mapped keys, and tensor shapes.
- Keep deterministic encoder/decoder default until streaming encode/decode are implemented.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/14-port-mimi-module-structure-from-moshilib.md`

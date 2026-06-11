# Load Mimi Model With MLX Swift

Status: ready-for-agent

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Introduce the first real Mimi runtime loader using MLX Swift and the prepared `mimi-pytorch-e351c8d8@125.safetensors` artifact. The goal is to construct a stateful Mimi codec object behind the existing encoder/decoder protocol boundaries without replacing the deterministic test doubles yet.

This issue is about model construction, weight loading, configuration checks, and lifecycle errors. Streaming encode/decode behavior is handled by follow-up issues.

## Acceptance criteria

- [ ] A Mimi runtime loader can locate the prepared Mimi artifact by role.
- [ ] The loader initializes MLX Swift structures needed by the Mimi codec.
- [ ] The loader validates expected sample rate, frame rate, codebook count, and artifact availability.
- [ ] Load failures surface as explicit user-visible codec errors.
- [ ] The deterministic encoder/decoder remain available for tests.
- [ ] Tests cover missing artifact, incompatible artifact role, and successful load using a small fake or metadata-only seam where full MLX load is impractical.
- [ ] Implementation references `ref/moshi-swift/MoshiLib/Mimi.swift` and keeps porting notes in comments or docs where behavior diverges.

## Notes

- Do not wire this into the UI as the default until encode/decode can run.
- Keep the loaded Mimi runtime reusable by both encoder and decoder follow-up slices.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/12-implement-real-huggingface-artifact-download-cache.md`


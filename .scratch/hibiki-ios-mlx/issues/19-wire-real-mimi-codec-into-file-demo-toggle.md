# Wire Real Mimi Codec Into File Demo Behind Toggle

Status: ready-for-agent

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Add an explicit file-demo runtime selection that can use the MLX Mimi encoder and decoder while keeping the deterministic path available.

This issue should make it possible to test real Mimi encode/decode around the existing deterministic Hibiki session before the real Hibiki model is ready.

## Acceptance criteria

- [ ] The file-based demo can choose deterministic Mimi or MLX Mimi without changing test defaults.
- [ ] MLX Mimi runtime preparation uses already prepared artifacts from the Model Artifact Store.
- [ ] The real Mimi encoder and decoder wrappers share one loaded `MLXMimiRuntime` instance.
- [ ] Real Mimi encode/decode events flow through existing Experiment Session observations.
- [ ] The deterministic Hibiki session remains usable as a bridge for this slice.
- [ ] The flow is explicitly presented as real Mimi codec execution with deterministic Hibiki placeholder output, not real translation.
- [ ] UI copy clearly distinguishes deterministic model output from real Mimi codec execution.
- [ ] Failures from Mimi runtime load, encode, and decode remain user-visible.
- [ ] Tests cover orchestration with fake MLX Mimi runtime seams.

## Notes

- This is not a translation-quality milestone because Hibiki may still be deterministic here.
- This issue gives us a device smoke path for Mimi before taking on the 3B Hibiki model.
- Keep this issue before real Hibiki work so Mimi artifact load, PCM shape, token-frame shape, streaming state, and decoded PCM can be debugged independently.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/17-implement-mlx-mimi-streaming-encode.md`
- `.scratch/hibiki-ios-mlx/issues/18-implement-mlx-mimi-streaming-decode.md`

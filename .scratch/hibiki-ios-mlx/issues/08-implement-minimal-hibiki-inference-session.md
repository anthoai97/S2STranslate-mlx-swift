# Implement Minimal Hibiki Inference Session

Status: ready-for-agent

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Implement the first minimal Hibiki inference session. Source audio or source token events should enter the session, run through the loaded model pipeline, and emit target text events plus generated audio-token events. Sampling and generation settings should be explicit and conservative by default.

This issue should aim for a narrow, inspectable inference path that follows the Python reference structure closely enough for trace comparisons and future optimization.

## Acceptance criteria

- [ ] The session can initialize a minimal Hibiki pipeline from prepared model artifacts.
- [ ] Source token or audio-derived events can be fed into generation.
- [ ] The session emits text events when produced by the model.
- [ ] The session emits generated audio-token events suitable for the Mimi decode path.
- [ ] Sampling and voice-transfer-related settings, where supported, are represented as explicit configuration.
- [ ] Tests cover initialization, event flow, stop behavior, failure propagation, and deterministic comparison where practical.
- [ ] Reference trace comparison is used for shapes, event order, or token outputs where available.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/03-load-and-cache-model-artifacts.md`
- `.scratch/hibiki-ios-mlx/issues/04-add-reference-trace-harness.md`
- `.scratch/hibiki-ios-mlx/issues/06-implement-mimi-streaming-encode-path.md`
- `.scratch/hibiki-ios-mlx/issues/07-implement-mimi-streaming-decode-and-playback-path.md`

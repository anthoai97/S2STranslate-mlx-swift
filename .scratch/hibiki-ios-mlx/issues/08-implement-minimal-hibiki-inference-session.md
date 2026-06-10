# Implement Minimal Hibiki Inference Session

Status: complete

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Implement the first minimal Hibiki inference session. Source audio or source token events should enter the session, run through the loaded model pipeline, and emit target text events plus generated audio-token events. Sampling and generation settings should be explicit and conservative by default.

This issue should aim for a narrow, inspectable inference path that follows the Python reference structure closely enough for trace comparisons and future optimization.

## Acceptance criteria

- [x] The session can initialize a minimal Hibiki pipeline from prepared model artifacts.
- [x] Source token or audio-derived events can be fed into generation.
- [x] The session emits text events when produced by the model.
- [x] The session emits generated audio-token events suitable for the Mimi decode path.
- [x] Sampling and voice-transfer-related settings, where supported, are represented as explicit configuration.
- [x] Tests cover initialization, event flow, stop behavior, failure propagation, and deterministic comparison where practical.
- [x] Reference trace comparison is used for shapes, event order, or token outputs where available.

## Implementation notes

- Added `HibikiInferenceSession`, `HibikiGenerationConfiguration`, `HibikiInferenceDescription`, `HibikiInferenceStep`, `HibikiTextOutput`, and `HibikiInferenceEvent`.
- Added `DeterministicHibikiInferenceSession` as the first stateful Swift inference boundary. It initializes from prepared artifacts, consumes source Mimi token frames, emits text-token events, and emits generated target audio token frames suitable for the Mimi decode path.
- Added `HibikiTranslationExperimentBackend` to stream fixture PCM through artifact preparation, Mimi encode, Hibiki inference, Mimi decode, and buffered playback.
- Wired the proof-of-concept UI to use the deterministic Hibiki translation backend on **Start**.
- Extended session observations with Hibiki status, step count, text token count, visible text count, generated audio frame count, and sampling summary.
- Added tests for artifact-based initialization, source-token stepping, generated text/audio event flow, uninitialized-step failure, inference failure propagation, full backend flow, and reference trace comparison against `model:hibikiStep` plus text skip events.

This slice does not run the real MLX-backed Hibiki model yet. The deterministic inference session is a replaceable contract test double for the future model-backed implementation.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/03-load-and-cache-model-artifacts.md`
- `.scratch/hibiki-ios-mlx/issues/04-add-reference-trace-harness.md`
- `.scratch/hibiki-ios-mlx/issues/06-implement-mimi-streaming-encode-path.md`
- `.scratch/hibiki-ios-mlx/issues/07-implement-mimi-streaming-decode-and-playback-path.md`

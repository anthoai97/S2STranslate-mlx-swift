# Implement Mimi Streaming Encode Path

Status: complete

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Implement the first stateful Mimi encode path. PCM chunks should flow through a streaming encoder and produce token frames while preserving state across chunk boundaries. The session should expose encoded frame events and enough diagnostics to inspect frame cadence and boundary behavior.

This slice should remain focused on the encode path. Decoding and playback are handled by a later issue.

## Acceptance criteria

- [x] PCM chunks can be encoded incrementally into Mimi token frames.
- [x] Encoder state persists across chunks and does not reset at frame boundaries.
- [x] Encoded frame cadence can be inspected through logs, metrics, or session events.
- [x] Tests cover state continuity across multiple chunks.
- [x] Tests compare available output shapes, event order, or token traces against the reference trace harness.
- [x] Errors from unsupported sample rates, malformed chunks, or unavailable codec assets surface through session failure states.

## Implementation notes

- Added `MimiStreamingEncoder`, `MimiEncoderDescription`, `MimiTokenFrame`, `MimiEncodeEvent`, and `MimiEncodeExperimentBackend`.
- Added `DeterministicMimiStreamingEncoder` as the first stateful Swift encoder boundary. It accumulates PCM samples across chunk boundaries, emits one token frame per 1,920 samples at 24 kHz, and preserves frame index across calls.
- Wired the proof-of-concept UI to prepare demo artifacts, stream fixture PCM chunks, and emit deterministic Mimi token frames on **Start**.
- Extended session observations with Mimi encode status, encoded frame count, codebook count, token count, frame duration, and last frame index.
- Added tests for partial-chunk state continuity, session metrics, unsupported sample-rate failure, unavailable codec failure, and reference trace comparison against `reference-trace-small.json`.

This slice does not load or run the real MLX-backed Mimi neural codec yet. The deterministic encoder is intentionally a replaceable contract test double for the future model-backed implementation.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/01-create-experiment-session-skeleton.md`
- `.scratch/hibiki-ios-mlx/issues/04-add-reference-trace-harness.md`
- `.scratch/hibiki-ios-mlx/issues/05-build-streaming-audio-input-slice.md`

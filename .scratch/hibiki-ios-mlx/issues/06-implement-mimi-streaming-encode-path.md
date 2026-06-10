# Implement Mimi Streaming Encode Path

Status: ready-for-agent

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Implement the first stateful Mimi encode path. PCM chunks should flow through a streaming encoder and produce token frames while preserving state across chunk boundaries. The session should expose encoded frame events and enough diagnostics to inspect frame cadence and boundary behavior.

This slice should remain focused on the encode path. Decoding and playback are handled by a later issue.

## Acceptance criteria

- [ ] PCM chunks can be encoded incrementally into Mimi token frames.
- [ ] Encoder state persists across chunks and does not reset at frame boundaries.
- [ ] Encoded frame cadence can be inspected through logs, metrics, or session events.
- [ ] Tests cover state continuity across multiple chunks.
- [ ] Tests compare available output shapes, event order, or token traces against the reference trace harness.
- [ ] Errors from unsupported sample rates, malformed chunks, or unavailable codec assets surface through session failure states.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/01-create-experiment-session-skeleton.md`
- `.scratch/hibiki-ios-mlx/issues/04-add-reference-trace-harness.md`
- `.scratch/hibiki-ios-mlx/issues/05-build-streaming-audio-input-slice.md`

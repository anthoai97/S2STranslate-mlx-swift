# Implement Mimi Streaming Decode and Playback Path

Status: ready-for-agent

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Implement the first stateful Mimi decode path and connect it to a playback-oriented output stream. Token frames should decode into audio chunks that can be written to a test sink and, in the proof-of-concept app, routed to a simple playback sink when running on a capable device.

The slice should demonstrate generated audio output as a stream, not as a batch created after all tokens are available.

## Acceptance criteria

- [ ] Mimi token frames can be decoded incrementally into audio chunks.
- [ ] Decoder state persists across streamed token frames.
- [ ] A test sink can receive decoded audio chunks without using device audio hardware.
- [ ] The app can route decoded chunks to a proof-of-concept playback sink.
- [ ] Tests cover decode event order, state continuity, stop behavior, and failure propagation.
- [ ] Available decode shapes, cadence, or fixtures are compared against the reference trace harness.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/01-create-experiment-session-skeleton.md`
- `.scratch/hibiki-ios-mlx/issues/04-add-reference-trace-harness.md`
- `.scratch/hibiki-ios-mlx/issues/06-implement-mimi-streaming-encode-path.md`

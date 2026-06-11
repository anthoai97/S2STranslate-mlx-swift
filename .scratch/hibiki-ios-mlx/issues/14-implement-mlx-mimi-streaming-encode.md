# Implement MLX Mimi Streaming Encode

Status: ready-for-agent

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Replace the deterministic Mimi encoder in the real file-translation path with an MLX-backed streaming Mimi encoder. The encoder should accept mono 24 kHz PCM chunks, preserve codec state across calls, and emit source Mimi token frames at the expected 12.5 Hz / 80 ms cadence.

This is the first real source-audio tokenization slice.

## Acceptance criteria

- [ ] A real `MimiStreamingEncoder` implementation wraps the loaded Mimi runtime.
- [ ] PCM chunks are encoded incrementally without resetting state at chunk boundaries.
- [ ] Emitted `MimiTokenFrame` values preserve frame index, timestamp, codebook count, and source audio frame metadata.
- [ ] The encoder rejects unsupported sample rates and malformed chunks with clear errors.
- [ ] Session observations continue to report encoded frame count, codebook count, token count, and cadence.
- [ ] Tests cover chunk-boundary state continuity using a fake runtime or tiny fixture seam.
- [ ] Reference trace comparison is added or updated for emitted frame shapes and cadence where deterministic parity is available.

## Notes

- Follow `ref/moshi-swift/MoshiLib/Mimi.swift` for streaming state and frame cadence.
- Keep `DeterministicMimiStreamingEncoder` available for fast unit tests.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/13-load-mimi-model-with-mlx-swift.md`


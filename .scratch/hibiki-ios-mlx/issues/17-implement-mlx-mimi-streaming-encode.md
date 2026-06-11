# Implement MLX Mimi Streaming Encode

Status: ready-for-agent

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Replace the deterministic Mimi encoder in the real file-translation path with an MLX-backed streaming Mimi encoder behind the existing `MimiStreamingEncoder` protocol.

The encoder should accept mono 24 kHz PCM chunks, preserve codec state across calls, and emit source Mimi token frames at the expected 12.5 Hz / 80 ms cadence.

## Acceptance criteria

- [ ] A real `MimiStreamingEncoder` implementation wraps the loaded MLX Mimi Runtime.
- [ ] PCM chunks convert to MLX arrays with shape compatible with Moshi's `StreamArray(MLXArray(pcm)[.newAxis, .newAxis])` pattern.
- [ ] PCM chunks are encoded incrementally without resetting state at chunk boundaries.
- [ ] Empty `StreamArray` outputs are handled without emitting bogus `MimiTokenFrame` values.
- [ ] When a PCM chunk does not produce a valid Mimi output yet, `encode(_:)` returns an empty frame array rather than padding or placeholder tokens.
- [ ] Emitted `MimiTokenFrame` values preserve frame index, timestamp, codebook count, and source audio frame metadata.
- [ ] Unsupported sample rates and malformed chunks fail with clear `MimiEncodeError` values.
- [ ] Session observations continue to report encoded frame count, codebook count, token count, and cadence.
- [ ] Tests cover chunk-boundary state continuity using a fake runtime or tiny fixture seam.
- [ ] Reference trace comparison is added or updated for emitted frame shapes and cadence where deterministic parity is available.

## Notes

- Follow `ref/moshi-swift/MoshiLib/Mimi.swift` `encodeStep(_:)` and `ref/moshi-swift/MoshiCLI/RunMimi.swift` streaming usage.
- Keep `DeterministicMimiStreamingEncoder` available for fast unit tests.
- Do not feed invented source token frames to Hibiki. Buffered Mimi state should be represented as no emitted frame.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/16-validate-mimi-runtime-metadata-and-warmup.md`

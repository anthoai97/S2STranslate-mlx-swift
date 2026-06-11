# Implement MLX Mimi Streaming Decode

Status: ready-for-agent

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Replace the deterministic Mimi decoder in the real translation path with an MLX-backed streaming Mimi decoder. The decoder should accept generated target audio token frames from Hibiki and emit PCM chunks incrementally so playback can begin before the full translation is complete.

## Acceptance criteria

- [ ] A real `MimiStreamingDecoder` implementation wraps the loaded Mimi runtime.
- [ ] Update `MimiStreamingDecoder.decode(_:)` to return `[DecodedAudioChunk]` so real streaming decode can emit zero, one, or multiple chunks per token frame.
- [ ] Generated audio token frames decode incrementally without resetting decoder state.
- [ ] Empty `StreamArray` outputs are handled without sending silence or placeholder chunks to playback.
- [ ] Decoded chunks preserve sample rate, frame index, timestamp, duration, and source token-frame metadata.
- [ ] Decode failures surface through `MimiDecodeError` and the Experiment Session failure path.
- [ ] Session observations continue to report decoded chunk count, decoded sample rate, decoded duration, and last decoded frame.
- [ ] Tests cover state continuity, bad token shape failure, and sink delivery using fake runtime seams where full MLX decode is impractical.
- [ ] Reference trace comparison is added or updated for decoded audio chunk event order and cadence.

## Notes

- Follow `ref/moshi-swift/MoshiLib/Mimi.swift` for decode state behavior.
- Keep `DeterministicMimiStreamingDecoder` available for fast tests.
- Do not send invented silence to playback when the decoder has not produced PCM yet.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/16-validate-mimi-runtime-metadata-and-warmup.md`
- `.scratch/hibiki-ios-mlx/issues/23-implement-hibiki-token-sampling-and-text-output.md`

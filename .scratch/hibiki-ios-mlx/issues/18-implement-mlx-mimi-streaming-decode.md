# Implement MLX Mimi Streaming Decode

Status: done

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Replace the deterministic Mimi decoder in the real translation path with an MLX-backed streaming Mimi decoder. The decoder should accept generated target audio token frames from Hibiki and emit PCM chunks incrementally so playback can begin before the full translation is complete.

## Acceptance criteria

- [x] A real `MimiStreamingDecoder` implementation wraps the loaded Mimi runtime.
- [x] Update `MimiStreamingDecoder.decode(_:)` to return `[DecodedAudioChunk]` so real streaming decode can emit zero, one, or multiple chunks per token frame.
- [x] Generated audio token frames decode incrementally without resetting decoder state.
- [x] Empty `StreamArray` outputs are handled without sending silence or placeholder chunks to playback.
- [x] Decoded chunks preserve sample rate, frame index, timestamp, duration, and source token-frame metadata.
- [x] Decode failures surface through `MimiDecodeError` and the Experiment Session failure path.
- [x] Session observations continue to report decoded chunk count, decoded sample rate, decoded duration, and last decoded frame.
- [x] Tests cover state continuity, bad token shape failure, and sink delivery using fake runtime seams where full MLX decode is impractical.
- [x] Reference trace comparison is added or updated for decoded audio chunk event order and cadence.

## Notes

- Follow `ref/moshi-swift/MoshiLib/Mimi.swift` for decode state behavior.
- Keep `DeterministicMimiStreamingDecoder` available for fast tests.
- Do not send invented silence to playback when the decoder has not produced PCM yet.
- Implemented the executable MLX decode graph path: token input shaping, quantizer decode, streaming upsample, decoder transformer cache use, Seanet decoder, decoded PCM extraction, and decoder/upsample weight targets.
- Verified with `swift test --filter MLXMimiRuntime`, `swift test --filter MLXMimiQuantization`, `swift test --filter MLXMimiGraphParameter`, full `swift test`, and `xcodebuild build -project S2STranslate.xcodeproj -scheme S2STranslate -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO`.

## Was blocked by

- `.scratch/hibiki-ios-mlx/issues/16-validate-mimi-runtime-metadata-and-warmup.md`
- `.scratch/hibiki-ios-mlx/issues/23-implement-hibiki-token-sampling-and-text-output.md`

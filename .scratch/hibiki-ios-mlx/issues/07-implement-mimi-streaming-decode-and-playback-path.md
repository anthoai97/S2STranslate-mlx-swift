# Implement Mimi Streaming Decode and Playback Path

Status: complete

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Implement the first stateful Mimi decode path and connect it to a playback-oriented output stream. Token frames should decode into audio chunks that can be written to a test sink and, in the proof-of-concept app, routed to a simple playback sink when running on a capable device.

The slice should demonstrate generated audio output as a stream, not as a batch created after all tokens are available.

## Acceptance criteria

- [x] Mimi token frames can be decoded incrementally into audio chunks.
- [x] Decoder state persists across streamed token frames.
- [x] A test sink can receive decoded audio chunks without using device audio hardware.
- [x] The app can route decoded chunks to a proof-of-concept playback sink.
- [x] Tests cover decode event order, state continuity, stop behavior, and failure propagation.
- [x] Available decode shapes, cadence, or fixtures are compared against the reference trace harness.

## Implementation notes

- Added `MimiStreamingDecoder`, `MimiDecoderDescription`, `DecodedAudioChunk`, `MimiDecodeEvent`, `PlaybackSink`, `PlaybackEvent`, and `BufferedPlaybackSink`.
- Added `DeterministicMimiStreamingDecoder` as the first stateful Swift decoder boundary. It emits one decoded PCM chunk per token frame and preserves output frame index across streamed frames.
- Added `MimiCodecPlaybackExperimentBackend` and `ArtifactAudioMimiPlaybackExperimentBackend` to stream fixture PCM through encode, decode, and buffered playback delivery.
- Extended session observations with decode status, decoded chunk count, decoded sample rate, decoded duration, playback status, playback chunk count, and playback duration.
- Wired the proof-of-concept UI to route decoded chunks to a buffered playback sink on **Start**.
- Added tests for decode state continuity, event/metric reporting, sink delivery, stop behavior, decode failure propagation, playback failure propagation, and reference trace comparison against `audio:mimiDecodeStep`.

This slice does not produce audible device playback yet. `BufferedPlaybackSink` is a proof-of-concept and test sink; a future device sink can route `DecodedAudioChunk` values to platform audio hardware behind the same `PlaybackSink` protocol.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/01-create-experiment-session-skeleton.md`
- `.scratch/hibiki-ios-mlx/issues/04-add-reference-trace-harness.md`
- `.scratch/hibiki-ios-mlx/issues/06-implement-mimi-streaming-encode-path.md`

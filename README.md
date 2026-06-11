# S2STranslate

S2STranslate is an experimental iOS app for running and inspecting streaming speech translation models on device. The target direction starts with Hibiki/Mimi experiments using MLX Swift and the `anquachdev/hbk-zero-3b-mlx-q4` model artifacts.

## Current phase

The app currently implements the Experiment Session skeleton, the first model artifact preparation boundary, simulator-friendly streaming audio input slices, downloadable French file input, deterministic stateful Mimi encode/decode boundaries, a deterministic Hibiki inference boundary, and a buffered playback sink. It does not run real MLX-backed model inference, live microphone capture, MLX-backed Mimi weights, or audible device playback yet.

What works today:

- A SwiftUI proof-of-concept screen for an Experiment Session
- A generic lifecycle: unloaded, preparing, ready, running, stopped, failed
- A manifest-driven artifact preparation backend with a demo provider
- Fixture and file-backed audio sources that emit timestamped 24 kHz chunks
- A small French sample catalog sourced from Kyutai's Hibiki-Zero sample Space, cached locally after first download
- A protocol-backed Mimi streaming encoder boundary that emits deterministic token frames at the 80 ms / 12.5 Hz cadence
- A protocol-backed Hibiki inference boundary that consumes source Mimi token frames and emits deterministic text plus generated target audio-token frames
- A protocol-backed Mimi streaming decoder boundary that emits deterministic decoded PCM chunks
- A buffered playback sink that receives decoded chunks without requiring device audio hardware
- Cache-first artifact preparation semantics covered by tests
- Distinct missing, inaccessible, corrupt, incompatible, and too-large artifact failures
- Session observations such as preparation progress, event count, last event, audio input status, sample rate, chunk count, streamed duration, Mimi frame count, codebook count, token count, Hibiki step count, text token count, generated audio frame count, decoded chunk count, and playback delivery count
- A Moshi-inspired running UI: info/output panel, status strip, centered primary control, and settings popover
- Automated Swift tests for the Experiment Session lifecycle, artifact preparation, reference traces, streaming audio input, file audio input, Mimi encode, deterministic Hibiki inference, Mimi decode, and buffered playback

## Demo flow

1. Open the app.
2. Use the gear button to choose a French sample input if needed.
3. Tap **Prepare** to run the demo artifact preparation path for the pinned Hibiki-Zero manifest.
4. Tap **Start** to download/cache the selected MP3 if needed, decode it to PCM chunks, and stream those chunks through the session backend, deterministic Mimi encoder, deterministic Hibiki inference, Mimi decoder, and buffered playback sink.
5. Watch the output panel show deterministic text and, if enabled, the observations panel update audio status, chunk count, sample rate, duration, Mimi frame count, codebook count, token count, Hibiki steps, generated audio frames, decoded chunks, and playback delivery.
6. Tap **Stop** to end the current attempt.
7. Tap **New Session** to reset after a stopped or failed terminal state.
8. Use the gear button to show or hide session observations or trigger a fake failure.

The demo artifact provider does not download real model weights yet. File audio, deterministic Mimi, and deterministic Hibiki observations are not real microphone capture, MLX token/audio quality, model latency, memory, translation quality, or audible playback.

## Development

Run the Experiment Session tests:

```sh
swift test
```

Build the iOS app without signing:

```sh
xcodebuild build -project S2STranslate.xcodeproj -scheme S2STranslate -destination generic/platform=iOS CODE_SIGNING_ALLOWED=NO
```

## Planning docs

- PRD: `.scratch/hibiki-ios-mlx/PRD.md`
- Local issues: `.scratch/hibiki-ios-mlx/issues/`
- Domain glossary: `CONTEXT.md`
- Reference trace format: `docs/reference-traces.md`
- Microphone smoke test: `docs/microphone-smoke-test.md`

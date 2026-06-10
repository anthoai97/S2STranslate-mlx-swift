# S2STranslate

S2STranslate is an experimental iOS app for running and inspecting streaming speech translation models on device. The target direction starts with Hibiki/Mimi experiments using MLX Swift and the `anquachdev/hbk-zero-3b-mlx-q4` model artifacts.

## Current phase

The app currently implements the Experiment Session skeleton. It does not run real model inference, microphone capture, Mimi encode/decode, or audio playback yet.

What works today:

- A SwiftUI proof-of-concept screen for an Experiment Session
- A generic lifecycle: unloaded, preparing, ready, running, stopped, failed
- A deterministic fake backend that drives preparation, run output, stop, and failure states
- Placeholder observations such as progress, event count, and last event
- A Moshi-inspired running UI: info/output panel, status strip, centered primary control, and settings popover
- Automated Swift tests for the Experiment Session lifecycle

## Demo flow

1. Open the app.
2. Tap **Prepare** to simulate session preparation.
3. Tap **Start** to simulate a running experiment and show placeholder output.
4. Tap **Stop** to end the current attempt.
5. Tap **New Session** to reset after a stopped or failed terminal state.
6. Use the gear button to show or hide placeholder observations or trigger a fake failure.

The placeholder observations are not model latency, memory, frame cadence, token count, audio chunks, or translation quality.

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

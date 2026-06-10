# Wire File-Based Translation Demo

Status: ready-for-agent

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Wire the first complete file-based translation demo through the app. A user should be able to prepare the model, choose or use a fixture French audio input, run the streaming pipeline, see English text events when available, hear or inspect generated audio output, and view basic metrics.

This is the first full vertical slice across artifact loading, audio input, Mimi encode, Hibiki generation, Mimi decode, output, UI state, and tests.

## Acceptance criteria

- [ ] The app can run a translation session from a file or bundled fixture input.
- [ ] The session streams through artifact loading, audio input, encode, inference, decode, and output without requiring microphone input.
- [ ] Text output is displayed when emitted by the model.
- [ ] Generated audio output is routed to playback or a visible/testable output sink.
- [ ] Basic latency, frame cadence, and memory or resource metrics are visible.
- [ ] Tests cover the successful file-based flow using fakes or fixtures where full model execution is impractical.
- [ ] Failure states from each major stage are surfaced in the UI.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/03-load-and-cache-model-artifacts.md`
- `.scratch/hibiki-ios-mlx/issues/05-build-streaming-audio-input-slice.md`
- `.scratch/hibiki-ios-mlx/issues/06-implement-mimi-streaming-encode-path.md`
- `.scratch/hibiki-ios-mlx/issues/07-implement-mimi-streaming-decode-and-playback-path.md`
- `.scratch/hibiki-ios-mlx/issues/08-implement-minimal-hibiki-inference-session.md`

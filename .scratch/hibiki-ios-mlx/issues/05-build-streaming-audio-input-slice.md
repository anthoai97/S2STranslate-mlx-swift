# Build Streaming Audio Input Slice

Status: ready-for-agent

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Add the first audio input path that feeds the experiment session with timestamped PCM chunks. The slice should support a simulator-friendly source such as a fixture or file input, plus a narrow microphone smoke path for device testing. It should prove the app can start, stream chunks, stop cleanly, and report chunk timing without depending on real model inference.

## Acceptance criteria

- [ ] The session can consume PCM chunks from a simulated or file-backed source.
- [ ] The app can start and stop an input stream from the proof-of-concept UI.
- [ ] Chunk timing, sample rate, and stream status are visible through session state or metrics.
- [ ] Tests cover chunk emission, stop behavior, and at least one input failure.
- [ ] A device-only microphone smoke path is available or documented without making automated tests depend on physical hardware.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/01-create-experiment-session-skeleton.md`

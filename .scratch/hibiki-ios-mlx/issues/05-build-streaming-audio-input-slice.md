# Build Streaming Audio Input Slice

Status: complete

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Add the first audio input path that feeds the experiment session with timestamped PCM chunks. The slice should support a simulator-friendly source such as a fixture or file input, plus a narrow microphone smoke path for device testing. It should prove the app can start, stream chunks, stop cleanly, and report chunk timing without depending on real model inference.

## Acceptance criteria

- [x] The session can consume PCM chunks from a simulated or file-backed source.
- [x] The app can start and stop an input stream from the proof-of-concept UI.
- [x] Chunk timing, sample rate, and stream status are visible through session state or metrics.
- [x] Tests cover chunk emission, stop behavior, and at least one input failure.
- [x] A device-only microphone smoke path is available or documented without making automated tests depend on physical hardware.

## Implementation notes

- Added `PCMChunk`, `AudioInputSource`, `FixtureAudioInputSource`, `FailingAudioInputSource`, `AudioInputExperimentBackend`, and `ArtifactAndAudioExperimentBackend`.
- Extended `ExperimentSession` observations with audio input status, chunk count, sample rate, streamed duration, and last audio frame.
- Wired the SwiftUI proof-of-concept to prepare demo artifacts first, then stream fixture 24 kHz PCM chunks on **Start**.
- Added `docs/microphone-smoke-test.md` as the device-only manual path for the future live microphone source.
- Verified with `swift test`.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/01-create-experiment-session-skeleton.md`

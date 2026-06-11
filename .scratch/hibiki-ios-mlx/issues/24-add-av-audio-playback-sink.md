# Add AVAudio Playback Sink

Status: ready-for-agent

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Add a device-capable playback sink behind the existing `PlaybackSink` protocol. The sink should use AVFoundation to play decoded PCM chunks progressively while preserving the buffered sink for tests and simulator-safe diagnostics.

## Acceptance criteria

- [ ] A new `PlaybackSink` implementation can start an AVAudio engine/player at the decoded sample rate.
- [ ] Decoded chunks are scheduled progressively as they arrive.
- [ ] Stop releases audio resources cleanly.
- [ ] Playback failures surface through `PlaybackSinkError` and Experiment Session failure state.
- [ ] The app can choose the device playback sink for real runs and the buffered sink for tests/previews.
- [ ] Tests cover start, receive, stop, and failure behavior with a fake or injectable audio engine seam.
- [ ] Device smoke-test notes explain expected audible behavior and limitations.

## Notes

- This issue makes voice output audible, but only after real Mimi decode produces meaningful PCM.
- Avoid making automated tests depend on physical audio hardware.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/07-implement-mimi-streaming-decode-and-playback-path.md`
- `.scratch/hibiki-ios-mlx/issues/18-implement-mlx-mimi-streaming-decode.md`

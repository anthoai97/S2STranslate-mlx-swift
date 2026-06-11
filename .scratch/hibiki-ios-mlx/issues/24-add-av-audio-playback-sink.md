# Add AVAudio Playback Sink

Status: done

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Add a device-capable playback sink behind the existing `PlaybackSink` protocol. The sink should use AVFoundation to play decoded PCM chunks progressively while preserving the buffered sink for tests and simulator-safe diagnostics.

## Acceptance criteria

- [x] A new `PlaybackSink` implementation can start an AVAudio engine/player at the decoded sample rate.
- [x] Decoded chunks are scheduled progressively as they arrive.
- [x] Stop releases audio resources cleanly.
- [x] Playback failures surface through `PlaybackSinkError` and Experiment Session failure state.
- [x] The app can choose the device playback sink for real runs and the buffered sink for tests/previews.
- [x] Tests cover start, receive, stop, and failure behavior with a fake or injectable audio engine seam.
- [x] Device smoke-test notes explain expected audible behavior and limitations.

## Notes

- This issue makes voice output audible, but only after real Mimi decode produces meaningful PCM.
- Avoid making automated tests depend on physical audio hardware.
- Implemented `AVAudioPlaybackSink` with an injectable engine seam, app default playback now uses the AV sink, and `BufferedPlaybackSink` remains available for tests and diagnostics.
- Verified with `swift test`, `swift test --filter AVAudioPlaybackSink`, and `xcodebuild build -project S2STranslate.xcodeproj -scheme S2STranslate -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO`.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/07-implement-mimi-streaming-decode-and-playback-path.md`
- `.scratch/hibiki-ios-mlx/issues/18-implement-mlx-mimi-streaming-decode.md`

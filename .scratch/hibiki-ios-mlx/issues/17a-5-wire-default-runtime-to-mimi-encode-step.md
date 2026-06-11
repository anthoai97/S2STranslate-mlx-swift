# Wire Default Runtime To Mimi Encode Step

Status: done

## Parent

`.scratch/hibiki-ios-mlx/issues/17a-implement-executable-mlx-mimi-encode-graph.md`

## What to build

Replace the empty-output default `MLXMimiDefaultRuntimeEngine.encode(_:)` behavior with the real streaming Mimi encode graph:

`encoder.step -> encoderTransformer(cache) -> downsample.step -> quantizer.encode`

The runtime should emit `MLXMimiEncodedFrame` values only when the graph produces complete Mimi token frames.

## Acceptance criteria

- [x] `MLXMimiDefaultRuntimeEngine.encode(_:)` calls the executable encode-step graph.
- [x] Empty intermediate stream outputs return `[]` and never emit placeholder token frames.
- [x] Produced frames preserve codebook count `16`.
- [x] Frame extraction preserves 12.5 Hz cadence for emitted frames.
- [x] Runtime graph construction, encode failures, and token extraction failures surface as `MimiRuntimeError` or `MimiEncodeError` with user-visible messages.
- [x] Existing `MLXMimiStreamingEncoder` state-continuity tests pass against the default runtime seam or a focused fake graph.

## Progress

- `MLXMimiDefaultRuntimeEngine.encode(_:)` now executes the Mimi encode-step chain: PCM stream input -> Seanet encoder step -> encoder transformer/cache step -> downsample step -> split residual quantizer encode.
- Added injectable seams for the runtime engine input builder, graph step, and token extractor so empty-stream behavior can be tested without constructing MLX tensors in SwiftPM.
- Added `MLXMimiTokenFrameExtractor` for `[batch, codebook, time]` Mimi token output validation and frame extraction.
- Empty graph output returns `[]` and does not emit placeholder token frames.
- Malformed token output shape/codebook count/buffer length failures surface as `MimiRuntimeError.loadFailed(...)`; `MLXMimiStreamingEncoder` continues converting runtime failures to user-visible `MimiEncodeError.unavailable(...)`.
- Real non-empty graph token extraction is covered by the opt-in local real-artifact fixture.

## Verification

- `swift test --filter MLXMimiRuntime` passes with 12 runtime tests.
- `S2S_RUN_REAL_MIMI_ARTIFACT_TESTS=1 swift test --filter MLXMimiRealArtifact` passes with the local Mimi safetensors artifact.
- `swift test` passes with 96 tests.
- `xcodebuild build -project S2STranslate.xcodeproj -scheme S2STranslate -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO` passes.

## Depends on

- `.scratch/hibiki-ios-mlx/issues/17a-2-implement-mimi-quantizer-encode-slice.md`
- `.scratch/hibiki-ios-mlx/issues/17a-3-implement-mimi-encoder-transformer-cache-slice.md`
- `.scratch/hibiki-ios-mlx/issues/17a-4-apply-mimi-weights-to-executable-graph.md`

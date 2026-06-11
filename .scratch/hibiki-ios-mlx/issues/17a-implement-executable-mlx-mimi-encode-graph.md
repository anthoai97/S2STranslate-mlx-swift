# Implement Executable MLX Mimi Encode Graph

Status: in-progress

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Replace the current empty-output default `MLXMimiRuntimeEngine.encode(_:)` implementation with a real executable Mimi streaming encode graph.

This issue fills the gap between the issue 17 adapter and actual Mimi source-token production. The existing `MLXMimiStreamingEncoder` wrapper should remain the public boundary; this issue makes the runtime engine behind it produce real source Mimi token frames from PCM.

## Acceptance criteria

- [x] Add repo-owned streaming array semantics equivalent to MoshiLib `StreamArray`, including empty output behavior.
- [ ] Port executable encoder-side Mimi layers required by `Mimi.encodeStep(_:)`: Seanet encoder, encoder transformer/cache path, downsample, and split residual vector quantizer encode.
- [ ] Apply loaded Mimi `ModuleParameters` from issue 15 to the executable graph with mapped keys and layout transforms.
- [ ] `MLXMimiDefaultRuntimeEngine.encode(_:)` calls the real streaming graph instead of returning `[]`.
- [ ] Empty intermediate `StreamArray` outputs remain empty and do not emit placeholder token frames.
- [ ] Produced token frames preserve codebook count `16` and frame cadence `12.5 Hz`.
- [ ] Add a reference/fixture comparison for at least one short PCM input where deterministic parity is available from Moshi Swift or the Python reference.
- [ ] Failures from graph construction, parameter application, or token extraction surface as `MimiRuntimeError`/`MimiEncodeError` with user-visible messages.
- [ ] Existing deterministic encoder tests and default UI path remain available.

## Notes

- Follow `ref/moshi-swift/MoshiLib/Mimi.swift` `encodeStep(_:)`:
  `encoder.step -> encoderTransformer(cache) -> downsample.step -> quantizer.encode`.
- Follow `ref/moshi-swift/MoshiLib/Streaming.swift` for empty stream behavior before porting individual layers.
- Keep the public app boundary at `MLXMimiStreamingEncoder`; do not route UI to this path until issue 19.
- If full graph parity is too large for one PR, land the executable graph in guarded vertical slices, but do not mark this issue done until real Mimi tokens come from the default runtime engine.

## Progress

- Added `MLXMimiStreamArray`, a repo-owned equivalent of MoshiLib `StreamArray` for empty/non-empty streaming array behavior.
- `MLXMimiDefaultRuntimeEngine.encode(_:)` now wraps PCM as `MLXMimiStreamArray(MLXArray(samples)[.newAxis, .newAxis])` instead of holding a raw MLX array directly.
- Added focused tests for empty-stream behavior and ensuring `map` does not call transforms for empty streams.

## Verification

- `swift test --filter MLXMimiStreamArray` passes with 2 stream-array tests.
- `swift test --filter StreamingMimiEncode` passes with 8 encode tests.
- `swift test` passes with 74 tests.
- `xcodebuild build -project S2STranslate.xcodeproj -scheme S2STranslate -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO` passes.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/15-load-mimi-safetensors-with-key-mapping.md`
- `.scratch/hibiki-ios-mlx/issues/16-validate-mimi-runtime-metadata-and-warmup.md`
- `.scratch/hibiki-ios-mlx/issues/17-implement-mlx-mimi-streaming-encode.md`

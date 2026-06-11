# Implement Executable MLX Mimi Encode Graph

Status: done

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Replace the current empty-output default `MLXMimiRuntimeEngine.encode(_:)` implementation with a real executable Mimi streaming encode graph.

This issue fills the gap between the issue 17 adapter and actual Mimi source-token production. The existing `MLXMimiStreamingEncoder` wrapper should remain the public boundary; this issue makes the runtime engine behind it produce real source Mimi token frames from PCM.

## Acceptance criteria

- [x] Add repo-owned streaming array semantics equivalent to MoshiLib `StreamArray`, including empty output behavior.
- [x] Port executable encoder-side Mimi layers required by `Mimi.encodeStep(_:)`: Seanet encoder, encoder transformer/cache path, downsample, and split residual vector quantizer encode.
- [x] Apply loaded Mimi weights from issue 15 to the executable graph with mapped keys and layout transforms.
- [x] `MLXMimiDefaultRuntimeEngine.encode(_:)` calls the real streaming graph instead of returning `[]`.
- [x] Empty intermediate `StreamArray` outputs remain empty and do not emit placeholder token frames.
- [x] Produced token frames preserve codebook count `16` and frame cadence `12.5 Hz`.
- [x] Add a reference/fixture comparison for at least one short PCM input where deterministic parity is available from Moshi Swift or the Python reference.
- [x] Failures from graph construction, parameter application, or token extraction surface as `MimiRuntimeError`/`MimiEncodeError` with user-visible messages.
- [x] Existing deterministic encoder tests and default UI path remain available.

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
- Ported the encoder-side Seanet convolution topology as repo-owned Swift objects: streamable conv1d state, residual add buffering, encoder layers, full Seanet encoder shape metadata, and Mimi downsample empty-stream behavior.
- Ported the split residual vector quantizer encode topology as repo-owned Swift objects: codebook encode/decode helpers, first/rest residual quantizers, projection targets, and empty-stream preservation.
- Ported the projected encoder transformer/cache topology as repo-owned Swift objects: linear projections, 8 transformer layers, RoPE attention metadata, layer scale, rotating KV caches, and model reset integration.
- Added explicit Mimi graph parameter target collection/application for conv, transformer, and quantizer objects; `MLXMimiRuntimeLoader` now loads mapped weights and applies them to a default runtime model.
- Wired `MLXMimiDefaultRuntimeEngine.encode(_:)` through the encode-step graph: encoder step, encoder transformer/cache step, downsample step, quantizer encode, and token frame extraction.
- Fixed Mimi weight key mapping for nested Moshi wrappers (`conv.conv`, `convtr.convtr`, and `*_transformer.transformer`) found in the real local safetensors artifact.
- Weight application uses the repo-owned explicit graph applier instead of `MLXNN.ModuleParameters`, because the executable graph is repo-owned Swift objects rather than an `MLXNN.Module` tree.
- Added an opt-in local real-artifact fixture that loads `mimi-pytorch-e351c8d8@125.safetensors`, emits four non-empty source-token frames from the default runtime, and compares Python-reference frame/codebook shape plus the stable token prefix.
- Full later residual-codebook token parity is not exact yet; the executable graph now produces real Mimi tokens and passes shape/cadence/reference-prefix checks, but bit-for-bit codec parity should remain a focused follow-up diagnostic.

## Verification

- `swift test --filter MLXMimiStreamArray` passes with 2 stream-array tests.
- `swift test --filter StreamingMimiEncode` passes with 8 encode tests.
- `swift test --filter MLXMimiModel` passes with 6 model tests.
- `swift test --filter MLXMimiQuantization` passes with 3 quantizer tests.
- `swift test --filter MLXMimiTransformer` passes with 5 transformer tests.
- `swift test --filter MLXMimiGraphParameter` passes with 4 graph parameter tests.
- `swift test --filter MLXMimiRuntime` passes with 12 runtime tests.
- `S2S_RUN_REAL_MIMI_ARTIFACT_TESTS=1 swift test --filter MLXMimiRealArtifact` passes with the local Mimi safetensors artifact.
- `swift test` passes with 96 tests.
- `xcodebuild build -project S2STranslate.xcodeproj -scheme S2STranslate -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO` passes.

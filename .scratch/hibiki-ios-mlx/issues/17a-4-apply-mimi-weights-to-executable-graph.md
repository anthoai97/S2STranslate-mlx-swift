# Apply Mimi Weights To Executable Graph

Status: done

## Parent

`.scratch/hibiki-ios-mlx/issues/17a-implement-executable-mlx-mimi-encode-graph.md`

## What to build

Connect the loaded Mimi safetensor parameters from the existing weight loader to the executable Mimi graph. The runtime should be able to construct the graph, apply mapped/layout-transformed parameters, and report actionable failures for missing, incompatible, or unexpected graph parameters.

This slice should not require the default runtime to emit real token frames yet, but it must make weight application a real runtime step instead of shape-only validation.

## Acceptance criteria

- [x] `MLXMimiRuntimeLoader` or the default runtime engine loads mapped Mimi parameters for the executable graph.
- [x] Graph parameter application uses the existing key and layout mapping from the weight loader.
- [x] Missing or incompatible parameter shapes surface as `MimiRuntimeError` with user-visible messages.
- [x] Weight application does not eagerly allocate large placeholder tensors before real parameters are available.
- [x] Tests cover successful application through injected tiny parameters and failure messages for missing/incompatible keys.

## Progress

- Added `S2STranslate/MLXMimiGraphParameters.swift`, an explicit graph-parameter applier that collects repo-owned Mimi graph targets by mapped key, validates expected shapes, and assigns real `MLXArray` payloads directly to conv, transformer, and quantizer objects.
- `MLXMimiRuntimeLoader` now uses `MLXMimiWeightLoader` and `MLXMimiGraphParameterApplier` to construct a weight-bearing `MLXMimiDefaultRuntimeEngine` instead of returning a metadata-only runtime.
- Graph target collection is shape-only until payload assignment, so constructing the target table does not eagerly allocate placeholder tensors.
- Added graph-parameter tests for expected target shapes, missing key failures, incompatible shape failures, shape-only injected parameter validation, and true injected `MLXArray` payload assignment.
- Added runtime-loader coverage proving incompatible graph parameter shapes surface as `MimiRuntimeError.loadFailed(...)`.
- Fixed real safetensor key mapping for nested Moshi module wrappers (`conv.conv`, `convtr.convtr`, and `*_transformer.transformer`) so the local Mimi artifact applies to the executable graph.
- Verified the local `mimi-pytorch-e351c8d8@125.safetensors` artifact loads through `MLXMimiRuntimeLoader` and produces non-empty default-runtime encode frames in the opt-in real-artifact test.

## Verification

- `swift test --filter MLXMimiGraphParameter` passes with 4 graph parameter tests.
- `swift test --filter MLXMimiRuntime` passes with 12 runtime tests.
- `S2S_RUN_REAL_MIMI_ARTIFACT_TESTS=1 swift test --filter MLXMimiRealArtifact` passes with the local Mimi safetensors artifact.
- `swift test` passes with 96 tests.
- `xcodebuild build -project S2STranslate.xcodeproj -scheme S2STranslate -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO` passes.

## Depends on

- `.scratch/hibiki-ios-mlx/issues/17a-1-land-seanet-streaming-encoder-slice.md`
- `.scratch/hibiki-ios-mlx/issues/17a-2-implement-mimi-quantizer-encode-slice.md`
- `.scratch/hibiki-ios-mlx/issues/17a-3-implement-mimi-encoder-transformer-cache-slice.md`

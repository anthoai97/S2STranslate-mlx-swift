# Implement Mimi Encoder Transformer Cache Slice

Status: done

## Parent

`.scratch/hibiki-ios-mlx/issues/17a-implement-executable-mlx-mimi-encode-graph.md`

## What to build

Implement the projected encoder transformer path required by `Mimi.encodeStep(_:)`, including the streaming cache behavior used across PCM chunks. This slice should make the transformer callable between Seanet encoder output and downsample without resetting state at chunk boundaries.

The goal is a narrow executable transformer path with cache continuity and shape guarantees, not full model parity yet.

## Acceptance criteria

- [x] Port the encoder-side projected transformer structure with input projection, transformer layers, and output projection shape.
- [x] Implement the KV cache path needed for streaming encode, including reset behavior.
- [x] Preserve Mimi transformer metadata: model dimension 512, 8 heads, 8 layers, causal rope, context 250.
- [x] Empty transformer input remains empty and does not advance cache.
- [x] Tests cover cache reset, chunk-boundary continuity with a tiny configuration, and output shape.
- [x] Existing deterministic encoder and fake-runtime tests remain unchanged.

## Progress

- Added `S2STranslate/MLXMimiTransformer.swift`, mirroring MoshiLib's transformer file boundary with repo-owned linear layers, feed-forward layers, attention, layer scale, projected transformer, and KV cache objects.
- Added rotating and simple KV cache implementations; the Mimi 2024-07 path constructs rotating caches with 8 layers, 8 heads, head dimension 64, and context 250.
- `MLXMimiModel` now owns encoder/decoder transformer caches and resets them through `resetEncodeState()` / `resetDecodeState()`.
- `MLXMimiProjectedTransformer.step(_:, cache:)` preserves empty `MLXMimiStreamArray` behavior by not invoking the transform on empty input.
- Tests intentionally avoid forcing numeric attention evaluation in SwiftPM because this environment has already shown MLX device-library failures when evaluating real MLX kernels. The callable graph exists; parity/numeric execution remains part of the later weight/runtime fixture slices.

## Verification

- `swift test --filter MLXMimiTransformer` passes with 5 transformer tests.
- `swift test --filter MLXMimiModel` passes with 6 model tests.
- `swift test --filter MLXMimiQuantization` passes with 3 quantizer tests.
- `swift test` passes with 96 tests.
- `xcodebuild build -project S2STranslate.xcodeproj -scheme S2STranslate -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO` passes.

## Depends on

- `.scratch/hibiki-ios-mlx/issues/17a-1-land-seanet-streaming-encoder-slice.md`

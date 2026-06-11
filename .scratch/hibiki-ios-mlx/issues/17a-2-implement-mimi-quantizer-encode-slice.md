# Implement Mimi Quantizer Encode Slice

Status: done

## Parent

`.scratch/hibiki-ios-mlx/issues/17a-implement-executable-mlx-mimi-encode-graph.md`

## What to build

Implement the executable split residual vector quantizer encode path used by Mimi source-token production. The slice should accept a latent MLX array from the encoder/downsample path and return codebook token arrays with the expected Mimi shape, while preserving empty stream behavior.

This should be verifiable with small deterministic tensors and fake/manual codebook values before real Mimi weights are loaded. The slice may implement codebook embedding reconstruction as an internal residual-encode dependency, but it does not implement or claim the public Mimi Streaming Decoder path.

## Acceptance criteria

- [x] Port Euclidean codebook encode and embedding reconstruction semantics needed by residual vector quantization.
- [x] Port residual vector quantizer encode for one first codebook plus the remaining codebooks.
- [x] Include quantizer input/output projection objects and shapes so real Mimi projection weights have stable graph targets.
- [x] `MLXMimiSplitResidualVectorQuantizer.encode(_:)` returns token arrays with codebook dimension preserved.
- [x] Empty `MLXMimiStreamArray` input to the quantizer stage remains empty.
- [x] Bad token/codebook shape failures are expressible as runtime or encode errors at the runtime boundary.
- [x] Tests cover small deterministic codebook selection, projection shape, and split codebook count using real tiny MLX math.

## Progress

- Added `S2STranslate/MLXMimiQuantization.swift` with Euclidean codebook encode/decode helpers, vector quantization, residual vector quantization, projection objects, and split residual quantizer wiring.
- Kept codebook embedding reconstruction internal to the quantizer path; this does not expose or claim the public Mimi decode path.
- Added `Tests/S2STranslateCoreTests/MLXMimiQuantizationTests.swift` for topology metadata, projection targets, split codebook counts, and empty stream preservation.
- Added deterministic tiny MLX math coverage for nearest-codebook selection.
- Runtime token extraction tests cover bad token/codebook shape failures as user-visible runtime errors.

## Verification

- `swift test --filter MLXMimiQuantization` passes with 4 quantizer tests.
- `swift test --filter MLXMimiModel` passes with 7 model tests.
- `swift test` passes with 96 tests.
- `xcodebuild build -project S2STranslate.xcodeproj -scheme S2STranslate -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO` passes.

## Depends on

- `.scratch/hibiki-ios-mlx/issues/17a-1-land-seanet-streaming-encoder-slice.md`

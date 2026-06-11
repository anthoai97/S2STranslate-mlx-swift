# Add Mimi Encode Reference Parity Fixture

Status: done

## Parent

`.scratch/hibiki-ios-mlx/issues/17a-implement-executable-mlx-mimi-encode-graph.md`

## What to build

Add a deterministic reference comparison for at least one short PCM input encoded through the real Mimi graph. The fixture should compare emitted source Mimi token frame shape, codebook count, frame cadence, and token values where parity is available from Moshi Swift or the Python reference.

This slice closes the confidence gap after the default runtime can produce real tokens.

## Acceptance criteria

- [x] Add or update a short PCM fixture and expected Mimi token-frame trace.
- [x] Reference comparison validates frame count, codebook count, cadence, and token values where deterministic parity is available.
- [x] The test distinguishes shape/cadence mismatch from token-value mismatch.
- [x] Fixture generation steps are documented in the issue or nearby test comments.
- [x] Existing deterministic encoder tests and default UI path remain available.

## Progress

- Added `Tests/S2STranslateCoreTests/MLXMimiRealArtifactTests.swift`, gated by `S2S_RUN_REAL_MIMI_ARTIFACT_TESTS=1` so normal tests do not require the 367 MB local safetensors artifact.
- The opt-in test loads `ref/hibiki-zero-mlx/weights/mimi-pytorch-e351c8d8@125.safetensors` through `MLXMimiRuntimeLoader`, runs the default encode graph on four 80 ms zero-PCM frames, and verifies four emitted Mimi frames with 16 codebooks each.
- Generated the reference trace from the local Python MLX Mimi implementation under `ref/hibiki-zero-mlx/moshi-mlx` using the same zero PCM and filtered 16-codebook load.
- The test checks Python-reference frame count, codebook count, cadence shape, and exact token values for the first three stable codebooks.
- Full residual-token parity for later codebooks is not yet exact; the Swift graph currently emits real tokens and matches the stable prefix, but later residual codebooks diverge from the Python reference and should be diagnosed separately before claiming bit-for-bit codec parity.

## Verification

- `S2S_RUN_REAL_MIMI_ARTIFACT_TESTS=1 swift test --filter MLXMimiRealArtifact` passes.
- `swift test` passes with 96 tests.
- `xcodebuild build -project S2STranslate.xcodeproj -scheme S2STranslate -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO` passes.

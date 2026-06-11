# Validate Mimi Runtime Metadata and Warmup

Status: done

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Add validation and a minimal warmup path for the loaded MLX Mimi Runtime before it is used by streaming encode or decode.

This issue should prove that the runtime has coherent metadata and can execute a tiny encode/decode-shaped path without becoming the default app translation path.

## Acceptance criteria

- [x] Runtime validation confirms sample rate `24000`, frame rate `12.5`, codebook count `16`, quantizer bins `2048`, and samples per frame `1920`.
- [x] Warmup creates a small zero PCM MLX array and exercises the Mimi runtime enough to catch obvious graph/parameter failures.
- [x] Warmup failures surface through explicit Mimi load/runtime errors.
- [x] The runtime exposes reset behavior for encoder and decoder streaming state.
- [x] Tests cover metadata validation, reset forwarding, and warmup failure through fake MLX seams where full execution is impractical.
- [x] The deterministic test doubles remain available and unchanged.

## Verification

- `swift test --filter MLXMimiRuntime` passes with 8 runtime tests.
- `swift test` passes with 69 tests.
- `xcodebuild build -project S2STranslate.xcodeproj -scheme S2STranslate -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO` passes.

## Implementation notes

- `MLXMimiRuntimeConfiguration` now carries `quantizerBins` so runtime metadata validation covers the full Mimi 2024-07 codec contract.
- `MLXMimiRuntime.validateMetadata()` rejects incompatible sample rate, frame rate, codebook count, quantizer bins, and samples per frame before warmup or runtime use.
- `MLXMimiRuntime.warmup(frameCount:)` defaults to Moshi's `4` frame guidance, producing a `[1, 1, 7680]` zero-PCM request for `1920 * 4` samples.
- `MLXMimiRuntimeEngine` is the test seam for reset and warmup behavior. The default engine creates a real MLX zero array and resets the current Mimi model shell; tests use a fake engine to avoid requiring MLX device execution.
- This issue still does not route the UI or deterministic file-demo path through real Mimi encode/decode.

## Notes

- Moshi's reference `warmup()` uses `1920 * 4` zero samples. Use that as guidance, but keep memory/device cost reasonable for iOS.
- This issue still does not wire the MLX Mimi Runtime into the UI default.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/15-load-mimi-safetensors-with-key-mapping.md`

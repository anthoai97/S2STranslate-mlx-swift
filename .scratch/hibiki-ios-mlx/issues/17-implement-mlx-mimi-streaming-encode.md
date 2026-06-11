# Implement MLX Mimi Streaming Encode

Status: in-progress

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Replace the deterministic Mimi encoder in the real file-translation path with an MLX-backed streaming Mimi encoder behind the existing `MimiStreamingEncoder` protocol.

The encoder should accept mono 24 kHz PCM chunks, preserve codec state across calls, and emit source Mimi token frames at the expected 12.5 Hz / 80 ms cadence.

## Acceptance criteria

- [x] A real `MimiStreamingEncoder` implementation wraps the loaded MLX Mimi Runtime.
- [x] PCM chunks convert to MLX arrays with shape compatible with Moshi's `StreamArray(MLXArray(pcm)[.newAxis, .newAxis])` pattern.
- [x] PCM chunks are encoded incrementally without resetting state at chunk boundaries.
- [x] Empty `StreamArray` outputs are handled without emitting bogus `MimiTokenFrame` values.
- [x] When a PCM chunk does not produce a valid Mimi output yet, `encode(_:)` returns an empty frame array rather than padding or placeholder tokens.
- [x] Emitted `MimiTokenFrame` values preserve frame index, timestamp, codebook count, and source audio frame metadata.
- [x] Unsupported sample rates and malformed chunks fail with clear `MimiEncodeError` values.
- [x] Session observations continue to report encoded frame count, codebook count, token count, and cadence.
- [x] Tests cover chunk-boundary state continuity using a fake runtime or tiny fixture seam.
- [ ] Reference trace comparison is added or updated for emitted frame shapes and cadence where deterministic parity is available.

## Verification

- `swift test --filter StreamingMimiEncode` passes with 8 encode tests.
- `swift test --filter MLXMimiRuntime` passes with 8 runtime tests.
- `swift test` passes with 72 tests.
- `xcodebuild build -project S2STranslate.xcodeproj -scheme S2STranslate -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO` passes.

## Implementation notes

- `MLXMimiStreamingEncoder` now implements the existing `MimiStreamingEncoder` protocol and wraps one loaded `MLXMimiRuntime`.
- `MLXMimiRuntimeEngine.encode(_:)` is the seam for streaming Mimi encode. Tests use a fake engine to prove state continuity, empty output handling, emitted token metadata, and bad-token-shape failures.
- The default runtime engine constructs the MLX PCM input using `MLXArray(samples)[.newAxis, .newAxis]`, then returns no token frames until the weight-bearing Mimi `encodeStep` graph is implemented.
- This means the adapter will not invent source tokens. If the runtime produces no `StreamArray` output, the encoder returns `[]`.
- The deterministic encoder remains available and unchanged for fast tests and the current default UI path.
- The remaining work for this issue is replacing the default engine's empty output with real `Mimi.encodeStep(StreamArray(...))` once the ported Mimi modules own executable MLX layers and loaded parameters.

## Notes

- Follow `ref/moshi-swift/MoshiLib/Mimi.swift` `encodeStep(_:)` and `ref/moshi-swift/MoshiCLI/RunMimi.swift` streaming usage.
- Keep `DeterministicMimiStreamingEncoder` available for fast unit tests.
- Do not feed invented source token frames to Hibiki. Buffered Mimi state should be represented as no emitted frame.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/16-validate-mimi-runtime-metadata-and-warmup.md`

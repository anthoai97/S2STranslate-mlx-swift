# Add MLX Swift Dependency and Mimi Runtime Shell

Status: done

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Introduce the smallest MLX Swift integration needed for a future MLX Mimi Runtime without loading real weights or replacing deterministic test doubles.

This issue establishes the dependency, the runtime shell, and the artifact lookup boundary. Model architecture porting, safetensors key mapping, warmup, streaming encode, and streaming decode are handled by follow-up issues.

## Acceptance criteria

- [x] The app/package can import the MLX Swift modules needed for Mimi work.
- [x] `Package.swift` links MLX products into `S2STranslateCore` so `swift test` covers the runtime shell.
- [x] `S2STranslate.xcodeproj` links the same MLX package products into the iOS app target.
- [x] MLX Swift is pinned to a known-good package revision after build verification, not left floating on `main`.
- [x] Package resolution files needed for reproducible app/test builds are present and ready to commit.
- [x] Add an `MLXMimiRuntime` shell that records codec configuration but does not perform real encode/decode yet.
- [x] Add an `MLXMimiRuntimeLoader` shell that locates the prepared `mimiWeights` artifact by role.
- [x] `MLXMimiRuntime` does not construct the real Mimi module graph yet; that is deferred to issue 14.
- [x] Add a `MimiRuntimeError` boundary for missing artifact role, missing artifact file, incompatible configuration, and runtime load failure.
- [x] `MLXMimiRuntime` is shaped as one reusable runtime that can later serve both an encoder wrapper and a decoder wrapper.
- [x] The runtime shell exposes separate encode-state reset, decode-state reset, and full reset entry points even if they are no-ops in this issue.
- [x] The loader validates artifact availability and returns a clear user-visible codec error when the role or file is missing.
- [x] The runtime shell exposes expected sample rate `24000`, frame rate `12.5`, codebook count `16`, and samples per frame `1920`.
- [x] The deterministic encoder/decoder remain available for tests.
- [x] Tests cover missing artifact, wrong artifact role, and successful metadata-only runtime creation.
- [x] Automated tests do not require real MLX device execution or the downloaded `mimiWeights` safetensors.
- [x] Porting notes reference `ref/moshi-swift/MoshiLib/Mimi.swift` and explain that real model construction is deferred.

## Verification

- `swift test` passes with the metadata-only `MLXMimiRuntime` shell and loader tests.
- `xcodebuild build -project S2STranslate.xcodeproj -scheme S2STranslate -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO` passes after installing Apple's Metal Toolchain component required by MLX Swift.

## Notes

- Do not wire this into the UI as the default.
- Keep the runtime shell reusable by both encoder and decoder follow-up slices.
- Treat `ref/moshi-swift` as implementation provenance, not a compiled app dependency.
- Add MLX Swift to both the Swift package and Xcode app target so command-line tests and app builds exercise the same runtime boundary.
- Pin MLX Swift to the exact revision that passes `swift test` and the iOS app build.
- Stop at a metadata/runtime shell in this issue. Real `Mimi`, `SeanetEncoder`, transformer, quantizer, downsample, and upsample construction belong to issue 14.
- Keep runtime load failures separate from `MimiEncodeError` and `MimiDecodeError`; encode/decode errors should represent streaming failures after a runtime has been accepted.
- Use one loaded `MLXMimiRuntime` for both encode and decode in real paths. Do not load separate Mimi runtimes for source encoding and target decoding unless a later device constraint forces that trade-off.
- Save real MLX execution and downloaded safetensors requirements for later runtime validation and smoke-test issues.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/12-implement-real-huggingface-artifact-download-cache.md`

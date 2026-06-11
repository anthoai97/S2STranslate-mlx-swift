# Port Mimi Module Structure From MoshiLib

Status: done

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Port the Mimi model structure needed to construct an MLX Mimi Runtime from the Moshi Swift reference. This issue should create the model classes and configuration shape, but it should not load real safetensors or wire encode/decode into the app flow yet.

The goal is to make the repo own the Mimi architecture boundary while keeping `ref/moshi-swift` as implementation provenance.

## Acceptance criteria

- [x] Port or adapt the Mimi configuration shape from `ref/moshi-swift/MoshiLib/Mimi.swift`.
- [x] Add repo-owned MLX module shells for the Mimi encoder path, decoder path, transformer path, downsample/upsample path, and quantizer path.
- [x] The module structure can be instantiated with the default 2024-07 configuration for `16` codebooks.
- [x] The module structure preserves the expected codec metadata: mono `24 kHz`, `12.5 Hz`, `1920` samples per frame, `2048` quantizer bins, and `16` codebooks.
- [x] The deterministic encoder/decoder remain default in UI and tests.
- [x] Tests cover configuration construction and module-shell instantiation without requiring full safetensors load.
- [x] Porting notes call out any deliberate divergence from MoshiLib naming, shapes, or state ownership.
- [x] The first pass keeps ported internals close to MoshiLib for traceability; app-specific shaping happens at the `MLXMimiRuntime` boundary.

## Verification

- `swift test` passes with 55 tests.
- `xcodebuild build -project S2STranslate.xcodeproj -scheme S2STranslate -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO` passes.

## Porting notes

- `MLXMimiConfiguration` mirrors MoshiLib `MimiConfig.mimi_2024_07`: Seanet v0.1, projected transformer dimensions, split RVQ settings, and derived downsample stride.
- Shell class names keep MoshiLib traceability but are prefixed with `MLXMimi` to avoid collisions with the reference code and future app-specific wrappers.
- The issue 14 shell classes intentionally do not inherit `MLXNN.Module` yet. Under the app target's default `MainActor` isolation, empty `Module` subclasses conflict with `MLXNN.Module.init`; real weight-bearing `Module` ownership should land with the safetensors/key-mapping and real runtime slices.

## Notes

- Follow `ref/moshi-swift/MoshiLib/Mimi.swift`, `Streaming.swift`, `Seanet.swift`, `Conv.swift`, `Transformer.swift`, and `Quantization.swift`.
- Do not implement real `MimiStreamingEncoder.encode` or `MimiStreamingDecoder.decode` in this issue.
- Prefer a close, debuggable port over a Swifty refactor in this issue. Refactor only after the runtime loads and passes smoke tests.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/13-add-mlx-swift-dependency-and-mimi-runtime-shell.md`

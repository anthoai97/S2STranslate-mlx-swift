# Land Seanet Topology And Streaming State Slice

Status: done

## Parent

`.scratch/hibiki-ios-mlx/issues/17a-implement-executable-mlx-mimi-encode-graph.md`

## What to build

Land the first Mimi Encode Graph slice behind the existing Mimi runtime shell: repo-owned streaming array activation semantics, Seanet encoder topology, streaming convolution state, residual add buffering, and Mimi downsample empty-output behavior.

This slice does not need to emit Mimi tokens or prove MLX convolution parity. It should make the encoder-side topology and streaming state real enough that later slices can apply weights and connect transformer/quantizer stages without replacing the public `MLXMimiStreamingEncoder` boundary. Keep the local port shaped like MoshiLib's source files so future work can compare `Mimi`, streaming, conv, Seanet, quantization, and transformer behavior directly against the reference.

## Acceptance criteria

- [x] `MLXMimiStreamArray` preserves Moshi-style empty stream behavior through `map`, `elu`, `cat2`, `split`, and `narrow`.
- [x] Seanet encoder topology matches Mimi 2024-07 channel/kernel/ratio metadata.
- [x] Streamable conv state buffers insufficient input and emits no placeholder arrays.
- [x] Downsample uses the Mimi stride-2 streamable conv shape and keeps empty input empty.
- [x] `MLXMimiModel.resetEncodeState()` resets encoder and downsample streaming state.
- [x] The local Mimi port is split into Moshi-shaped files instead of concentrating implementation details in `MLXMimiModel.swift`.
- [x] Streaming helpers live in `MLXMimiStreaming.swift`, preserving `MLXMimi...` type names while mirroring MoshiLib file boundaries.
- [x] Focused tests cover empty stream behavior and Seanet topology.
- [x] The slice does not require executing MLX conv kernels in Swift package tests.
- [x] Full Swift package tests and iOS simulator build pass.

## Progress

- Split the local Mimi port into Moshi-shaped files:
  - `MLXMimiStreaming.swift`
  - `MLXMimiConv.swift`
  - `MLXMimiSeanet.swift`
- Kept `MLXMimiModel.swift` focused on the top-level Mimi component graph.
- Added focused reset coverage for encode streaming state.
- Added focused buffering coverage proving insufficient streaming input is retained without placeholder output.
- Non-empty MLX padding/conv parity remains covered by the later real-artifact runtime fixture rather than this topology slice.

## Verification

- `swift test --filter MLXMimiModel` passes with 7 model tests.
- `swift test` passes with 96 tests.
- `xcodebuild build -project S2STranslate.xcodeproj -scheme S2STranslate -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO` passes.

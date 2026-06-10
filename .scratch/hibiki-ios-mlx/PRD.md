# PRD: Hibiki and Mimi Streaming Experiments on iOS with MLX Swift

Status: ready-for-agent

## Problem Statement

This repo is intended to make Kyutai-style streaming speech models practical to experiment with on iOS devices using MLX Swift, but the current app is only a minimal SwiftUI shell. A developer cannot yet load the `anquachdev/hbk-zero-3b-mlx-q4` model weights, run the Mimi codec in a fully streaming manner, or exercise a speech-to-speech translation loop from an iOS proof of concept.

The main user problem is not polished end-user translation. The problem is that researchers and mobile ML developers need a small, understandable Swift codebase where they can inspect, modify, and measure the streaming pieces of Hibiki and Mimi on device.

## Solution

Build an experimental iOS-first implementation that uses MLX Swift to run the streaming pieces needed for Hibiki experiments:

- A fully streaming Mimi codec implementation that can encode and decode audio chunk by chunk.
- A Hibiki inference pipeline based primarily on the Python reference in `infer_mlx_fast.py`.
- Model loading and local caching for `anquachdev/hbk-zero-3b-mlx-q4`.
- A proof-of-concept iOS app that can capture or load French speech, stream it through the model, and emit English speech plus optional text translation.
- Developer-oriented instrumentation for latency, frame cadence, memory pressure, and model-stage timings.

The product should optimize for experimentation, traceability to reference implementations, and correctness of the streaming architecture before visual polish.

## User Stories

1. As a mobile ML developer, I want to run Hibiki on an iOS device, so that I can experiment with streaming speech translation outside a desktop Python environment.
2. As a mobile ML developer, I want the implementation to use MLX Swift, so that model execution can use Apple's MLX stack directly from Swift.
3. As a researcher, I want the Swift implementation to stay close to `infer_mlx_fast.py`, so that I can compare behavior against the Python reference.
4. As a researcher, I want the Mimi codec to process audio incrementally, so that I can test real streaming behavior instead of offline batch conversion.
5. As a researcher, I want the codec to preserve streaming state between chunks, so that boundaries between frames do not reset the model.
6. As a researcher, I want to encode microphone PCM into Mimi tokens, so that live speech can be fed into downstream speech models.
7. As a researcher, I want to decode Mimi tokens into audio frames, so that generated target speech can be played progressively.
8. As a developer, I want stable frame timing around the model's expected cadence, so that the pipeline can be reasoned about under live audio constraints.
9. As a developer, I want model weights loaded from local device storage after first download, so that repeated experiments do not depend on network access.
10. As a developer, I want clear errors when model artifacts are missing, corrupt, too large, or incompatible, so that setup failures are diagnosable.
11. As a developer, I want the app to show model loading progress, so that first-run weight preparation does not look frozen.
12. As a developer, I want a minimal recording UI, so that I can test microphone input without building a separate harness.
13. As a developer, I want a file-based audio input path, so that I can reproduce the same translation run repeatedly.
14. As a developer, I want generated English audio to play while generation continues, so that latency can be evaluated as a streaming system.
15. As a developer, I want optional text translation output, so that I can inspect target content even when audio playback is disabled.
16. As a developer, I want timestamped text output when available, so that I can align translation text with generated audio.
17. As a researcher, I want access to sampling parameters, so that I can experiment with quality, latency, and stability tradeoffs.
18. As a researcher, I want access to voice-transfer related controls when supported by the model, so that I can evaluate the effect without changing code.
19. As a developer, I want latency metrics per stage, so that I can find whether capture, codec, model inference, or playback is the bottleneck.
20. As a developer, I want memory metrics during model load and inference, so that device viability is visible early.
21. As a developer, I want logs that identify each streamed chunk and generated frame, so that streaming bugs can be traced.
22. As a developer, I want deterministic test fixtures where possible, so that Swift and Python outputs can be compared on known inputs.
23. As a developer, I want golden traces from the Python implementation, so that porting errors are caught before testing only through the UI.
24. As a developer, I want the app to make unsupported states explicit, so that users know French-to-English is the initial target.
25. As a developer, I want simulator-friendly non-microphone tests, so that most validation does not require a physical device.
26. As a developer, I want device-only smoke tests for audio capture and playback, so that platform integration is still covered.
27. As a maintainer, I want model code separated from SwiftUI views, so that the experimental pipeline can evolve without coupling to the proof-of-concept UI.
28. As a maintainer, I want clean module boundaries for audio IO, codec, model inference, artifact loading, and UI state, so that future model variants can be swapped in.
29. As a maintainer, I want the README to explain setup and first-run expectations, so that contributors can reproduce experiments.
30. As a maintainer, I want documented limitations and safety notes, so that the project is not mistaken for a production translator or impersonation tool.

## Implementation Decisions

- The repo will remain an experimental MLX Swift project, not a polished translation product.
- The first supported model artifact target is `anquachdev/hbk-zero-3b-mlx-q4`.
- The first language direction is French-to-English, matching the Hibiki model card.
- The first iOS app is a proof of concept for experimentation and instrumentation.
- The implementation will use MLX Swift as the model execution layer.
- The Python `infer_mlx_fast.py` implementation is the primary behavioral reference for Hibiki inference structure, token flow, sampling, cache/state handling, and generation loop shape.
- `kyutai-labs/moshi-swift` is the primary Swift reference for Mimi, Moshi, Hibiki, iOS-oriented integration, and common workarounds.
- The Mimi codec will be implemented as a streaming component with explicit state, chunk input, and chunk output.
- Audio capture, codec processing, Hibiki inference, and playback will be separate components connected by streaming interfaces.
- The SwiftUI app will depend on a session-level view model rather than directly calling codec or model internals.
- The session layer will expose observable state for unloaded, preparing, ready, running, failed, and stopped states.
- Model artifact management will support local cache discovery before network download.
- The artifact loader will treat `https://huggingface.co/anquachdev/hbk-zero-3b-mlx-q4` as the canonical Hugging Face source for first-run model weight download.
- The app will surface first-run model preparation as a user-visible state.
- The pipeline will support both microphone input and file input for reproducible tests.
- Generated audio output will be streamable to playback as frames become available.
- Text output will be optional but included when produced by the model.
- Sampling and generation settings will be represented as explicit configuration, with conservative defaults.
- Device performance instrumentation will be part of the first implementation, not a later polish item.
- The code should prefer small, inspectable Swift types over clever abstraction because the repo's goal is experimentation.
- The UI should expose only the controls needed to run and inspect experiments: load model, choose input, start, stop, status, text output, and basic metrics.
- The implementation should avoid assuming stable network availability after the initial model download.

## Testing Decisions

- Good tests should validate external behavior at streaming boundaries: given input chunks, the system emits expected tokens, audio frames, text events, state transitions, or errors.
- Tests should avoid asserting private implementation details such as internal tensor variable names unless those names are part of a model conversion contract.
- The highest-value seam is the Mimi codec: feed known PCM chunks and verify encoded token cadence, decode behavior, state continuity, and no reset at chunk boundaries.
- The next seam is the Hibiki inference session: feed known source tokens or audio fixtures and verify emitted text/audio event structure, stop behavior, and deterministic behavior under fixed sampling settings where possible.
- The artifact loading seam should verify local cache hit, missing artifact error, incompatible artifact error, and first-run download/preparation flow using a fake artifact provider.
- The audio IO seam should use simulated sources and sinks for automated tests, with device smoke tests reserved for microphone and playback integration.
- The UI seam should verify user-visible state transitions: unloaded to preparing, preparing to ready, ready to running, running to stopped, and failure states.
- Golden traces should be generated from the Python reference for small fixtures and used to compare tensor shapes, token sequences, timing cadence, and generated event order where deterministic comparison is practical.
- Performance tests should record coarse latency and memory budgets rather than brittle exact timings.
- Since the current repo has no test target yet, implementation should add a test target or an equivalent Swift testing setup early.

## Out of Scope

- Production-grade translation UX.
- Additional language pairs beyond French-to-English.
- Training, fine-tuning, or modifying Hibiki weights.
- App Store readiness, account systems, analytics, subscriptions, or onboarding.
- Server-backed inference.
- Offline batch translation as the primary workflow.
- Full Moshi dialogue functionality unless required as a shared implementation dependency.
- Malicious voice impersonation, deceptive voice cloning, or features designed to bypass consent.
- Perfect parity with every upstream model variant in the first milestone.

## Further Notes

- Current checkpoint: the first Experiment Session skeleton is implemented. The app has a Moshi-inspired fake-backed control surface with prepare, start, stop, failure-demo, and new-session flows, plus placeholder observations and automated lifecycle tests. It still does not perform real model loading, audio capture, Mimi encode/decode, Hibiki inference, or playback.
- Hibiki is a simultaneous speech-to-speech and speech-to-text translation model. The referenced model card describes the mobile Hibiki variant as a hierarchical Transformer producing speech and text tokens at 12.5 Hz, with audio generated at about 1.1 kbps.
- Mimi is the streaming neural audio codec foundation used by Moshi and Hibiki. The Moshi Swift README describes Mimi as processing 24 kHz audio into a 12.5 Hz representation at 1.1 kbps in a fully streaming manner, with 80 ms frame latency.
- MLX Swift can be added to Xcode as a Swift package dependency. The app should avoid linking multiple copies of MLX through nested frameworks.
- Reference links:
  - https://huggingface.co/anquachdev/hbk-zero-3b-mlx-q4
  - https://huggingface.co/kyutai/hibiki-1b-mlx-bf16
  - https://github.com/ml-explore/mlx-swift
  - https://github.com/kyutai-labs/moshi-swift
  - https://github.com/huybik/hibiki-zero-mlx/blob/main/src/infer_mlx_fast.py

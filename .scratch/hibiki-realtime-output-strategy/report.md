# Hibiki Realtime Output Strategy Research Report

## Table of Contents

1. [Define realtime output policy and budget](#define-realtime-output-policy-and-budget) - Summary: Use `generatedRealtimeFactor = generatedAudioDurationSeconds / processingWallTimeSeconds` as the single gating metric for output mode selection. Classify `< 1.0x` as `sub-realti... | Recommended Next Step: Implement a minimal `RealtimeOutputPolicy` that consumes `generatedRealtimeFactor`, classifies `< 1.0x` as `sub-realtime`, `>= 1.0x` and `< 1.25x` as `bare-realtime`, and `>= 1....
2. [Route sub-realtime real-file runs to deferred playback](#route-sub-realtime-real-file-runs-to-deferred-playback) - Summary: The current real-file model path is measurably sub-realtime, not live-capable. Issue 28 and the benchmark docs place generated realtime around 0.216x to 0.242x, with device logs... | Recommended Next Step: Change `S2STranslate/ContentView.swift` so the default device real-file backend routes known sub-realtime model-backed runs to `DeferredAudioPlaybackSink(wrapped: AVAudioPlaybac...
3. [Add playback-only diagnostics checks](#add-playback-only-diagnostics-checks) - Summary: The recommended check is a device-only playback probe that reuses `PlaybackSink` and `AVAudioPlaybackSink` instead of creating a separate audio path. It should run two inputs: a... | Recommended Next Step: Add a guarded device test or debug action that replays saved `DecodedAudioChunk` output and synthetic 24 kHz mono Float32 sine chunks through `AVAudioPlaybackSink`, records exis...
4. [Split Hibiki benchmark timing into substages](#split-hibiki-benchmark-timing-into-substages) - Summary: The current benchmark measures `inference.step(sourceAudioTokens:)` as one 328.986 ms average bucket, but the real call path spans multiple distinct stages: sequence-state reads... | Recommended Next Step: Implement the timer breakdown in the real-file benchmark path first, with explicit `MLX.eval(...)` boundaries and aggregated Depformer timers, then rerun the existing 40-source-...
5. [Optimize the first measured Hibiki bottleneck](#optimize-the-first-measured-hibiki-bottleneck) - Summary: The current local benchmark already identifies the first measured bottleneck: `Hibiki step` is far larger than Mimi encode or decode. On the 40-chunk benchmark, generated realti... | Recommended Next Step: First land substage instrumentation, then immediately prototype an MLX-native sampler path that removes `.asArray(Float.self)` from `MLXHibikiRuntime.step` and `MLXHibikiModel.s...
6. [Decide mobile-live model profile strategy](#decide-mobile-live-model-profile-strategy) - Summary: Recommendation: yes, plan for a second mobile-live profile if the current q4 path stays far below realtime after substage profiling and one focused optimization pass. Upstream K... | Recommended Next Step: Proceed with a dual-profile plan, but gate it on one short evidence pass: split `Hibiki step`, apply one profiler-led cleanup to the current q4 path, and if the resulting device...

## Detailed Content

### Define realtime output policy and budget

_Source: `Define_realtime_output_policy_and_budget.json`_

#### Basic Info

**Item Name:** Define realtime output policy and budget

**Research Focus:** Define a policy for classifying the current Hibiki model-backed translation path into sub-realtime, bare-realtime, and practical-realtime output strategies using the repo's generatedRealtimeFactor metric.

**Summary:** Use `generatedRealtimeFactor = generatedAudioDurationSeconds / processingWallTimeSeconds` as the single gating metric for output mode selection. Classify `< 1.0x` as `sub-realtime` and route to deferred or buffered output instead of default live playback, classify `>= 1.0x` and `< 1.25x` as `bare-realtime` and allow live playback only as a warned diagnostic mode, and classify `>= 1.25x` as `practical-realtime` and default live playback. The current measured `0.216x` Hibiki path is firmly sub-realtime, so the immediate policy should disable default live playback for this path and treat performance recovery as separate follow-up work.

#### Technical Features

**Relevant Primary Sources:**

- Title: Hugging Face Audio Course: Inverse Real-Time Factor (RTFx) | Url: https://huggingface.co/learn/audio-course/en/chapter5/evaluation | Signal: Defines the higher-is-better throughput metric `audio duration / processing time`, with `> 1.0` faster than real time, `= 1.0` exact real time, and `< 1.0` slower than real time.
- Title: Kyutai Hibiki-Zero model card | Url: https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16 | Signal: States that Hibiki-Zero generates text and audio tokens at a constant `12.5 Hz`, implying an `80 ms` output cadence that can be used as the frame budget for live output policy.
- Title: Kyutai moshi-swift README | Url: https://github.com/kyutai-labs/moshi-swift | Signal: States that Mimi converts `24 kHz` audio to a `12.5 Hz` representation with `80 ms` latency and that the iOS app is a proof of concept, which supports a conservative live-output policy.
- Title: Kyutai Hibiki README | Url: https://github.com/kyutai-labs/hibiki/blob/main/README.md | Signal: Says the MLX-Swift path is experimental on iPhone and that the `1B` model is ideal for on-device inference, which suggests the current heavier path should not be assumed live-capable.
- Title: MLX Lazy Evaluation documentation | Url: https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html | Signal: Explains that each graph evaluation has fixed overhead, which is relevant when a streaming loop evaluates small graphs every frame.

**Local Code Or Docs To Check:**

- Path: Tests/S2STranslateCoreTests/RealFileFrenchEnglishSmokeTests.swift | Reason: Defines `generatedRealtimeFactor` as `decodedAudioDurationSeconds / processingSeconds` and is the authoritative benchmark/reporting seam.
- Path: docs/real-file-french-english-smoke-test.md | Reason: Already documents that values below `1.0` mean the model flow is slower than playback consumption.
- Path: .scratch/real-file-benchmark/latest/benchmark.md | Reason: Current benchmark baseline showing `0.216x`, `27.463 s` processing time, `5.920 s` generated audio, and stage timing summaries.
- Path: .scratch/hibiki-ios-mlx/issues/28-investigate-choppy-real-time-translation-playback.md | Reason: Shows the device-side underrun pattern and a post-prebuffer production rate around `0.23x` realtime.
- Path: S2STranslate/AVAudioPlaybackSink.swift | Reason: Contains `AVAudioPlaybackSink`, `DeferredAudioPlaybackSink`, and `BufferedStreamingAudioPlaybackSink`, which are the concrete output strategies available today.
- Path: S2STranslate/ExperimentSession.swift | Reason: Owns user-visible observations and is the right place to expose the selected realtime classification and output strategy.
- Path: S2STranslate/StreamingMimiDecode.swift | Reason: Defines playback diagnostics reporting that can confirm whether a live-capable classification is actually holding queue depth.
- Path: S2STranslate/ContentView.swift | Reason: Currently wires the real-file path to buffered pseudo-streaming and is the first integration point for policy-driven mode selection.

**Implementation Implications:**

- Introduce a small policy type that maps `generatedRealtimeFactor` to `sub-realtime`, `bare-realtime`, or `practical-realtime`, then maps that class to a concrete sink strategy.
- For the current `0.216x` baseline, do not default to `AVAudioPlaybackSink` or small-prebuffer pseudo-streaming. Choose `DeferredAudioPlaybackSink` or an equivalent delayed mode that preserves text and WAV artifacts.
- Treat the `1.0x` threshold as the hard minimum for any live playback attempt because the metric already uses a higher-is-better throughput ratio.
- Treat the `1.0x` to `< 1.25x` band as a diagnostic or warned live mode rather than the default user path, because exact-real-time throughput leaves little room for queue recovery, audio scheduling jitter, or device variance.
- Treat `>= 1.25x` as the default-live band for this repo because an `80 ms` cadence with a `1.25x` throughput target implies an average processing budget of about `64 ms` per generated `80 ms` chunk, leaving approximately `16 ms` of slack.
- Record both the selected class and the numeric factor in observations so logs and screenshots explain why the app used live, buffered, or output-only behavior.
- Keep the benchmark formula stable across tests, docs, and runtime policy to avoid contradictory interpretations of the same metric.

#### Performance Metrics

**Measurement Plan:**

- Continue to compute `generatedRealtimeFactor` from the benchmark harness and from comparable device runs using the same fixture and same report formula.
- Attach the policy decision to every real-file run so the report captures factor, class, chosen output strategy, and whether live playback was allowed, warned, or disallowed.
- For any run in the `bare-realtime` or `practical-realtime` bands, also record playback diagnostics including pending seconds, schedule gap, and underrun count, because throughput alone does not prove stable queue health.
- Split the current `Hibiki step` bucket into substages before optimization work so the repo can identify which part of the remaining gap blocks promotion from `sub-realtime` to `bare-realtime`.
- Use the same French-to-English fixture for before/after comparisons and keep tail-flush behavior explicit, because delayed output changes generated duration and perceived throughput.

**Expected Signal:**

- At the current `0.216x` baseline, the policy should consistently classify the run as `sub-realtime` and choose deferred or buffered output instead of default live playback.
- A future run should not be considered even `bare-realtime` until `generatedRealtimeFactor` reaches at least `1.0x` and playback diagnostics show that pending audio does not immediately collapse after playback starts.
- A future run should not be promoted to `practical-realtime` until `generatedRealtimeFactor` reaches at least `1.25x` and playback diagnostics show stable positive queue depth with no repeated underruns on target hardware.
- If throughput rises above `1.0x` but playback still fails while pending queue depth stays healthy, the remaining blocker is likely audio-path configuration rather than model starvation.

#### Milestone Significance

**Recommended Next Step:** Implement a minimal `RealtimeOutputPolicy` that consumes `generatedRealtimeFactor`, classifies `< 1.0x` as `sub-realtime`, `>= 1.0x` and `< 1.25x` as `bare-realtime`, and `>= 1.25x` as `practical-realtime`, then wire the real-file path to choose `DeferredAudioPlaybackSink` for the current `0.216x` class while exposing the class and thresholds in observations.

**Acceptance Criteria:**

- The repo has one explicit policy definition for `sub-realtime`, `bare-realtime`, and `practical-realtime` based on `generatedRealtimeFactor`.
- A run at `0.216x` is classified as `sub-realtime` and does not default to live playback.
- The `1.0x` threshold is treated as the hard minimum for any live attempt, and the `1.25x` threshold is treated as the default-live threshold for this repo.
- Experiment observations or benchmark output show the measured factor, the chosen class, and the chosen output strategy.
- Buffered or deferred mode still preserves generated text and WAV artifacts for sub-realtime runs.
- Playback-only diagnostics remain available so the repo can distinguish model starvation from AVAudio configuration issues after the policy lands.

**Risks:**

- The `1.25x` practical threshold is a repo policy rather than an upstream contract and may need tuning by device class or future sink behavior.
- If the policy relies only on a stale last-known benchmark, a model revision or hardware change could misclassify a run.
- Buffered or deferred playback can make the app sound better while hiding unresolved AVAudio configuration issues unless playback-only checks remain part of validation.
- Generated audio duration can exceed source duration because of delayed output and tail flush, so all comparisons must keep using the same report formula rather than mixing source-duration and generated-duration interpretations.

**Open Questions:**

- Should live gating use the last benchmark result, a short runtime warmup, or a rolling median of recent runs on the same device and model revision?
- Should the `bare-realtime` band ever auto-start live playback with a larger prebuffer, or should it remain a diagnostic-only mode until the path reaches `practical-realtime`?
- Should thresholds vary between iPhone, simulator, and Mac targets, or should the policy stay device-agnostic and rely on measured factor plus playback diagnostics?
- Should text remain fully live while audio is deferred for `sub-realtime` runs, or should the UI describe the entire run as delayed output mode?
- Which substage inside the current `Hibiki step` bucket is the first bottleneck worth optimizing for the required `4.63x` to `5.79x` throughput gain?

**Sources:**

- Type: local_doc | Path: .scratch/real-file-benchmark/latest/benchmark.md | Why It Matters: Authoritative local baseline for the current `0.216x` throughput and stage timings.
- Type: local_doc | Path: .scratch/hibiki-ios-mlx/issues/28-investigate-choppy-real-time-translation-playback.md | Why It Matters: Authoritative local diagnosis showing live prebuffer drain and repeated underruns at about `0.23x` post-prebuffer production speed.
- Type: local_doc | Path: docs/real-file-french-english-smoke-test.md | Why It Matters: Documents the repo's current interpretation that values below `1.0` are slower than playback consumption.
- Type: local_code | Path: Tests/S2STranslateCoreTests/RealFileFrenchEnglishSmokeTests.swift | Why It Matters: Defines the benchmark formula and report output used by the repo.
- Type: web | Title: Hugging Face Audio Course: Inverse Real-Time Factor (RTFx) | Url: https://huggingface.co/learn/audio-course/en/chapter5/evaluation | Why It Matters: Provides the external higher-is-better throughput interpretation that matches the repo metric.
- Type: web | Title: Kyutai Hibiki-Zero model card | Url: https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16 | Why It Matters: Provides the `12.5 Hz` generation cadence used to convert throughput goals into per-chunk timing budgets.
- Type: web | Title: Kyutai moshi-swift README | Url: https://github.com/kyutai-labs/moshi-swift | Why It Matters: Provides the `80 ms` Mimi frame latency and the proof-of-concept framing for iOS.
- Type: web | Title: Kyutai Hibiki README | Url: https://github.com/kyutai-labs/hibiki/blob/main/README.md | Why It Matters: Provides the experimental iPhone note and the upstream preference for the smaller `1B` model on-device.
- Type: web | Title: MLX Lazy Evaluation documentation | Url: https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html | Why It Matters: Explains why per-frame graph evaluation overhead is a plausible contributor to poor streaming throughput.

#### Uncertain Fields

- realtime_budget_impact

### Route sub-realtime real-file runs to deferred playback

_Source: `Route_subrealtime_realfile_runs_to_deferred_playback.json`_

#### Basic Info

**Item Name:** Route sub-realtime real-file runs to deferred playback

**Research Focus:** Use the existing playback sink boundary to stop broken live audio on sub-realtime model-backed real-file runs while preserving generated text output and existing WAV artifact workflows.

**Summary:** The current real-file model path is measurably sub-realtime, not live-capable. Issue 28 and the benchmark docs place generated realtime around 0.216x to 0.242x, with device logs showing roughly 0.23x after the 2 second prebuffer drains. Because `BufferedStreamingAudioPlaybackSink(prebufferDurationSeconds: 2)` starts AVAudio before enough output exists, playback repeatedly underruns. The safest immediate change is to classify these runs as sub-realtime and route the app path to `DeferredAudioPlaybackSink(wrapped: AVAudioPlaybackSink())`, which preserves decode and text generation flow and delays audible playback until all generated chunks are buffered.

#### Technical Features

**Relevant Primary Sources:**

- .scratch/hibiki-ios-mlx/issues/28-investigate-choppy-real-time-translation-playback.md: device diagnosis shows the 2 second prebuffer drains, underruns reach 115, and model output arrives at about 0.23x realtime.
- .scratch/hibiki-realtime-output-strategy/PRD.md: defines sub-realtime output handling as a product requirement and states current baselines around 0.216x to 0.242x should not be presented as smooth simultaneous playback.
- S2STranslate/AVAudioPlaybackSink.swift: `DeferredAudioPlaybackSink` already buffers decoded chunks until `finish()`, while `BufferedStreamingAudioPlaybackSink` starts once a prebuffer threshold is reached.
- S2STranslate/StreamingHibikiInference.swift: the real-file backend emits Hibiki text and decoded chunk events during generation, then delegates audio delivery through the injected `PlaybackSink`.
- Tests/S2STranslateCoreTests/AVAudioPlaybackSinkTests.swift: tests already prove deferred playback does not start wrapped audio until `finish()` and that buffered pseudo-streaming starts once a threshold is met.
- Tests/S2STranslateCoreTests/RealFileFrenchEnglishSmokeTests.swift and docs/real-file-french-english-smoke-test.md: existing smoke and benchmark flows already preserve `translation.txt` and `translation.wav` without depending on live AVAudio playback.

**Local Code Or Docs To Check:**

- S2STranslate/ContentView.swift: `defaultTranslateBackend` currently injects `BufferedStreamingAudioPlaybackSink(prebufferDurationSeconds: 2)` for device real-file runs.
- S2STranslate/AVAudioPlaybackSink.swift: verify `DeferredAudioPlaybackSink` and `BufferedStreamingAudioPlaybackSink` behavior and diagnostics exposure.
- S2STranslate/StreamingHibikiInference.swift: confirm `playbackSink.start`, `playbackSink.receive`, and `playbackSink.finish` are the only audio-delivery seams for the real-file run.
- Tests/S2STranslateCoreTests/AVAudioPlaybackSinkTests.swift: keep coverage for deferred scheduling and buffered-start behavior.
- Tests/S2STranslateCoreTests/RealFileFrenchEnglishSmokeTests.swift: keep `translation.txt` and `translation.wav` generation working through buffered/offline capture paths.
- .scratch/hibiki-ios-mlx/issues/28-investigate-choppy-real-time-translation-playback.md and .scratch/hibiki-realtime-output-strategy/two-solution-research.md: use as the grounding documents for why sub-realtime routes should not default to live playback.

**Implementation Implications:**

- Add the output-strategy decision before constructing `RealFileHibikiTranslationExperimentBackend`; the smallest immediate version can hardcode the current q4 real-file path as sub-realtime.
- For sub-realtime runs, replace the current app-path sink with `DeferredAudioPlaybackSink(wrapped: AVAudioPlaybackSink())` and leave the generation loop unchanged.
- Do not modify `appendHibikiGeneratedFrameEvents`; it already emits text and decoded audio events before playback scheduling, so generated text visibility is preserved.
- Keep `BufferedStreamingAudioPlaybackSink` only for future live-capable or explicitly forced-live runs because a larger prebuffer alone does not solve a 0.23x producer.
- Keep `BufferedPlaybackSink` in smoke and benchmark tests for artifact writing. If the interactive app must also save the exact deferred-playback run as a WAV, add a small recording or tee sink because `DeferredAudioPlaybackSink` does not expose buffered chunks.

#### Performance Metrics

**Realtime Budget Impact:**

- This change does not improve model throughput; it only prevents AVAudio from consuming output faster than the model produces it.
- It removes the need for an impractically large initial live buffer. Issue 28 estimated about 33 seconds of buffered generated audio would be needed to hide underruns at roughly 0.23x realtime.
- It converts the user experience from broken pseudo-live playback to delayed but smooth playback, which fits inspection and demo correctness but not simultaneous translation.

**Measurement Plan:**

- Use the existing real-file benchmark path and record `generatedRealtimeFactor`; classify values below 1.0x as sub-realtime.
- After routing to deferred playback, rerun the device real-file flow and verify that no live underrun growth occurs during generation because wrapped AVAudio does not start until `finish()`.
- Replay already-buffered decoded chunks through `AVAudioPlaybackSink` and verify `underrunCount` remains zero or within an explicitly accepted tolerance.
- Keep the smoke and benchmark artifact checks: `translation.txt` must remain nonempty when tail flush is enabled, and `translation.wav` must remain nonempty.
- Expose or log the selected output strategy and the reason, for example the measured realtime factor that caused deferred routing.

**Expected Signal:**

- Sub-realtime real-file runs stop producing repeated live-playback underruns because `AVAudioPlaybackSink` is no longer started while the model is still starving the queue.
- Generated text still appears during the run because text events are emitted before playback delivery in the current inference loop.
- Post-generation playback is smooth when the already-buffered decoded chunks are finally scheduled through AVAudio.
- Benchmarks still report roughly 0.216x to 0.242x generated realtime factor, confirming the change is a routing fix rather than a performance fix.

#### Milestone Significance

**Recommended Next Step:** Change `S2STranslate/ContentView.swift` so the default device real-file backend routes known sub-realtime model-backed runs to `DeferredAudioPlaybackSink(wrapped: AVAudioPlaybackSink())`, and record an observation string that explains playback is intentionally deferred because current generated realtime is below the live threshold.

**Acceptance Criteria:**

- With sub-realtime capability, the device real-file app path no longer defaults to `BufferedStreamingAudioPlaybackSink(prebufferDurationSeconds: 2)` for audible output.
- Generated text remains visible during the run and the session still emits decoded audio chunk events.
- Deferred playback starts only after generation finishes and produces smooth playback from already-buffered chunks.
- Existing smoke and benchmark flows continue to write nonempty `translation.txt` and `translation.wav` artifacts.
- The selected output strategy is visible in observations or logs so users understand the run is delayed by design, not failing silently.

**Risks:**

- This is a product-safety workaround, not a realtime recovery. Simultaneous translation remains unsolved until model throughput improves.
- If the UI does not clearly explain the mode switch, users may interpret delayed playback as a regression rather than an intentional truth-in-capability change.
- CoreAudio channel-map or converter warnings may still appear when deferred playback finally starts because the workaround does not change AVAudio configuration.
- If the interactive app needs same-run WAV export, an additional recording sink layer will be needed because the deferred sink only forwards chunks to wrapped playback at finish.

**Open Questions:**

- Should the first implementation use a hardcoded current capability assumption for the q4 real-file path, or add a small `RealtimeOutputPolicy` immediately?
- Does the interactive app need to persist `translation.wav` from the same deferred-playback run, or is the existing smoke and benchmark artifact path sufficient?
- After deferred routing is in place, should remaining CoreAudio channel-map warnings block playback, or can they be treated as a separate cleanup issue?
- What practical live threshold should re-enable buffered live playback on target hardware: only `>= 1.0x`, or a safer margin such as `>= 1.25x`?

**Sources:**

- .scratch/hibiki-ios-mlx/issues/28-investigate-choppy-real-time-translation-playback.md
- .scratch/hibiki-realtime-output-strategy/PRD.md
- .scratch/hibiki-realtime-output-strategy/two-solution-research.md
- docs/real-file-french-english-smoke-test.md
- S2STranslate/ContentView.swift
- S2STranslate/AVAudioPlaybackSink.swift
- S2STranslate/StreamingHibikiInference.swift
- Tests/S2STranslateCoreTests/AVAudioPlaybackSinkTests.swift
- Tests/S2STranslateCoreTests/RealFileFrenchEnglishSmokeTests.swift

### Add playback-only diagnostics checks

_Source: `Add_playbackonly_diagnostics_checks.json`_

#### Basic Info

**Item Name:** Add playback-only diagnostics checks

**Research Focus:** Isolate AVAudio scheduling health from model starvation by replaying already-generated Decoded Audio Chunks and deterministic 24 kHz mono Float32 PCM through the existing Playback Sink boundary on device.

**Summary:** The recommended check is a device-only playback probe that reuses `PlaybackSink` and `AVAudioPlaybackSink` instead of creating a separate audio path. It should run two inputs: already-generated decoded chunks and synthetic 24 kHz mono Float32 PCM, then classify the result from existing playback diagnostics plus route and interruption notifications. This will let the app prove a simple distinction: if playback-only runs stay healthy while model-backed live runs underrun below realtime, the dominant problem is model starvation rather than AVAudio scheduling.

#### Technical Features

**Relevant Primary Sources:**

- Title: Apple Developer Documentation: scheduleBuffer(_:completionCallbackType:completionHandler:) | Url: https://developer.apple.com/documentation/avfaudio/avaudioplayernode/schedulebuffer%28_%3Acompletioncallbacktype%3Acompletionhandler%3A%29 | Why It Matters: Apple documents that the callback type is selectable. For diagnostics, this is important because the default completion handler is not the same signal as device playback completion.
- Title: Apple Developer Documentation: AVAudioPlayerNodeCompletionCallbackType.dataPlayedBack | Url: https://developer.apple.com/documentation/avfaudio/avaudioplayernodecompletioncallbacktype/dataplayedback | Why It Matters: Apple defines a callback that corresponds to data finishing in the playback device. That is the best fit for a playback-health diagnostic counter.
- Title: Apple Audio Session Programming Guide: Responding to Interruptions | Url: https://developer.apple.com/library/archive/documentation/Audio/Conceptual/AudioSessionProgrammingGuide/HandlingAudioInterruptions/HandlingAudioInterruptions.html | Why It Matters: Interruptions immediately stop audio and require explicit recovery logic. Playback-only diagnostics should record these events separately so they are not misread as model starvation.
- Title: Apple Audio Session Programming Guide: Responding to Route Changes | Url: https://developer.apple.com/library/archive/documentation/Audio/Conceptual/AudioSessionProgrammingGuide/HandlingAudioHardwareRouteChanges/HandlingAudioHardwareRouteChanges.html | Why It Matters: Route changes are expected playback hazards for headset unplug, Bluetooth changes, and output changes. A playback-only health check should log route-change reasons beside the audio metrics.
- Title: Apple Developer Documentation: Performing offline audio processing | Url: https://developer.apple.com/documentation/avfaudio/performing-offline-audio-processing | Why It Matters: Apple notes that offline manual rendering does not use the live audio device path. It is useful for non-device processing, but it cannot validate AVAudio scheduling health or CoreAudio playback warnings.

**Local Code Or Docs To Check:**

- Path: S2STranslate/AVAudioPlaybackSink.swift | Why: This is the real device playback seam. It already tracks scheduled samples, completed samples, pending duration, schedule gap, and underrun count.
- Path: S2STranslate/StreamingMimiDecode.swift | Why: This defines `PlaybackDiagnosticsSnapshot` emission through `playbackDiagnosticsEvent(from:)`, which the playback-only probe should reuse instead of inventing a new metric model.
- Path: S2STranslate/ExperimentSession.swift | Why: This is where playback diagnostics become `ExperimentObservations`, so a diagnostic result can be surfaced without a large new UI surface.
- Path: Tests/S2STranslateCoreTests/AVAudioPlaybackSinkTests.swift | Why: These tests already prove the seam shape and buffered-streaming behavior. They are the closest place to add non-device regression coverage for the new probe logic.
- Path: docs/real-file-french-english-smoke-test.md | Why: This doc already distinguishes model-flow benchmarking from AVAudio playback and is the right place to add a playback-only device checklist.
- Path: .scratch/hibiki-realtime-output-strategy/PRD.md | Why: The PRD explicitly requires playback-only diagnostics that reuse the same Playback Sink boundary and synthetic PCM checks.

**Implementation Implications:**

- Add one opt-in device-only playback diagnostic entry point that reuses `PlaybackSink` and runs with `AVAudioPlaybackSink`.
- Run two input classes through the same sink: already-generated `DecodedAudioChunk` values and deterministic synthetic 24 kHz mono Float32 chunks.
- Keep the synthetic chunk cadence aligned with the existing pipeline frame size: 80 ms per chunk at 24 kHz equals 1,920 samples per chunk.
- Record route-change, interruption, and media-services-reset notifications during the probe and attach them to the diagnostic output so AVAudio environment failures are separated from starvation.
- For diagnostic accuracy, introduce a device-played completion signal in the AVAudio path by using `scheduleBuffer(_:completionCallbackType:completionHandler:)` with `.dataPlayedBack`, or by adding a separate diagnostics-only counter with that callback type.
- Do not use offline manual rendering as proof of playback health because it bypasses the live output device path that the diagnostic needs to validate.
- Keep the live model benchmark separate from the playback-only probe; the point of this check is classification, not throughput measurement.

#### Performance Metrics

**Realtime Budget Impact:** This adds almost no model-time cost because it removes MLX, Mimi, and Hibiki from the critical path. Deferred playback-only runs cost roughly the audio duration plus setup overhead, and paced synthetic runs cost roughly audio duration plus the chosen prebuffer. The main budget value is diagnostic classification: a passing playback-only run means sub-realtime model production remains the primary reason for live underruns.

**Measurement Plan:**

- Probe A: replay already-generated `DecodedAudioChunk` values through `AVAudioPlaybackSink` as a fully buffered run so model production speed is not part of the result.
- Probe B: generate 5 to 10 seconds of synthetic 24 kHz mono Float32 PCM, split into 80 ms chunks, and replay it through the same sink in deferred mode.
- Probe C: run the same synthetic PCM through `BufferedStreamingAudioPlaybackSink` with a small prebuffer such as 0.5 to 1.0 seconds while the producer sends 80 ms chunks faster than realtime, for example every 20 to 40 ms, so the queue should stay positive if AVAudio scheduling is healthy.
- During each probe, collect `scheduledBufferCount`, `completedBufferCount`, `pendingBufferCount`, `scheduledSampleCount`, `completedSampleCount`, `pendingSampleCount`, `pendingDurationMilliseconds`, `lastScheduleGapMilliseconds`, `underrunCount`, and `elapsedMilliseconds` from `PlaybackDiagnosticsSnapshot`.
- Also collect route changes, interruptions, media-services-reset notifications, thrown playback errors, and any CoreAudio warnings seen during the same run window.
- Compare the playback-only results with a model-backed live run. If playback-only remains healthy while the live run drains pending duration to zero and increments underruns below realtime, classify the live issue as model starvation.

**Expected Signal:**

- Healthy AVAudio path: playback-only probes complete without thrown playback errors, `underrunCount` stays at 0, and pending duration does not collapse to zero until the normal end of playback.
- Healthy AVAudio but starved model path: playback-only probes pass, but the model-backed live path shows pending duration draining after prebuffer and repeated queue-empty events as new 80 ms chunks arrive slower than playback consumes them.
- Unhealthy AVAudio path: playback-only probes also underrun, stall, or fail on a stable route even though their producer is not sub-realtime.
- Environment-caused failure: route-change or interruption events occur during the probe and line up with a failure or queue reset, which should be reported separately from steady-state scheduling health.
- More accurate completion accounting: once device-played completion is used for diagnostics, completed duration should lag scheduled duration until the device really plays the audio instead of appearing complete too early.

#### Milestone Significance

**Recommended Next Step:** Add a guarded device test or debug action that replays saved `DecodedAudioChunk` output and synthetic 24 kHz mono Float32 sine chunks through `AVAudioPlaybackSink`, records existing playback diagnostics plus route and interruption events, and emits a simple classification such as `healthy`, `environment-interrupted`, or `unhealthy`.

**Acceptance Criteria:**

- The playback-only diagnostic reuses the same `PlaybackSink` boundary as production playback.
- A device replay of already-generated chunks completes on a stable audio route with `underrunCount == 0`.
- A deferred replay of synthetic 24 kHz mono Float32 chunks completes on a stable audio route with `underrunCount == 0`.
- A paced synthetic replay with prebuffer and a producer faster than realtime keeps positive pending duration until near normal completion and does not increment underruns.
- The diagnostic output includes scheduled, completed, and pending duration or sample counts, last schedule gap, underrun count, and environment events.
- Completed-playback metrics are based on device-played completion semantics rather than only the default render-thread scheduling callback.
- When playback-only passes but a model-backed live run underruns below realtime, the result is reported as model starvation rather than AVAudio playback failure.

**Risks:**

- The current `scheduleBuffer(buffer) { ... }` callback semantics can make completed playback look healthier than it really is, because the default callback is not the same as device-played completion.
- A fully buffered replay can prove route and scheduling health, but by itself it will not expose live pacing bugs; that is why the faster-than-realtime paced synthetic probe is also needed.
- Device audio behavior still varies across speakers, headphones, Bluetooth routes, backgrounding, and interruptions, so a clean playback-only result should be recorded with route context.
- Synthetic sine-wave PCM can validate scheduling and format handling, but it may not expose every artifact that real decoded speech does; replaying already-generated chunks is still necessary.

**Open Questions:**

- Should the diagnostic live as a guarded `swift test` similar to the existing smoke tests, or as an in-app debug action for quicker device iteration?
- What is the preferred source for already-generated chunks: in-memory captured `DecodedAudioChunk` values, a fixture file re-decoded into chunks, or a saved WAV converted back into chunk boundaries?
- Do we want two completion counters in diagnostics, one for render completion and one for device-played completion, or is a single device-played counter enough?
- What threshold for `lastScheduleGapMilliseconds` should count as unhealthy on target devices once enough probe data has been collected?

**Sources:**

- Title: Apple Developer Documentation: scheduleBuffer(_:completionCallbackType:completionHandler:) | Url: https://developer.apple.com/documentation/avfaudio/avaudioplayernode/schedulebuffer%28_%3Acompletioncallbacktype%3Acompletionhandler%3A%29 | Type: official
- Title: Apple Developer Documentation: AVAudioPlayerNodeCompletionCallbackType.dataPlayedBack | Url: https://developer.apple.com/documentation/avfaudio/avaudioplayernodecompletioncallbacktype/dataplayedback | Type: official
- Title: Apple Developer Documentation: prepare(withFrameCount:) | Url: https://developer.apple.com/documentation/avfaudio/avaudioplayernode/prepare%28withframecount%3A%29 | Type: official
- Title: Apple Developer Documentation: Performing offline audio processing | Url: https://developer.apple.com/documentation/avfaudio/performing-offline-audio-processing | Type: official
- Title: Apple Audio Session Programming Guide: Responding to Interruptions | Url: https://developer.apple.com/library/archive/documentation/Audio/Conceptual/AudioSessionProgrammingGuide/HandlingAudioInterruptions/HandlingAudioInterruptions.html | Type: official
- Title: Apple Audio Session Programming Guide: Responding to Route Changes | Url: https://developer.apple.com/library/archive/documentation/Audio/Conceptual/AudioSessionProgrammingGuide/HandlingAudioHardwareRouteChanges/HandlingAudioHardwareRouteChanges.html | Type: official
- Title: Stack Overflow: Timing issues: Metronome using AVAudioEngine scheduleBuffer's completion handler | Url: https://stackoverflow.com/questions/67341908/timing-issues-metronome-using-avaudioengine-schedulebuffers-completion-handler | Type: community
- Title: AudioKit issue #2910: player did not see an IO cycle | Url: https://github.com/AudioKit/AudioKit/issues/2910 | Type: community
- Title: S2STranslate local source: S2STranslate/AVAudioPlaybackSink.swift | Url: file://S2STranslate/AVAudioPlaybackSink.swift | Type: local
- Title: S2STranslate local source: docs/real-file-french-english-smoke-test.md | Url: file://docs/real-file-french-english-smoke-test.md | Type: local
- Title: S2STranslate local source: .scratch/hibiki-realtime-output-strategy/PRD.md | Url: file://.scratch/hibiki-realtime-output-strategy/PRD.md | Type: local

### Split Hibiki benchmark timing into substages

_Source: `Split_Hibiki_benchmark_timing_into_substages.json`_

#### Basic Info

**Item Name:** Split Hibiki benchmark timing into substages

**Research Focus:** Add profiler-grade timing inside the real MLX Hibiki step so the benchmark separates state bookkeeping, main transformer evaluation, text-logit extraction and sampling, Depformer slice evaluation, Depformer logit extraction and sampling, generated-token state writes, and final Swift output or token-frame construction.

**Summary:** The current benchmark measures `inference.step(sourceAudioTokens:)` as one 328.986 ms average bucket, but the real call path spans multiple distinct stages: sequence-state reads and writes in `MLXHibikiDefaultRuntimeEngine.step`, `model.mainStep(...)`, one text-logit `asArray(Float.self)` read, `model.sampleDepformer(...)`, repeated per-slice `logits.asArray(Float.self)` reads, and final `HibikiTextOutput` plus `MimiTokenFrame` construction in `MLXHibikiInferenceSession.step`. The most important measurement rule is to place explicit `MLX.eval(...)` boundaries before any Swift-side `asArray(...)` extraction, because MLX is lazy and `asArray(...)` itself calls `self.eval()` before copying bytes into a Swift array. That split will let the benchmark show whether the 80 ms per-frame budget is mainly lost in MLX compute, repeated GPU-to-CPU extraction, CPU sampling, or state and token-frame bookkeeping.

#### Technical Features

**Relevant Primary Sources:**

- Kind: local_code | Reference: Tests/S2STranslateCoreTests/RealFileFrenchEnglishSmokeTests.swift:343-464 | Why It Matters: Shows the benchmark currently wraps the entire Hibiki step in one timer and reports only coarse stage summaries.
- Kind: local_code | Reference: S2STranslate/MLXHibikiRuntime.swift:182-233 | Why It Matters: Shows the real Hibiki engine step order: state store, delayed token lookup, main step, text sampling, Depformer sampling, and generated-token state updates.
- Kind: local_code | Reference: S2STranslate/MLXHibikiRuntime.swift:293-342 | Why It Matters: Shows the session-level work after engine.step: frame index mutation, text decoding, and generated `MimiTokenFrame` creation.
- Kind: local_code | Reference: S2STranslate/MLXHibikiModel.swift:166-225 | Why It Matters: Shows `mainStep(...)` and the per-slice Depformer loop, including repeated `logits.asArray(Float.self)` calls.
- Kind: local_dependency_source | Reference: .build/checkouts/mlx-swift/Source/MLX/MLXArray+Bytes.swift:123-136 | Why It Matters: Confirms `asArray(...)` forces `self.eval()` and then copies data into a Swift array.
- Kind: local_dependency_source | Reference: .build/checkouts/mlx-swift/Source/MLX/Transforms+Eval.swift:10-23 and .build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/transforms.cpp:257-269 | Why It Matters: Confirms `MLX.eval(...)` is the explicit evaluation boundary and waits for scheduled outputs.
- Kind: official_doc | Reference: https://ml-explore.github.io/mlx/build/html/usage/quick_start.html | Why It Matters: Documents that MLX operations are lazy and that `item()` or array conversion trigger evaluation.
- Kind: official_doc | Reference: https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html | Why It Matters: Documents evaluation trade-offs and warns against over-fragmenting the graph with too many eval calls.
- Kind: official_doc | Reference: https://ml-explore.github.io/mlx/build/html/python/devices_and_streams.html | Why It Matters: Defines stream synchronization semantics for any future async profiling path.
- Kind: upstream_issue | Reference: https://github.com/ml-explore/mlx-swift-lm/issues/124 | Why It Matters: Shows upstream Swift inference already treats per-token GPU-to-CPU sync and repeated eval boundaries as real performance risks.

**Local Code Or Docs To Check:**

- Tests/S2STranslateCoreTests/RealFileFrenchEnglishSmokeTests.swift:343-464
- S2STranslate/MLXHibikiRuntime.swift:182-233
- S2STranslate/MLXHibikiRuntime.swift:293-342
- S2STranslate/MLXHibikiModel.swift:166-225
- .scratch/real-file-benchmark/latest/benchmark.md
- docs/hibiki-mlx-porting-notes.md
- .build/checkouts/mlx-swift/Source/MLX/MLXArray+Bytes.swift:123-136
- .build/checkouts/mlx-swift/Source/MLX/MLXArray.swift:502-508
- .build/checkouts/mlx-swift/Source/MLX/Transforms+Eval.swift:10-23
- .build/checkouts/mlx-swift/Source/MLX/Stream.swift:116-119

**Implementation Implications:**

- Instrument inside `MLXHibikiDefaultRuntimeEngine.step(...)` and `MLXHibikiInferenceSession.step(...)`, not only around `inference.step(...)` in the benchmark harness, because the current top-level timer cannot see state prep versus MLX work versus Swift output work.
- Add explicit evaluation boundaries before Swift extraction. For the text path, evaluate `mainOutput.transformerOutput` and `mainOutput.textLogits` together, then time `mainOutput.textLogits.asArray(Float.self)` separately. For the Depformer path, evaluate each slice `logits` before timing `logits.asArray(Float.self)`.
- Keep `asArray(...)` timing separate from CPU sampling timing. Today `tokenSampler.sample(logits: mainOutput.textLogits.asArray(Float.self), ...)` folds MLX evaluation, byte copy, and CPU top-k or softmax work into one number.
- Aggregate Depformer timings by category first: `depformerEvalMs`, `depformerLogitsExtractMs`, and `depformerSamplingCpuMs`. Optionally add per-slice arrays later if the aggregate still hides a skewed first-slice or late-slice hotspot.
- Track session-level bookkeeping separately: `frameIndexLockMs`, `generatedTokenStateWriteMs`, and `tokenFrameCreationMs`. Those are unlikely to explain 329 ms alone, but the split is useful for proving they are not the main bottleneck.
- Do not add `Stream.synchronize()` in the blocking `MLX.eval(...)` path unless profiling switches to `asyncEval(...)`. The pinned MLX C++ implementation shows `eval(...)` already waits for outputs.

#### Performance Metrics

**Realtime Budget Impact:** Current Hibiki Step Average Ms: 328.986; Current Mimi Encode Average Ms: 21.846; Current Mimi Decode Average Ms: 22.044; Current Measured Pipeline Stage Sum Ms: 372.876; Frame Budget Ms At 12 5 Hz: 80.0; Hibiki Step Vs Budget Ratio: 4.11x the full per-frame budget; Pipeline Stage Sum Vs Budget Ratio: 4.66x the full per-frame budget; Why The Split Matters: Without substage timing, an optimization to a small CPU bookkeeping path can look active while the real regression remains hidden inside MLX compute or repeated GPU-to-CPU extraction.

**Measurement Plan:**

Instrumentation Location: Add a benchmark-only breakdown object that is populated inside `MLXHibikiDefaultRuntimeEngine.step(...)` and `MLXHibikiInferenceSession.step(...)`, then surfaced through `RealFileModelFlowBenchmarkReport` beside the existing coarse `hibikiStepMilliseconds` summary.; Timers: - Name: hibikiFrameIndexLockMs | Where: Around the `state.withLock` block in `MLXHibikiInferenceSession.step(...)` that reads configuration and increments `nextFrameIndex`. | Captures: Session mutex and frame-index bookkeeping before engine work starts.
- Name: hibikiSourceStateStoreMs | Where: Around `sequenceState.storeSourceTokens(...)` in `MLXHibikiDefaultRuntimeEngine.step(...)`. | Captures: Insertion of source Mimi tokens into delayed state.
- Name: hibikiMainInputLookupMs | Where: Around `textTokenForMainStep(...)` and `audioTokensForMainStep(...)`. | Captures: Delayed text and audio token lookup for the main transformer input.
- Name: hibikiMainTransformerEvalMs | Where: Build `mainOutput = model.mainStep(...)`, then time `MLX.eval(mainOutput.transformerOutput, mainOutput.textLogits)`. | Captures: Main transformer compute, output norm, and text head evaluation without Swift extraction cost.
- Name: hibikiTextLogitsExtractMs | Where: After `hibikiMainTransformerEvalMs`, time `mainOutput.textLogits.asArray(Float.self)`. | Captures: Swift array extraction and memory copy for text logits only.
- Name: hibikiTextSamplingCpuMs | Where: Time `tokenSampler.sample(...)` after the Swift logits array already exists. | Captures: Pure CPU top-k ranking, temperature scaling, and sample selection.
- Name: hibikiDepformerEvalMs | Where: Inside `model.sampleDepformer(...)`, for each slice time `MLX.eval(logits)` after `linearIn + embedding + transformer + norm + linearOut` produce the slice logits, then accumulate. | Captures: Aggregate MLX compute for Depformer slice logits across the full generated-codebook loop.
- Name: hibikiDepformerLogitsExtractMs | Where: Inside each slice, after `MLX.eval(logits)`, time `logits.asArray(Float.self)` and accumulate. | Captures: Aggregate GPU-to-CPU extraction and Swift array copy for Depformer logits.
- Name: hibikiDepformerSamplingCpuMs | Where: Inside each slice, time `sampler.sample(...)` after the Swift logits array already exists and accumulate. | Captures: Aggregate CPU sampling across all generated audio codebooks.
- Name: hibikiGeneratedStateWriteMs | Where: Around `sequenceState.storeGeneratedTokens(...)`. | Captures: Write-back of sampled text and generated audio tokens into delayed state.
- Name: hibikiTokenFrameCreationMs | Where: In `MLXHibikiInferenceSession.step(...)`, time `HibikiTextOutput(...)`, `textTokenDecoder?.piece(for:)`, and `MimiTokenFrame(...)` creation after `engine.step(...)` returns. | Captures: Swift object construction and final step packaging.
- Name: hibikiAccountedMs | Where: Sum of all substage timers per step. | Captures: Accounted time that can be compared with the existing coarse `hibikiStepMilliseconds` timer.
- Name: hibikiUnaccountedMs | Where: Existing coarse Hibiki step wall time minus `hibikiAccountedMs`. | Captures: Instrumentation gaps or extra async work that leaked across stage boundaries.; Reporting Rules: - Keep the existing coarse `hibikiStepMilliseconds` summary for continuity and add a second table for Hibiki substages.
- Report count, average, p50, p95, and max for every substage, matching the current benchmark summary format.
- For Depformer, also report `sliceCountPerStep = depformerLogitsExtractCount / hibikiStepCount` so the benchmark confirms expected per-step multiplicity.; Boundary Rules: - Do not time `model.mainStep(...)` alone as compute because it mainly builds lazy MLX arrays. Close the compute window with `MLX.eval(...)`.
- Do not time `asArray(...)` without a preceding explicit `MLX.eval(...)` if the goal is to distinguish compute from extraction; `asArray(...)` otherwise includes both.
- Only use `Stream.synchronize()` if a later profiling experiment switches to `asyncEval(...)` or custom streams.

**Expected Signal:**

- The split will show whether the current 328.986 ms average is dominated by main-transformer evaluation, aggregate Depformer evaluation, repeated logits extraction, or CPU sampling.
- If GPU-to-CPU extraction is a major cost, `hibikiTextLogitsExtractMs + hibikiDepformerLogitsExtractMs` will be large even when `hibikiTextSamplingCpuMs` and `hibikiDepformerSamplingCpuMs` stay relatively small.
- If model compute is the real bottleneck, `hibikiMainTransformerEvalMs` and or `hibikiDepformerEvalMs` will absorb most of the coarse Hibiki timer while extraction and bookkeeping remain minor.
- If the accounted sum does not closely match the current coarse timer, the remaining gap will identify missing work such as implicit evaluation, object packaging outside the measured window, or benchmark-harness overhead.

#### Milestone Significance

**Recommended Next Step:** Implement the timer breakdown in the real-file benchmark path first, with explicit `MLX.eval(...)` boundaries and aggregated Depformer timers, then rerun the existing 40-source-chunk benchmark that produced the 328.986 ms baseline so the before or after comparison stays apples-to-apples.

**Acceptance Criteria:**

- The benchmark JSON and markdown include a dedicated Hibiki substage table in addition to the existing coarse stage table.
- The new timers isolate at least these categories: main transformer evaluation, text logits extraction, text sampling CPU, Depformer evaluation, Depformer logits extraction, Depformer sampling CPU, state updates, and token-frame creation.
- The sum of Hibiki substages stays within 5 percent of the existing coarse Hibiki wall time on the same benchmark run, with any gap exposed as `hibikiUnaccountedMs`.
- The report makes it obvious which single Hibiki substage has the highest average and p95 latency, so optimization work can target one measured bottleneck instead of the entire 329 ms bucket.

**Risks:**

- Extra profiling boundaries can perturb MLX scheduling. Adding many `MLX.eval(...)` calls may slightly change absolute latency, so the substage split should be benchmark-only or guarded by a profiling flag.
- Per-slice Depformer timers create more samples and more reporting data. If the raw report becomes noisy, keep the public report aggregated and store per-slice detail only in JSON.
- If future models change the generated codebook count, any fixed-width Depformer table will drift. Use runtime counts derived from the loaded topology instead of hard-coded slice totals.
- If the benchmark continues to use the coarse timer without the accounted-versus-unaccounted check, implicit evaluation can still hide inside the wrong bucket.

**Sources:**

- Description: Current benchmark report with 328.986 ms Hibiki average | Location: .scratch/real-file-benchmark/latest/benchmark.md
- Description: Real-file benchmark harness and coarse Hibiki timing | Location: Tests/S2STranslateCoreTests/RealFileFrenchEnglishSmokeTests.swift:343-464
- Description: Hibiki runtime engine step order | Location: S2STranslate/MLXHibikiRuntime.swift:182-233
- Description: Hibiki session output construction | Location: S2STranslate/MLXHibikiRuntime.swift:293-342
- Description: Hibiki model main step and Depformer sampling loop | Location: S2STranslate/MLXHibikiModel.swift:166-225
- Description: MLX Swift `asArray(...)` implementation | Location: .build/checkouts/mlx-swift/Source/MLX/MLXArray+Bytes.swift:123-136
- Description: MLX Swift `eval(...)` wrapper and blocking C++ implementation | Location: .build/checkouts/mlx-swift/Source/MLX/Transforms+Eval.swift:10-23; .build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/transforms.cpp:257-269
- Description: MLX Quick Start: lazy evaluation and implicit evaluation on read | Location: https://ml-explore.github.io/mlx/build/html/usage/quick_start.html
- Description: MLX Lazy Evaluation guide | Location: https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html
- Description: MLX Devices and Streams reference | Location: https://ml-explore.github.io/mlx/build/html/python/devices_and_streams.html
- Description: MLX Distributed guide note on `MLX_METAL_FAST_SYNCH=1` | Location: https://ml-explore.github.io/mlx/build/html/usage/distributed.html
- Description: Upstream Swift inference performance issue about per-token sync and eval boundaries | Location: https://github.com/ml-explore/mlx-swift-lm/issues/124

#### Uncertain Fields

- open_questions

### Optimize the first measured Hibiki bottleneck

_Source: `Optimize_the_first_measured_Hibiki_bottleneck.json`_

#### Basic Info

**Item Name:** Optimize the first measured Hibiki bottleneck

**Research Focus:** Find the highest-leverage optimization path inside the currently dominant `Hibiki step` of Swift MLX Hibiki inference, with emphasis on lazy evaluation, compilation, host array materialization, synchronization, Depformer cost, and memory pressure.

**Summary:** The current local benchmark already identifies the first measured bottleneck: `Hibiki step` is far larger than Mimi encode or decode. On the 40-chunk benchmark, generated realtime is `0.216x` with average `Hibiki step` `328.986 ms`, versus `21.846 ms` Mimi encode and `22.044 ms` Mimi decode; a nearby run measured `0.242x` with `292.809 ms` average `Hibiki step`. The Swift path currently forces CPU materialization of logits at least 17 times per generated frame: once for text logits in `MLXHibikiRuntime.step`, then once per generated codebook slice inside `MLXHibikiModel.sampleDepformer`, after which `HibikiTopKTokenSampler` sorts those arrays on the CPU. Relative to upstream references, the best first optimization path is to remove repeated host materialization and CPU-side sampling from the hot loop, then measure whether the remaining time is main-transformer compute, Depformer compute, or compile/synchronization overhead.

#### Technical Features

**Relevant Primary Sources:**

- .scratch/real-file-benchmark/latest/benchmark.md: latest local benchmark shows `0.216x` generated realtime with average `Hibiki step` `328.986 ms`.
- .scratch/real-file-benchmark/text-check/benchmark.md: nearby validation benchmark shows `0.242x` generated realtime with average `Hibiki step` `292.809 ms`.
- S2STranslate/MLXHibikiRuntime.swift: `step(...)` currently materializes `mainOutput.textLogits.asArray(Float.self)` before sampling.
- S2STranslate/MLXHibikiModel.swift: `sampleDepformer(...)` currently materializes `logits.asArray(Float.self)` once per Depformer slice.
- S2STranslate/StreamingHibikiInference.swift: `HibikiTopKTokenSampler` sorts full Swift `[Float]` logits arrays on the CPU.
- ref/moshi-swift/MoshiLib/LM.swift: upstream MLX-Swift Moshi/Hibiki reference keeps sampling in MLX arrays, calls `textToken.eval()` and `audioTokens.eval()`, and does not convert logits to `[Float]` in the per-slice loop.
- ref/hibiki-zero-mlx/src/infer_mlx_fast.py: Python reference overlaps CPU Mimi encode/decode with GPU LM work and only synchronizes the current text token per frame.
- https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html: MLX documents fixed overhead per graph evaluation and warns against evaluating too frequently or via implicit reads such as `item()` or array conversion.
- https://ml-explore.github.io/mlx/build/html/usage/compile.html: MLX documents `compile()` as a way to fuse work and reduce runtime and memory, while also warning that compiled functions should be reused and kept pure with explicit state capture.
- https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html: MLX documents unified memory and stream-based CPU/GPU scheduling, which matters for host/device synchronization and memory pressure analysis.
- https://github.com/ml-explore/mlx-swift-lm/issues/124: recent community report attributes major Swift/Python inference gaps to per-token synchronization, `asyncEval` serialization, and missing stream-context fusion opportunities.
- https://github.com/kyutai-labs/hibiki: Hibiki README says the `1B` variant is ideal for on-device inference and the MLX-Swift implementation was tested on iPhone 16 Pro.
- https://huggingface.co/kyutai/hibiki-1b-mlx-bf16: model card describes Hibiki-M as a `1.7B` mobile model producing speech/text tokens at `12.5 Hz`.

**Local Code Or Docs To Check:**

- S2STranslate/MLXHibikiRuntime.swift: inspect `step(...)` for text-logit materialization, sequence-state bookkeeping, and final token storage.
- S2STranslate/MLXHibikiModel.swift: inspect `mainStep(...)`, `sampleDepformer(...)`, cache reset behavior, and per-slice loop structure.
- S2STranslate/StreamingHibikiInference.swift: inspect `HibikiTopKTokenSampler` because its full-array sort cost is directly in the hot loop.
- Tests/S2STranslateCoreTests/RealFileFrenchEnglishSmokeTests.swift: extend the benchmark harness so `Hibiki step` is split into substages rather than treated as one timer.
- docs/hibiki-mlx-porting-notes.md: verify the intended Hibiki-Zero topology, especially the 16 generated codebooks and sequential Depformer execution.
- ref/moshi-swift/MoshiLib/LM.swift and ref/moshi-swift/MoshiLib/Utils.swift: use as the closest upstream Swift reference for MLX-native sampling and cache handling.
- ref/hibiki-zero-mlx/src/infer_mlx_fast.py: use as the closest reference for overlapping CPU Mimi work with GPU LM work.

#### Performance Metrics

**Realtime Budget Impact:**

- At the current `0.216x` run, the system needs about `4.63x` more end-to-end throughput just to reach bare realtime and about `5.79x` to reach `1.25x` practical realtime.
- Using the latest benchmark averages, the total model-flow cost per generated `80 ms` frame is roughly `372.876 ms`. If Mimi encode and decode stay near `44 ms` combined, `Hibiki step` would need to fall from about `329 ms` to about `36 ms`, roughly a `9x` reduction, for the same architecture to sustain bare realtime without overlap.
- Using the `0.242x` benchmark, the same calculation still leaves only about `42 ms` for `Hibiki step`, so even the better run implies about a `7x` step reduction if encode/decode stay unchanged.
- This means a successful first optimization should be judged by whether it materially shrinks the dominant bucket and clarifies the remaining gap, not by whether it alone reaches realtime.

#### Milestone Significance

**Recommended Next Step:** First land substage instrumentation, then immediately prototype an MLX-native sampler path that removes `.asArray(Float.self)` from `MLXHibikiRuntime.step` and `MLXHibikiModel.sampleDepformer`. This is the smallest optimization with the strongest evidence behind it because the current code performs whole-logit host materialization and CPU sorting in the hottest loop, while the upstream Moshi/Hibiki MLX-Swift reference keeps sampling in MLX arrays.

**Acceptance Criteria:**

- The benchmark report no longer treats `Hibiki step` as a single opaque bucket; it reports at least `mainStep`, `textSampling`, and `depformerTotal` timings.
- After the first optimization patch, the steady-state Hibiki path no longer converts text or Depformer logits to `[Float]` before sampling.
- The same 40-chunk fixture shows either a `>=20%` improvement in `Hibiki step` p50 or a clear measurement proving that materialization/sampling is not the dominant internal substage.
- Real-file smoke behavior remains correct: nonempty visible text, nonzero Hibiki step count, and nonzero generated audio/decode counts still pass.
- The resulting data makes the next optimization decision obvious: continue with compile/overlap if compute dominates, or continue reducing boundary overhead if sampling/materialization still dominates.

**Sources:**

- .scratch/real-file-benchmark/latest/benchmark.md
- .scratch/real-file-benchmark/text-check/benchmark.md
- S2STranslate/MLXHibikiRuntime.swift
- S2STranslate/MLXHibikiModel.swift
- S2STranslate/StreamingHibikiInference.swift
- Tests/S2STranslateCoreTests/RealFileFrenchEnglishSmokeTests.swift
- docs/hibiki-mlx-porting-notes.md
- ref/moshi-swift/MoshiLib/LM.swift
- ref/moshi-swift/MoshiLib/Utils.swift
- ref/hibiki-zero-mlx/src/infer_mlx_fast.py
- https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html
- https://ml-explore.github.io/mlx/build/html/usage/compile.html
- https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html
- https://github.com/ml-explore/mlx-swift-lm/issues/124
- https://github.com/kyutai-labs/hibiki
- https://huggingface.co/kyutai/hibiki-1b-mlx-bf16

#### Uncertain Fields

- expected_signal
- implementation_implications
- measurement_plan
- open_questions
- risks

### Decide mobile-live model profile strategy

_Source: `Decide_mobilelive_model_profile_strategy.json`_

#### Basic Info

**Item Name:** Decide mobile-live model profile strategy

**Research Focus:** Decide whether the repo should add a smaller mobile-live Hibiki profile if the current `anquachdev/hbk-zero-3b-mlx-q4` path cannot get close to realtime on device, using upstream Kyutai mobile guidance and local benchmark evidence.

**Summary:** Recommendation: yes, plan for a second mobile-live profile if the current q4 path stays far below realtime after substage profiling and one focused optimization pass. Upstream Kyutai smartphone guidance is tied to the smaller original Hibiki-M path on iPhone 16 Pro, while the current local repo is pinned to a 3B Hibiki-Zero q4 profile whose official upstream materials emphasize GPU use rather than iPhone deployment. The main tradeoff is that a mobile-live profile should improve throughput materially, but it may reduce speaker similarity and audio fidelity because the upstream mobile Hibiki-M variant uses 8 RVQ quantizers instead of 16, and it may also narrow language coverage if the mobile-live fallback is original Hibiki-M rather than Hibiki-Zero.

#### Technical Features

**Relevant Primary Sources:**

- Title: Kyutai Hibiki README | Url: https://github.com/kyutai-labs/hibiki | Source Type: official_repo | Notes: - States that Hibiki produces text and audio tokens at 12.5 Hz.
- States that the MLX-Swift path can run on an iPhone and was tested on iPhone 16 Pro, but is experimental.
- Lists two released sizes: Hibiki 2B with 16 RVQ per stream and Hibiki 1B with 8 RVQ per stream, described as ideal for on-device inference.
- Title: Hugging Face model card for kyutai/hibiki-1b-mlx-bf16 | Url: https://huggingface.co/kyutai/hibiki-1b-mlx-bf16 | Source Type: official_model_card | Notes: - Identifies the model as Hibiki-M for mobile.
- Describes a 1.7B-parameter hierarchical Transformer producing tokens at 12.5 Hz.
- Says the model is intended for real-time streaming translation settings.
- Title: High-Fidelity Simultaneous Speech-To-Speech Translation | Url: https://arxiv.org/html/2502.03382v2 | Source Type: paper | Notes: - Reports that the smaller distilled Hibiki-M runs faster than real time on an iPhone 16 Pro.
- States Hibiki-M is competitive with Seamless on short-form and long-form translation.
- Explains that Hibiki-M has lower long-form speaker similarity because it models 8 quantizers instead of 16, giving half the audio bitrate.
- Title: kyutai-labs/moshi-swift README | Url: https://github.com/kyutai-labs/moshi-swift | Source Type: official_repo | Notes: - Frames the repo as an experimental iOS implementation for Moshi and Hibiki variants.
- Says the main goal is experimentation on iOS devices and the included iOS app is only a proof of concept.
- Uses `make run-1b`, which aligns with the smaller mobile-oriented profile rather than a larger profile.
- Title: Kyutai Hibiki-Zero README | Url: https://github.com/kyutai-labs/hibiki-zero | Source Type: official_repo | Notes: - Describes Hibiki-Zero as a real-time multilingual speech translation model.
- States that the released model is 3B and requires an NVIDIA GPU with 8 GB to 12 GB VRAM.
- Does not present an official iPhone or MLX mobile deployment path.
- Title: Hugging Face model card for kyutai/hibiki-zero-3b-pytorch-bf16 | Url: https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16 | Source Type: official_model_card | Notes: - Describes Hibiki-Zero as a 3B-parameter model at 12.5 Hz and about 2.2 kbps audio bitrate.
- Confirms current official release scope is multilingual X-to-English real-time translation.
- Reinforces that the current official checkpoint family is large and GPU-oriented.

**Local Code Or Docs To Check:**

- Path: S2STranslate/ModelRuntimeManifest.json | Reason: The app is currently hard-wired to `anquachdev/hbk-zero-3b-mlx-q4`, so any mobile-live plan requires profile-aware artifact selection.
- Path: S2STranslate/ModelArtifactPreparation.swift | Reason: Artifact preparation is pinned to one repo and one required Hibiki q4 weight file.
- Path: docs/hibiki-mlx-porting-notes.md | Reason: Documents that the current Swift runtime was built around a q4 Hibiki-Zero-style artifact with 32 total codebooks and 16 generated codebooks.
- Path: .scratch/real-file-benchmark/latest/benchmark.md | Reason: Provides the latest measured generated realtime factor and per-stage timings for the current path.
- Path: .scratch/hibiki-ios-mlx/issues/28-investigate-choppy-real-time-translation-playback.md | Reason: Confirms that live playback failure is mainly model starvation rather than only AVAudio configuration.
- Path: docs/real-file-french-english-smoke-test.md | Reason: Defines the current interpretation of generated realtime factor and the model-flow timing report.
- Path: S2STranslate/ContentView.swift | Reason: Shows the current playback sink choices that would need to become profile- or capability-aware.

#### Performance Metrics

**Realtime Budget Impact:**

Current Local State: The latest local benchmark generates 5.920 s of output audio in 27.463 s, for a generated realtime factor of 0.216x.; Frame Budget Math: - Hibiki and Mimi operate at about 12.5 Hz, so one frame is about 80 ms of audio.
- Current measured average per generated frame is about 21.846 ms Mimi encode + 328.986 ms Hibiki step + 22.044 ms Mimi decode = 372.876 ms total.
- Bare realtime requires the full pipeline to stay at or under about 80 ms per frame.
- A practical live target around 1.25x requires about 64 ms per frame.; Implication For Current Q4 Path: At 0.216x, the current path needs about 4.63x more throughput for bare realtime and about 5.79x for a practical live target. With encode and decode already near 44 ms combined, the remaining Hibiki budget would be about 36 ms for bare realtime or about 20 ms for a practical target, far below the current 328.986 ms Hibiki step average.; Decision Signal: These numbers make the current q4 path look like a buffered or offline profile today, not a credible live profile.

**Measurement Plan:**

- First split the current `Hibiki step` bucket into substages: main transformer, text sampling, Depformer loop, token extraction, MLX evaluation boundaries, and Swift materialization overhead.
- Repeat the benchmark on the same fixture after one low-level cleanup pass to see whether the q4 path moves materially toward 1.0x or remains far below budget.
- If a candidate mobile-live profile is available, run the same benchmark harness and device playback path on the same fixture and hardware.
- Compare each profile on generated realtime factor, p50 and p95 stage times, memory pressure, startup time, visible-text cadence, underrun count, and final WAV artifact completeness.
- Record profile-specific language coverage and quality observations, especially speaker similarity, naturalness, and translation adequacy.

#### Milestone Significance

**Recommended Next Step:** Proceed with a dual-profile plan, but gate it on one short evidence pass: split `Hibiki step`, apply one profiler-led cleanup to the current q4 path, and if the resulting device benchmark is still clearly sub-realtime, add a separate mobile-live profile rather than continuing to market the q4 path as live-capable. In practice, keep the current q4 3B path as buffered or offline by default and treat the smaller profile as the live experiment path.

**Acceptance Criteria:**

- The repo can describe at least two model intents clearly: a live-capable mobile profile and a buffered or offline higher-capability profile.
- The current q4 profile is no longer assumed to be live by default when its measured generated realtime factor is below 1.0x.
- A candidate mobile-live profile is evaluated on the same fixture and device with the same benchmark harness.
- The mobile-live profile reaches at least 1.0x generated realtime factor on target hardware, with a practical goal of at least 1.25x.
- Live playback with the chosen mobile-live profile finishes without repeated underruns after a bounded prebuffer.
- Profile selection and UI copy make quality and language-coverage differences explicit.

**Sources:**

- Title: Local benchmark: real file model flow | Url: file://.scratch/real-file-benchmark/latest/benchmark.md | Source Type: local | Why It Matters: Primary evidence for the current 0.216x generated realtime factor and per-stage timing.
- Title: Local issue 28: choppy real-time playback investigation | Url: file://.scratch/hibiki-ios-mlx/issues/28-investigate-choppy-real-time-translation-playback.md | Source Type: local | Why It Matters: Confirms that live playback failure comes mainly from model starvation after the prebuffer drains.
- Title: Local model manifest | Url: file://S2STranslate/ModelRuntimeManifest.json | Source Type: local | Why It Matters: Shows the current repo is pinned to one q4 3B profile.
- Title: Local Hibiki MLX porting notes | Url: file://docs/hibiki-mlx-porting-notes.md | Source Type: local | Why It Matters: Documents the current runtime assumptions around q4 weights and generated codebooks.
- Title: Kyutai Hibiki README | Url: https://github.com/kyutai-labs/hibiki | Source Type: official
- Title: Hugging Face: kyutai/hibiki-1b-mlx-bf16 | Url: https://huggingface.co/kyutai/hibiki-1b-mlx-bf16 | Source Type: official
- Title: High-Fidelity Simultaneous Speech-To-Speech Translation | Url: https://arxiv.org/html/2502.03382v2 | Source Type: paper
- Title: kyutai-labs/moshi-swift README | Url: https://github.com/kyutai-labs/moshi-swift | Source Type: official
- Title: Kyutai Hibiki-Zero README | Url: https://github.com/kyutai-labs/hibiki-zero | Source Type: official
- Title: Hugging Face: kyutai/hibiki-zero-3b-pytorch-bf16 | Url: https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16 | Source Type: official

#### Uncertain Fields

- expected_signal
- implementation_implications
- open_questions
- risks

# Two-Solution Research: Realtime Output Strategy for Sub-Realtime Hibiki Model Flow

Status: research-draft
Date: 2026-06-13

## Executive Summary

The current Model-Backed Translation Path is producing real French-to-English text and generated English audio, but it is not producing audio fast enough for live playback. The latest local 40-source-chunk benchmark produced `5.920s` of generated audio in `27.463s`, a generated realtime factor of `0.216x`. Issue 28 device logs measured a similar post-prebuffer production rate around `0.23x`. A live player needs at least `1.0x`; a practical target is closer to `1.2x-1.5x` so the playback queue can survive scheduling jitter.

This research recommends two parallel solutions:

1. **Solution A: Capability-Gated Buffered Output**
   Make the app honest and usable immediately. Detect sub-realtime model flow, avoid live playback when it will underrun, and route the generated output to smooth deferred or buffered playback plus inspectable text/audio artifacts.

2. **Solution B: Realtime Recovery Track**
   Treat true live playback as a performance research track. Split Hibiki step timing into substages, compare against upstream mobile assumptions, and optimize only the measured bottleneck until generated realtime factor reaches `>=1.0x`.

These are not competing in the short term. Solution A should land first because it stops broken voice output. Solution B can then run as an evidence-driven optimization program without forcing every demo to sound broken until the model is fast enough.

## Current Evidence

### Local benchmark

Authoritative local file:

- `.scratch/real-file-benchmark/latest/benchmark.md`

Latest 40-source-chunk run:

- Source duration: `3.200s`
- Generated audio duration: `5.920s`
- Processing time: `27.463s`
- Generated realtime factor: `0.216x`
- Mimi encode average: `21.846ms`
- Hibiki step average: `328.986ms`
- Mimi decode average: `22.044ms`

The per-frame math is the key signal. Mimi/Hibiki operates at a 12.5 Hz cadence, so one frame represents about `80ms` of audio. Current measured average per generated frame is roughly:

```text
21.846ms encode + 328.986ms Hibiki + 22.044ms decode = 372.876ms
```

That is about:

```text
80 / 372.876 = 0.215x
```

This matches the report-level generated realtime factor. To reach `1.0x`, the model-flow average must be at or below `80ms` per generated frame. To reach a practical `1.25x`, it must be at or below `64ms` per frame.

### Device playback diagnosis

Authoritative local file:

- `.scratch/hibiki-ios-mlx/issues/28-investigate-choppy-real-time-translation-playback.md`

Issue 28 concluded:

- Source/sample playback can schedule the whole file and keep pending audio healthy.
- Real-file translation starts playback after a 2 second prebuffer.
- After playback begins, each `80ms` decoded chunk arrives roughly every `350ms`.
- The prebuffer drains, then every new chunk plays immediately and the queue drains again.
- The run ends with repeated underruns.

This means the dominant failure is model starvation. CoreAudio warnings still deserve cleanup, but they are not the first explanation for broken voice output.

### External primary-source constraints

Primary sources line up with the local measurements:

- Kyutai describes Hibiki as streaming/simultaneous translation that processes source speech and generates target speech chunk by chunk.
- Hibiki produces text and audio tokens at a constant `12.5Hz`, which implies an `80ms` frame cadence.
- The Hibiki model card describes the mobile variant as a hierarchical Transformer producing speech and text tokens at `12.5Hz` with audio at about `1.1kbps`.
- The `moshi-swift` README describes Mimi as processing `24kHz` audio into a `12.5Hz` representation with `80ms` frame latency.
- Kyutai's Hibiki README says the MLX-Swift path can run on an iPhone, but also calls that code experimental and specifically mentions iPhone 16 Pro testing.
- MLX documentation emphasizes lazy evaluation and warns that overly frequent evaluations can be inefficient because graph evaluation has fixed overhead. This is directly relevant to per-frame Swift MLX loops.

## Realtime Budget

Use the same metric everywhere:

```text
generatedRealtimeFactor = generatedAudioDurationSeconds / processingWallTimeSeconds
```

Interpretation:

- `< 1.0x`: model flow is slower than playback; live playback eventually underruns.
- `1.0x`: barely keeps up under ideal conditions.
- `1.2x-1.5x`: practical live target with room for audio scheduling and UI overhead.
- `2.0x+`: comfortable for experiments and recovery from transient stalls.

At current `0.216x`, the system needs about:

```text
1.0 / 0.216 = 4.63x throughput improvement
1.25 / 0.216 = 5.79x throughput improvement
```

This is too large for "increase prebuffer from 2s to 4s" to be a real fix. A 2s buffer drains after roughly:

```text
2 / (1 - 0.216) = 2.55s of playback after playback starts
```

If the app wants smooth audio with today's speed, it must either wait for most/all output to be generated or switch to a deliberately delayed buffered mode.

## Solution A: Capability-Gated Buffered Output

### Goal

Make the current app produce smooth, inspectable output now, without pretending the Model-Backed Translation Path is live-capable.

This solution accepts that the current runtime is sub-realtime. It changes product behavior so generated English audio is not played in a way that guarantees audible gaps.

### User-visible behavior

For Model-Backed Translation, the Experiment Session chooses an output strategy:

```text
live-capable     -> start live playback with bounded prebuffer
sub-realtime     -> generate output, then play buffered/deferred audio smoothly
audio-disabled   -> generate text/audio artifacts only
```

Default behavior should be `Auto`:

- If measured or predicted generated realtime factor is healthy, use live playback.
- If measured or predicted generated realtime factor is below threshold, use buffered/deferred playback.
- If playback diagnostics show AVAudio cannot sustain already-buffered output, fall back to output-only artifacts and report that playback integration is unhealthy.

The UI should say this through existing observations rather than a large new surface. Example observation strings:

- `Output mode: buffered because generated realtime factor is 0.216x`
- `Live target: >=1.0x minimum, >=1.25x practical`
- `Generated audio will play after enough output is available`

### Implementation shape

Prefer small new types:

```text
RealtimeCapability
OutputStrategy
OutputStrategyDecision
RealtimeOutputPolicy
```

The decision inputs should be simple and testable:

- generated realtime factor, if a prior benchmark or warmup run exists
- live playback pending duration and underrun count, if running
- user-selected mode, if later added
- minimum live threshold
- practical live threshold

Initial policy:

```text
if forcedLive:
    live
else if generatedRealtimeFactor >= practicalLiveTarget:
    live
else if generatedRealtimeFactor >= minimumLiveTarget and playback diagnostics are healthy:
    liveWithWarning
else:
    deferredOrBuffered
```

For today's state, `0.216x` always selects `deferredOrBuffered`.

### Where this maps locally

Existing useful seams:

- `ExperimentSession` owns user-visible lifecycle and observations.
- `ExperimentObservations` already records playback diagnostics and generated output.
- `PlaybackSink` is already the output boundary.
- `DeferredAudioPlaybackSink` already buffers chunks until finish, then schedules smooth playback.
- `BufferedStreamingAudioPlaybackSink` already implements pseudo-streaming with a prebuffer.
- `RealFileFrenchEnglishSmokeTests` already has a real model smoke and benchmark path.

The smallest useful implementation is to make the real-file app path choose `DeferredAudioPlaybackSink` when live capability is known to be below threshold. The more complete version adds a policy object and records the selected mode in observations.

### Recommended acceptance criteria

- The app records an output strategy for each Model-Backed Translation run.
- With generated realtime factor below `1.0x`, the app does not default to live AVAudio playback.
- The app can generate text and WAV artifacts while playback is deferred.
- Deferred playback of generated chunks is smooth when fed already-buffered output.
- Existing playback diagnostics remain visible.
- The docs explain that current generated realtime factor is below live target and that buffered output is intentional.

### Tests

Simulator-friendly:

- Given a simulated `0.216x` capability, `RealtimeOutputPolicy` selects deferred/buffered output.
- Given a simulated `1.3x` capability, policy selects live output.
- Given a simulated playback-only failure, policy selects output-only artifacts.
- Experiment Session observations include selected output strategy.
- Deferred sink receives chunks during generation and only schedules wrapped playback at finish.

Opt-in/device:

- Run real-file generation with deferred playback and confirm the generated WAV is nonempty.
- Feed already-generated decoded chunks through `AVAudioPlaybackSink` and confirm no underruns.
- Feed synthetic 24 kHz mono PCM through `AVAudioPlaybackSink` and confirm no underruns.

### Benefits

- Stops broken voice output quickly.
- Keeps current model work useful for text/audio inspection.
- Preserves the live playback path for future fast configurations.
- Makes demos honest: sub-realtime generation is presented as delayed/buffered output.
- Low technical risk because most seams already exist.

### Risks

- It does not solve simultaneous translation.
- Users may perceive buffered output as a regression if the UI does not explain why live mode is disabled.
- If deferred playback uses the same AVAudio configuration, CoreAudio warnings may still need cleanup when playback begins.

### Recommendation

Implement Solution A first. It is the product-safe path and gives the project a stable baseline while performance work continues.

## Solution B: Realtime Recovery Track

### Goal

Find out whether this model path can reach `>=1.0x` on target hardware, and make every optimization measurable.

This solution does not assume a single magic fix. Current numbers require a roughly `4.6x` throughput improvement for bare realtime and roughly `5.8x` for a practical `1.25x` target. That magnitude means work must be profiler-led.

### Research questions

1. Is the current Swift MLX Hibiki implementation doing extra evaluations, memory reads, scalar conversions, or synchronization inside each frame?
2. Is the `Hibiki step` bucket dominated by main transformer, Depformer, sampling, state/token bookkeeping, or Swift/MLX boundary overhead?
3. Does MLX compilation or better evaluation grouping reduce per-frame overhead?
4. Is the current q4 3B artifact larger/slower than the upstream mobile path assumptions?
5. Does the upstream 1B MLX-Swift path reach realtime on comparable hardware, and should this app expose a smaller model option?
6. Does tail flush use the same performance profile as source-driven generation?

### Implementation shape

First split the benchmark. The current single `Hibiki step` timer is too coarse:

```text
Hibiki step
  main transformer/token logits
  text sampling
  Depformer audio token sampling
  state/cache update
  generated token frame creation
  Swift array extraction/conversion
```

Then produce a benchmark report with:

- generated realtime factor
- total per-frame time
- p50/p95/max for each substage
- MLX evaluation count per frame, if inspectable
- memory peak/cache metrics, if available through MLX Swift
- warm first-frame vs steady-state timings

Only after that, optimize the dominant substage.

### Optimization hypotheses

These are ordered by likely value against the current evidence.

1. **Evaluation-bound MLX loop**
   MLX lazy evaluation can be hurt by too-frequent evaluation or implicit scalar/array materialization. If the Swift step path extracts arrays or scalar items multiple times per frame, consolidate evaluation at the outer frame boundary and avoid premature materialization.

2. **Uncompiled repeated graph**
   Per-frame inference repeatedly executes the same main/depth transformer structure. If MLX Swift supports the needed compile path for this graph shape, compile stable portions or cache callable graph structure. The acceptance test is before/after p50 Hibiki step time on the same fixture.

3. **Depformer dominates**
   Hibiki predicts dependent audio codebooks through a Depth Transformer. If substage timing shows Depformer dominates, investigate whether codebook count, loop structure, or token extraction is causing the gap. This may also validate using an upstream 1B/8-RVQ model for mobile live mode while keeping the larger q4 path for offline/buffered output.

4. **CPU/GPU synchronization or memory pressure**
   If per-frame p95 spikes correlate with array conversion, memory pressure, or GPU synchronization, reduce host/device transfers and add memory diagnostics.

5. **Model-size mismatch**
   Kyutai's public docs describe Hibiki 1B as ideal for on-device inference and the MLX-Swift implementation as tested on iPhone 16 Pro. If the current q4 3B path cannot approach realtime after low-level cleanup, add model selection: smaller mobile model for live experiments, current q4 model for buffered/offline experiments.

### Target budgets

Use these budgets for optimization issues:

```text
Bare live target:
  encode + Hibiki + decode <= 80ms/frame

Practical live target:
  encode + Hibiki + decode <= 64ms/frame

Current latest:
  encode + Hibiki + decode ~= 373ms/frame
```

If Mimi encode and decode stay near `22ms` each, then Hibiki must fall under:

```text
bare live Hibiki budget = 80 - 22 - 22 = 36ms/frame
practical Hibiki budget = 64 - 22 - 22 = 20ms/frame
```

That is an enormous drop from the current `329ms` average. So Solution B should also consider reducing codec overhead, overlapping stages, or using a smaller mobile model. Hibiki-only optimization may not be enough if encode/decode remain unchanged.

### Pipeline alternatives to evaluate

#### B1. Optimize current q4 path

Keep the current model artifact and make the Swift/MLX implementation faster.

Use when:

- profiling shows obvious evaluation/materialization overhead
- current graph is far slower than upstream expectations for similar model size
- q4 path is required for the research goal

Exit criterion:

- generated realtime factor improves meaningfully on the same fixture
- p50 Hibiki step moves toward the 80ms total-frame budget

#### B2. Add a mobile-live model profile

Keep the current q4 path as a buffered/high-capability experiment, but add a smaller mobile model profile for live playback experiments.

Use when:

- current q4 path remains far below realtime after low-level cleanup
- upstream mobile model docs suggest a smaller model is the intended iPhone path
- the app needs a credible live demo sooner than the large model can be optimized

Exit criterion:

- mobile-live profile reaches `>=1.0x` generated realtime factor on target hardware
- the app clearly labels model/profile tradeoffs

### Tests

Automated:

- Benchmark report includes substage timing fields.
- Benchmark report includes generated realtime factor and per-frame budget interpretation.
- A regression test verifies that before/after benchmark comparison can detect improvement or regression.
- Simulated benchmark summaries classify `subRealtime`, `bareRealtime`, and `practicalRealtime`.

Opt-in real:

- Run the same French fixture before and after each optimization.
- Save markdown, JSON, text, and WAV artifacts.
- Compare generated realtime factor, p50, p95, and max timings.
- Check translation text/audio did not disappear while optimizing.

Device:

- Run live-capable profile through `AVAudioPlaybackSink`.
- Confirm pending playback duration does not trend to zero.
- Confirm underrun count stays zero or within an explicitly accepted tolerance.

### Benefits

- Gives a path back to true Streaming Translation.
- Prevents unfocused performance work.
- Can prove when current hardware/model combination is not viable.
- Creates reusable benchmark infrastructure for future model variants.

### Risks

- It may prove the current q4 3B path cannot reach realtime on the target device.
- Substage instrumentation itself can perturb timing if implemented with excessive synchronization.
- MLX Swift compile/evaluation tools may not support every graph shape needed by this model.
- A smaller mobile profile may change output quality, voice similarity, or translation behavior.

### Recommendation

Start Solution B only after Solution A is in place or at least after the app can avoid broken live playback. The first B issue should be profiling, not optimization. The current `Hibiki step` bucket is too coarse to justify code changes.

## Decision Matrix

| Criterion | Solution A: Capability-Gated Buffered Output | Solution B: Realtime Recovery Track |
| --- | --- | --- |
| Fixes broken voice quickly | Yes | No |
| Preserves true live goal | Yes, by gating it | Yes, directly |
| Requires model optimization | No | Yes |
| Risk | Low | Medium to high |
| User-visible improvement | Immediate smooth output | Only after speedups land |
| Research value | Honest capability reporting | Bottleneck discovery and speedups |
| Best first issue | Output strategy policy | Hibiki substage profiler |

## Recommended Plan

1. Land a small output-strategy policy that classifies generated realtime factor against `1.0x` and `1.25x`.
2. Wire the real-file app path to use deferred/buffered output when capability is below `1.0x`.
3. Add observations/docs so users know why playback is delayed.
4. Add playback-only diagnostics to prove AVAudio can play already-buffered output smoothly.
5. Split the benchmark's `Hibiki step` timer into substages.
6. Run one before/after benchmark for any optimization candidate.
7. Decide whether to keep optimizing the current q4 path or add a smaller mobile-live model profile.

## Issue Breakdown Seed

These are issue seeds, not final issue files:

1. **Define realtime output policy and budget**
   - Type: AFK
   - Outcome: `RealtimeOutputPolicy` classifies sub-realtime, bare-realtime, and practical-realtime runs.

2. **Route sub-realtime real-file runs to deferred playback**
   - Type: AFK
   - Outcome: current `0.216x` path produces smooth post-generation playback instead of choppy live playback.

3. **Add playback-only diagnostics checks**
   - Type: AFK
   - Outcome: already-generated chunks and synthetic PCM prove whether AVAudio can play without model starvation.

4. **Split Hibiki benchmark timing into substages**
   - Type: AFK
   - Outcome: benchmark identifies the exact dominant stage inside `Hibiki step`.

5. **Optimize the first measured Hibiki bottleneck**
   - Type: AFK
   - Outcome: one measured performance improvement with before/after benchmark report.

6. **Decide mobile-live model/profile strategy**
   - Type: HITL
   - Outcome: choose current q4 optimization, smaller mobile model profile, or buffered-only model-backed demo for now.

## Primary Sources Consulted

- Kyutai Hibiki README: https://github.com/kyutai-labs/hibiki
- Kyutai Hibiki model card: https://huggingface.co/kyutai/hibiki-1b-mlx-bf16
- Hibiki paper: https://arxiv.org/abs/2502.03382
- Moshi paper: https://arxiv.org/abs/2410.00037
- Moshi Swift README: https://github.com/kyutai-labs/moshi-swift
- MLX lazy evaluation docs: https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html
- MLX compilation docs: https://ml-explore.github.io/mlx/build/html/usage/compile.html
- Apple AVAudioEngine docs: https://developer.apple.com/documentation/avfaudio/avaudioengine
- Apple AVAudioPlayerNode buffer scheduling docs: https://developer.apple.com/documentation/avfaudio/avaudioplayernode/schedulebuffer(_:completioncallbacktype:completionhandler:)

## Local Evidence Consulted

- `.scratch/hibiki-realtime-output-strategy/PRD.md`
- `.scratch/hibiki-ios-mlx/issues/28-investigate-choppy-real-time-translation-playback.md`
- `.scratch/real-file-benchmark/latest/benchmark.md`
- `docs/real-file-french-english-smoke-test.md`
- `S2STranslate/AVAudioPlaybackSink.swift`
- `S2STranslate/StreamingMimiDecode.swift`
- `Tests/S2STranslateCoreTests/RealFileFrenchEnglishSmokeTests.swift`

# PRD: Realtime Output Strategy for Sub-Realtime Hibiki Model Flow

Status: ready-for-agent

## Problem Statement

The current Model-Backed Translation Path can produce French-to-English text and generated English audio, but the generated audio is produced far slower than realtime playback consumes it. Device logs and benchmark runs show generated realtime factors around `0.216x` to `0.242x`, with Hibiki stepping dominating latency. At that speed, a small prebuffer drains quickly and the `AVAudioPlaybackSink` repeatedly underruns, so the user hears broken, choppy voice output.

From the user's perspective, this is a large product and engineering problem rather than a narrow playback bug. The app appears to offer Streaming Translation, but the current runtime cannot sustain smooth live output. The experiment needs a strategy that keeps the app useful and truthful while the model flow is sub-realtime: benchmark the actual capability, avoid misleading live playback when it cannot work, provide a smooth buffered/offline output path, and identify the bottlenecks that would need to improve before true simultaneous playback can be offered again.

## Solution

Build a Realtime Output Strategy around the Experiment Session that measures generated output speed, chooses an appropriate playback mode, and exposes clear diagnostics.

The first experience should stop treating all Model-Backed Translation runs as live-playable. If the current benchmark or live diagnostics show the model flow is below realtime, the app should use a buffered or post-generation playback mode that produces smooth audio instead of broken audio. The UI should make this state explicit: the model can generate output, but current generation speed is below the threshold required for simultaneous playback.

The system should keep true Streaming Translation as a target, not pretend it has been achieved. It should define the required generated realtime factor, record stage-level model timings, distinguish model starvation from AVAudio failures, and make before/after performance work measurable. A completed version of this PRD should let a developer answer three questions from one run:

- Can this Model-Backed Translation Path play smoothly in live mode on this device?
- If not, which fallback mode gives the user a clean generated voice output?
- Which model stage must improve before live playback is viable?

## User Stories

1. As a mobile ML developer, I want the app to measure generated realtime factor for a Model-Backed Translation Path, so that I know whether generated audio can keep up with playback.
2. As a mobile ML developer, I want the app to define the minimum realtime target as `1.0x`, so that live playback has an explicit pass/fail threshold.
3. As a mobile ML developer, I want a practical target above `1.0x`, so that playback has headroom for scheduling jitter and device variability.
4. As a mobile ML developer, I want the app to show when a run is sub-realtime, so that I do not mistake choppy playback for an AVAudio-only bug.
5. As a mobile ML developer, I want the app to avoid starting live playback when the model flow is far below realtime, so that the generated voice does not sound broken.
6. As a researcher, I want smooth buffered playback of generated audio after enough output has accumulated, so that I can inspect voice quality even when true simultaneous playback is not viable.
7. As a researcher, I want generated text to remain visible during sub-realtime generation, so that I can inspect translation content while audio is still being produced.
8. As a researcher, I want generated audio to be written to an inspectable artifact, so that I can replay and compare output outside the live app.
9. As a developer, I want a benchmark report that includes source duration, generated audio duration, processing time, generated realtime factor, and stage timing summaries, so that performance regressions are obvious.
10. As a developer, I want Hibiki step timing split into meaningful substages, so that I can tell whether the main transformer, text sampling, Depformer, state management, or decode handoff is the bottleneck.
11. As a developer, I want Mimi encode and Mimi decode timings in the same report, so that I can avoid over-focusing on Hibiki when codec stages are actually slow.
12. As a developer, I want playback diagnostics to show scheduled, completed, and pending audio duration, so that I can prove whether playback underruns are caused by model starvation.
13. As a developer, I want a playback-only diagnostic using already-generated output, so that I can test AVAudio playback separately from model generation.
14. As a developer, I want a synthetic steady PCM playback diagnostic, so that I can verify the `Playback Sink` can sustain 24 kHz mono output without depending on MLX.
15. As a developer, I want the Experiment Session to choose between live playback, buffered playback, and output-only modes using measured capability, so that the app behavior matches the device's actual performance.
16. As a developer, I want the selected playback mode to be visible in observations, so that device logs and screenshots explain what happened.
17. As a developer, I want the app to estimate required initial buffer when generation is below realtime, so that buffer-based workarounds are evaluated honestly.
18. As a developer, I want the app to preserve the Input End Flush behavior in buffered and output-only modes, so that file-mode translation can still produce delayed text/audio after source input ends.
19. As a developer, I want benchmark output to be stable enough for before/after optimization comparisons, so that each performance change can be judged against the same fixture.
20. As a maintainer, I want the workaround strategy separated from the original choppy-playback investigation, so that issue 28 can close once the cause is understood and follow-up work can proceed cleanly.
21. As a maintainer, I want simulator-friendly tests for mode selection and report generation, so that most strategy logic can be validated without a physical device.
22. As a maintainer, I want device-only checks for AVAudio behavior, so that platform integration remains covered without blocking all contributors.
23. As a maintainer, I want unsupported states to be explicit, so that the app does not imply production-quality simultaneous translation before the model flow reaches realtime.
24. As a maintainer, I want the README or experiment workflow docs to explain the current realtime limitation, so that new contributors understand why the app may choose buffered playback.

## Implementation Decisions

- The PRD treats issue 28 as the diagnosis source: the dominant failure mode is model starvation, not the first-order AVAudio scheduling path.
- The strategy will define generated realtime factor as generated audio duration divided by processing wall time.
- Live playback requires generated realtime factor greater than or equal to `1.0x`; a practical target should be higher than `1.0x` to tolerate scheduling jitter.
- Current measured baselines around `0.216x` to `0.242x` are considered sub-realtime and should not be presented as smooth simultaneous playback.
- The Experiment Session should support at least three output strategies for Model-Backed Translation: live playback attempt, buffered playback, and output-only generation with saved artifacts.
- The app should prefer smooth, honest output over early but broken audio when measured generation speed is far below realtime.
- Buffered playback should wait for enough generated audio to avoid immediate underruns, and should communicate that it is buffered rather than simultaneous.
- Output-only generation should still produce text and audio artifacts for inspection.
- The benchmark report should remain available outside the app as an automated, opt-in test path for local artifact runs.
- Benchmark reports should include enough structured data for later automation to compare current runs against previous baselines.
- Hibiki timing should be split below the current single "Hibiki step" bucket before optimization issues are assigned.
- Playback-only diagnostics should reuse the same `Playback Sink` boundary as the app instead of creating a separate audio path.
- Synthetic PCM diagnostics should verify playback format and scheduling independent of MLX, Mimi, and Hibiki.
- The UI should expose the selected output strategy through existing Experiment Session observations rather than adding a large new control surface.
- The existing French-to-English file fixture remains the primary reproducible input for strategy validation.
- The strategy should not depend on network availability after model artifacts are already prepared.
- The strategy should preserve explicit model artifact and runtime errors; fallback playback behavior should not hide real failures.
- Performance optimization is allowed only after the benchmark can identify which model stage is responsible for the gap.
- Any optimization work should report before/after generated realtime factor and stage timing on the same fixture.

## Testing Decisions

- Good tests should assert user-visible Experiment Session behavior and generated reports, not private implementation details.
- The highest automated test boundary is the Experiment Session: given measured or simulated generation speed, it should select and report the expected output strategy.
- The existing real-file smoke and benchmark tests are the prior art for opt-in full-model validation with local MLX artifacts.
- Simulator-friendly tests should cover strategy selection, benchmark summary formatting, underrun interpretation, and required-buffer estimation without loading MLX weights.
- Playback sink tests should cover diagnostics for scheduled duration, completed duration, pending duration, and underrun count.
- A playback-only device checklist should validate already-buffered decoded audio through the real AVAudio sink.
- A synthetic PCM device checklist should validate steady 24 kHz mono playback through the real AVAudio sink.
- Real model benchmark tests should stay opt-in behind environment variables because they require large local artifacts, significant memory, and device-specific performance.
- Benchmark tests should write inspectable markdown, JSON, text, and WAV artifacts so humans can verify both performance and output quality.
- Regression tests for sub-realtime handling should prove that the app avoids broken live playback when generated realtime factor is far below `1.0x`.
- Regression tests for live-capable behavior should use simulated generation speed at or above target, so that the live playback path remains covered even before the real model reaches target speed.

## Out of Scope

- Guaranteeing that the current 3B q4 Hibiki model reaches realtime on all iOS devices.
- Replacing the model with a smaller or different model family.
- Training, fine-tuning, or changing model weights.
- Adding new language pairs beyond the current French-to-English target.
- Building a production translation UX.
- Removing the existing live playback path entirely.
- Solving every CoreAudio channel-map or converter warning unless playback-only diagnostics prove they affect smooth buffered output.
- Perfect translation quality evaluation.
- Server-backed inference.
- App Store readiness, analytics, subscriptions, or onboarding.

## Further Notes

- Issue 28 measured repeated playback underruns after the 2 second prebuffer drained. The decoded chunk cadence was roughly 80 ms of generated audio every 350 ms, approximately `0.23x` realtime.
- The benchmark harness measured `0.216x` generated realtime factor on a 40-source-chunk run with tail flush. Hibiki step averaged roughly `329 ms` per generated frame, compared with roughly `22 ms` for Mimi encode and `22 ms` for Mimi decode.
- A 20-source-chunk speed-only benchmark with tail flush disabled can produce an empty text artifact because visible text may arrive after the first few seconds of source audio. Text-output probes should keep tail flush enabled and use enough source chunks.
- Buffering alone is not a satisfying simultaneous-translation solution at the current speed. A run around `0.23x` realtime would need a very large initial buffer to avoid underruns, which changes the product behavior from simultaneous playback to delayed or offline playback.
- This PRD should become the parent for follow-up issues that define the realtime budget, split Hibiki profiling, add playback-only diagnostics, implement graceful fallback behavior, and then optimize the measured bottleneck.

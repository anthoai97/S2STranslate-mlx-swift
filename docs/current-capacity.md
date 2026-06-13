# Current Capacity

Last updated: 2026-06-13

## Summary

The current q4 Hibiki real-file path can generate French-to-English text and English audio artifacts, but it is still sub-realtime. After the first measured optimization, the latest 40-source-chunk benchmark produces generated audio at `0.311x` realtime. That is a material improvement from the previous `0.230x` baseline, but it remains below the `1.0x` hard minimum for live playback and below the `1.25x` practical-realtime target.

The app should therefore treat the q4 3B path as a deferred or offline-quality profile, not as smooth simultaneous speech playback.

## Realtime Policy

`RealtimeOutputPolicy` currently classifies generated realtime factor as:

| Capability | Generated realtime factor | Default interpretation |
| --- | ---: | --- |
| Sub-realtime | `< 1.0x` | Do not present as smooth live playback. |
| Bare realtime | `>= 1.0x` and `< 1.25x` | Live may work, but with little scheduling headroom. |
| Practical realtime | `>= 1.25x` | Default-live candidate with jitter headroom. |

Known sub-realtime real-file runs are routed through deferred playback by default, while generated text and saved audio artifacts remain available for inspection.

## Latest Benchmark

Fixture:

- Model revision: `558daadd9272df9432642783b57b02756ff34d5b`
- Source: French Europarl short fixture
- Source chunks: `40`
- Tail flush: enabled
- Benchmark artifact: `.scratch/real-file-benchmark/issue-36/after-topk-smallonly/benchmark.md`
- Comparison artifact: `.scratch/real-file-benchmark/issue-36/comparison.md`

Latest measured output:

| Metric | Value |
| --- | ---: |
| Generated realtime factor | `0.311x` |
| Source duration | `3.200s` |
| Generated audio duration | `5.680s` |
| Processing time | `18.266s` |
| Hibiki steps | `71` |
| Visible text characters | `62` |
| Translation text bytes | `62` |
| Translation wav bytes | `549376` |

## Stage Capacity

Current average stage timings from the latest benchmark:

| Stage | Avg |
| --- | ---: |
| Mimi encode | `22.045 ms` |
| Tail Mimi encode | `20.912 ms` |
| Hibiki step | `212.749 ms` |
| Mimi decode | `22.594 ms` |

Hibiki remains the dominant stage. Its current substage averages are:

| Hibiki substage | Avg |
| --- | ---: |
| Main transformer evaluation | `57.063 ms` |
| Text logits extraction | `0.089 ms` |
| Text sampling | `10.718 ms` |
| Depformer evaluation | `87.412 ms` |
| Depformer logits extraction | `0.901 ms` |
| Depformer sampling | `56.267 ms` |
| State/cache updates | `0.049 ms` |
| Generated frame construction | `0.000 ms` |

## Recent Improvement

The first measured bottleneck was text sampling. Before optimization, the sampler sorted the full logits array even when only a small `topK` candidate set was needed. The current implementation keeps the same sampler behavior, but uses a bounded ranked candidate list for small top-k requests such as text `topK = 25`. Larger top-k requests keep the full sort path because the intermediate benchmark showed bounded insertion was slower for Depformer `topK = 250`.

Measured before/after:

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Generated realtime factor | `0.230x` | `0.311x` | `+35.1%` |
| Processing time | `26.066s` | `18.266s` | `-29.9%` |
| Hibiki step avg | `307.241 ms` | `212.749 ms` | `-30.8%` |
| Text sampling avg | `107.483 ms` | `10.718 ms` | `-90.0%` |

The optimized benchmark still produced nonempty translation text and generated audio with tail flush enabled.

## Current Capability Boundaries

Supported now:

- Reproducible real-file French-to-English benchmark runs with JSON, markdown, text, and WAV artifacts.
- Realtime capability classification through one policy definition.
- Deferred playback routing for known sub-realtime q4 real-file runs.
- Experiment observations that report selected output strategy and realtime capability.
- Playback-only diagnostics for decoded chunks and synthetic 24 kHz mono PCM.
- Hibiki substage timings with explicit MLX evaluation boundaries.

Not yet supported:

- Smooth live playback for the current q4 3B path.
- A confirmed mobile-live model profile.
- A product claim of simultaneous speech translation for the measured q4 path.

## Remaining Decision

The remaining PRD child issue is `#37 Decide Mobile-Live Model Profile Strategy`, currently labeled `ready-for-human`. The next decision is whether to keep the q4 3B path as buffered/offline by default and evaluate a separate smaller mobile-live profile candidate through the same benchmark and playback diagnostics.

# Hibiki Paper / Reference Implementation Capability Comparison

Last updated: 2026-06-13

## Executive Summary

The Hibiki paper and `ref/moshi-swift` describe the target shape for true simultaneous speech-to-speech translation: a constant-framerate streaming loop where Mimi encodes 24 kHz audio into 12.5 Hz token frames, Hibiki samples text/audio tokens per frame, Mimi decodes generated audio, and the app plays it immediately.

`S2STranslate` is now architecture-aligned enough to run the real q4 file flow and produce text/WAV artifacts, but it is not yet live-capable for the current packaged profile. The latest measured generated realtime factor is `0.311x`, so the current q4 3B path needs about `3.2x` more end-to-end throughput to reach bare realtime and about `4.0x` to reach the app's practical-realtime threshold.

## Reference Targets

### Hibiki Paper

- Model goal: simultaneous S2ST and S2TT with streaming source input, streaming target output, and joint text/audio token generation.
- Codec frame contract: Mimi uses 24 kHz audio and a 12.5 Hz representation, so each token frame represents about 80 ms.
- Inference policy: constant-framerate temperature sampling, not a complex wait/read/write policy.
- Reported mobile target: distilled Hibiki-M remains faster than realtime on an iPhone 16 Pro for a minute of inference, including batch size 2 for classifier-free guidance.
- Reported quality metrics: ASR-BLEU, LAAL latency, speaker similarity, audio quality/naturalness, and human evaluation.

### `ref/moshi-swift`

- Reference implementation goal: experimental MLX Swift implementation for iOS experimentation.
- It includes fully streaming Mimi and support for Moshi/Hibiki variants.
- The live loop shape is:
  - microphone PCM
  - `mimi.encodeStep`
  - per-token-frame `LMGen.step`
  - `gen.lastAudioTokens`
  - `mimi.decodeStep`
  - `AudioPlayer.send`
- Sampling stays inside MLX arrays through `categorical`, `softmax`, `argSort`, top-p, or top-k helpers.

## Our Current Capability

Current packaged profile:

- Model repo: `anquachdev/hbk-zero-3b-mlx-q4`
- Revision: `558daadd9272df9432642783b57b02756ff34d5b`
- Required artifacts: `config.json`, `hibiki.q4.safetensors`, Mimi weights, SentencePiece tokenizer.
- Runtime assumptions: q4, group size 32, `hidden_scale = 6`, `kv_repeat = 2`, `rope_concat`, Depformer output LayerNorms, 16 generated audio codebooks.

Latest benchmark:

| Metric | Value |
| --- | ---: |
| Generated realtime factor | `0.311x` |
| Source duration | `3.200s` |
| Generated audio duration | `5.680s` |
| Processing time | `18.266s` |
| Hibiki steps | `71` |
| Hibiki step avg | `212.749 ms` |
| Mimi encode avg | `22.045 ms` |
| Mimi decode avg | `22.594 ms` |

The app policy is correct for this measurement: default to deferred playback, allow diagnostic live playback only when explicitly toggled, and do not claim smooth simultaneous speech output for this profile.

## Capability Matrix

| Dimension | Paper target | `ref/moshi-swift` | `S2STranslate` now | Gap |
| --- | --- | --- | --- | --- |
| Streaming contract | 24 kHz, 12.5 Hz Mimi frames, streaming source and target | Implements streaming Mimi encode/decode and live app loop | Implements MLX Mimi streaming encode/decode and real-file flow | Flow exists, but not yet fast enough for live default |
| Inference policy | Constant-framerate temperature sampling | `LMGen.step` per frame, immediate `lastAudioTokens` decode | Same conceptual main-step plus Depformer flow | Our sampling extracts logits to Swift arrays |
| Mobile realtime | Hibiki-M > `1.0x` on iPhone 16 Pro | iOS proof of concept intended for device experiments | Latest q4 path is `0.311x` | Need separate mobile-live profile or major optimization |
| Practical playback | Continuous target speech | AudioPlayer ring buffer plays decoded chunks | `RealtimeOutputPolicy` defaults sub-realtime runs to deferred playback | Correct product behavior, but not paper-level experience |
| Voice transfer / CFG | Speaker-conditioning labels plus CFG; batch size 2 for CFG | README claims Hibiki variant support, but no obvious CFG UX in inspected loop | `voiceTransferEnabled` exists in config but is not wired/measured | Need explicit CFG/profile decision before claiming voice transfer |
| Quality evaluation | ASR-BLEU, LAAL, speaker similarity, naturalness/human eval | Perf traces and app demo paths | Artifact health, text length, WAV size, stage timing | Need quality metrics before comparing paper quality |
| Model-family support | Full Hibiki and smaller Hibiki-M | Advertises support for all variants | One hard-coded q4 profile | Issue #37 should choose mobile-live profile strategy |

## Performance Gap

The current path produces one generated audio frame per Hibiki step. Since Mimi is 12.5 Hz, a live system must sustain roughly one generated frame every 80 ms, with headroom for encode, decode, audio scheduling, and UI.

Current measured averages:

| Substage | Avg |
| --- | ---: |
| Main transformer evaluation | `57.063 ms` |
| Text sampling | `10.718 ms` |
| Depformer evaluation | `87.412 ms` |
| Depformer sampling | `56.267 ms` |
| Mimi decode | `22.594 ms` |

Hibiki alone averages `212.749 ms` per step, before considering source encode and playback scheduling. That is the core reason live output is not reliable yet.

The strongest implementation-level clue from the comparison is sampling placement. `ref/moshi-swift` samples on MLX arrays, while our runtime extracts logits to Swift arrays for text and every Depformer slice. The text sampler optimization already improved generated realtime factor from `0.230x` to `0.311x`; Depformer sampling remains a visible `56.267 ms` average per generated frame.

## Recommended Next Tracer Bullets

1. Evaluate a mobile-live profile, preferably the smallest available Hibiki/Hibiki-M-compatible profile, through the same real-file benchmark harness.
2. Add a benchmark axis for batch size 1 vs batch size 2 so CFG cost is visible before voice-transfer claims.
3. Prototype MLX-native sampling for Depformer and text to remove host array extraction from the hot path.
4. Add paper-adjacent quality metrics: ASR transcription of generated WAV, BLEU against fixture reference when available, and a latency metric comparable to LAAL.
5. Update small documentation drift: `docs/hibiki-mlx-porting-notes.md` still says full real-file smoke is unverified, and `ContentView` still routes with the older conservative `0.23x` constant while the latest benchmark is `0.311x`.

## Sources

- Paper: https://arxiv.org/html/2502.03382v2 and https://arxiv.org/pdf/2502.03382
- Reference Swift implementation: `ref/moshi-swift/README.md`, `ref/moshi-swift/MoshiLib/LM.swift`, `ref/moshi-swift/MoshiLib/Utils.swift`, `ref/moshi-swift/Moshi/ContentView.swift`
- Current capacity: `docs/current-capacity.md`
- Benchmark artifact: `.scratch/real-file-benchmark/issue-36/after-topk-smallonly/benchmark.md`
- Current runtime: `S2STranslate/MLXHibikiRuntime.swift`, `S2STranslate/MLXHibikiModel.swift`, `S2STranslate/RealtimeOutputPolicy.swift`

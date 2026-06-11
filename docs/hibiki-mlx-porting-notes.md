# Hibiki MLX Porting Notes

Issue: `.scratch/hibiki-ios-mlx/issues/22-implement-mlx-hibiki-model-load-and-step.md`

## References

- `ref/hibiki-zero-mlx/src/infer_mlx_fast.py`
  - Loads `config.json`, `hibiki.q4.safetensors`, and `tokenizer_spm_48k_multi6_2.model`.
  - Applies MLX quantization with `bits=4` and `group_size=32` before strict weight load.
- `ref/hibiki-zero-mlx/hibiki_zero/inference.py`
  - Documents streaming/file inference flow and post-input flushing behavior.
- `ref/moshi-swift/MoshiLib/LM.swift`
  - Provides the closest Swift LM/Depformer structure for cache setup, main model step, text sampling, and generated audio-token frame emission.

## Hibiki-Zero Deltas

- `hidden_scale=6` drives feed-forward dimensions for the main transformer and Depformer.
- `kv_repeat=2` requires grouped-query attention in the main transformer.
- `rope_concat` maps to RoPE with non-interleaved layout.
- Depformer audio logits require per-slice output `LayerNorm` before each `linear_out`.

## Current Swift Boundary

- `MLXHibikiInferenceSession` validates prepared config, q4 weights, and tokenizer artifacts.
- `MLXHibikiDefaultRuntimeEngine` enforces q4 group size 32 and the Hibiki-Zero architecture deltas.
- `MLXMimiTransformer` now supports the main Hibiki transformer attention deltas: grouped-query attention with `kv_repeat=2` and `rope_concat` RoPE layout.
- `MLXHibikiModelConfig` maps `config.json` into the real LM/Depformer topology: main transformer, Depformer transformer, text/audio vocab sizes, 32 total audio codebooks, 16 generated codebooks, 16 source codebooks, delays, and per-step Depformer weight schedule.
- `MLXHibikiLanguageModel` owns the real graph shell shapes for text/audio embeddings, main transformer, output norm/head, and per-slice Depformer modules.
- `MLXHibikiGraphParameterApplier` maps the converted MLX q4 artifact into 2,015 graph tensors: quantized embedding/linear groups (`weight`, `scales`, `biases`) plus dense norm tensors, including the Hibiki-specific per-slice Depformer output norms.
- `MLXHibikiLanguageModel.mainStep` now executes the cached main LM path: text/audio embeddings, main transformer, output norm, and text logits.
- `MLXHibikiDefaultRuntimeEngine.step` keeps Hibiki's delayed sequence state, greedily/top-k samples text logits, executes the sequential Depformer slices, and returns generated audio tokens.
- Full real-file smoke remains unverified on the 3B q4 artifact; real tokenizer decoding and manual audio/text quality checks are still pending.

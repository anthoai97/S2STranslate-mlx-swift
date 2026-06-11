# Implement MLX Hibiki Model Load and Step

Status: in-progress

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Implement the first MLX-backed Hibiki inference session that can load `config.json`, `hibiki.q4.safetensors`, and the tokenizer artifact, construct the model using the documented Hibiki-Zero architecture deltas, and step the model from source Mimi token frames.

This slice should focus on model construction, cache/state setup, and one-step event structure. Sampling details and polished text output can be refined in the next issue.

## Acceptance criteria

- [x] A real `HibikiInferenceSession` implementation can initialize from prepared model artifacts.
- [x] The loader respects q4 MLX safetensors group size 32 semantics.
- [x] The implementation accounts for documented architecture deltas: `hidden_scale=6`, `kv_repeat=2`, `rope_concat`, and per-slice Depformer output LayerNorm.
- [x] The session accepts source Mimi token frames and performs at least one model step.
- [x] Generated event structure includes text-token candidate data and generated target audio-token frames suitable for Mimi decode.
- [ ] Errors from missing config, missing weights, missing tokenizer, unsupported shapes, and MLX load failures surface through the Experiment Session.
- [x] Tests cover initialization and step behavior with fake model seams where real 3B inference is impractical.
- [x] Porting notes reference `ref/hibiki-zero-mlx/src/infer_mlx_fast.py`, `ref/hibiki-zero-mlx/hibiki_zero/inference.py`, and `ref/moshi-swift/MoshiLib/LM.swift`.

## Notes

- This is likely the highest-risk issue. Prefer a narrow traceable implementation over broad optimization.
- Keep `DeterministicHibikiInferenceSession` as the default for fast tests until the real path is stable.
- Current checkpoint adds `MLXHibikiInferenceSession`, validates real prepared config/weights/tokenizer files, enforces q4 group size 32 and Hibiki-Zero architecture deltas, supports executable grouped-query `kv_repeat=2` + `rope_concat` attention in the shared MLX transformer, maps `config.json` into the full LM/Depformer topology, builds an `MLXHibikiLanguageModel` graph shell during default engine load, and maps the converted MLX q4 safetensors into the 2,015 graph parameter tensors expected by the shell.
- The default engine loads/validates and applies q4 safetensors to the graph shell but still throws for the full 3B MLX step graph. Remaining work is replacing the model-step seam with actual LM/Depformer graph execution and token sampling.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/12-implement-real-huggingface-artifact-download-cache.md`
- `.scratch/hibiki-ios-mlx/issues/17-implement-mlx-mimi-streaming-encode.md`

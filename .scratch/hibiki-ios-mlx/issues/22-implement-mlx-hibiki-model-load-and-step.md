# Implement MLX Hibiki Model Load and Step

Status: ready-for-agent

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Implement the first MLX-backed Hibiki inference session that can load `config.json`, `hibiki.q4.safetensors`, and the tokenizer artifact, construct the model using the documented Hibiki-Zero architecture deltas, and step the model from source Mimi token frames.

This slice should focus on model construction, cache/state setup, and one-step event structure. Sampling details and polished text output can be refined in the next issue.

## Acceptance criteria

- [ ] A real `HibikiInferenceSession` implementation can initialize from prepared model artifacts.
- [ ] The loader respects q4 MLX safetensors group size 32 semantics.
- [ ] The implementation accounts for documented architecture deltas: `hidden_scale=6`, `kv_repeat=2`, `rope_concat`, and per-slice Depformer output LayerNorm.
- [ ] The session accepts source Mimi token frames and performs at least one model step.
- [ ] Generated event structure includes text-token candidate data and generated target audio-token frames suitable for Mimi decode.
- [ ] Errors from missing config, missing weights, missing tokenizer, unsupported shapes, and MLX load failures surface through the Experiment Session.
- [ ] Tests cover initialization and step behavior with fake model seams where real 3B inference is impractical.
- [ ] Porting notes reference `ref/hibiki-zero-mlx/src/infer_mlx_fast.py`, `ref/hibiki-zero-mlx/hibiki_zero/inference.py`, and `ref/moshi-swift/MoshiLib/LM.swift`.

## Notes

- This is likely the highest-risk issue. Prefer a narrow traceable implementation over broad optimization.
- Keep `DeterministicHibikiInferenceSession` as the default for fast tests until the real path is stable.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/12-implement-real-huggingface-artifact-download-cache.md`
- `.scratch/hibiki-ios-mlx/issues/17-implement-mlx-mimi-streaming-encode.md`

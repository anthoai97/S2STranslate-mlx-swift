# Implement Real Hugging Face Artifact Download Cache

Status: ready-for-agent

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Replace the proof-of-concept `DemoModelArtifactProvider` with a real cache-first provider that can download the pinned Hibiki-Zero runtime artifacts from Hugging Face, store them in app-local cache storage, and return validated `ModelArtifactHandle` values to the existing preparation path.

This should keep the existing `ModelArtifactProviding` boundary and make the app's **Prepare** step capable of preparing real runtime files without requiring Python or repo-local helper scripts.

## Acceptance criteria

- [ ] The app can discover cached model artifacts by filename and role before attempting a network download.
- [ ] Missing runtime artifacts are downloaded from the pinned `ModelRuntimeManifest` repository and revision.
- [ ] Downloaded artifacts are written atomically into app-local cache storage.
- [ ] Progress and clear failure states are surfaced through the existing Experiment Session preparation path.
- [ ] Required runtime files are limited to `config.json`, `hibiki.q4.safetensors`, `mimi-pytorch-e351c8d8@125.safetensors`, and `tokenizer_spm_48k_multi6_2.model`.
- [ ] Development-only files such as `mlx_hibiki_patch.py` and `verify_mlx_q4.py` are not required by the app at runtime.
- [ ] Tests cover cache hit, successful fake download, partial download cleanup, HTTP/download failure, and manifest filename mismatch.

## Notes

- Keep the demo provider for unit tests or preview-only flows if useful, but the UI should be able to opt into the real provider.
- Avoid loading the model in this issue; this slice only proves artifact availability and cache semantics.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/02-confirm-model-artifact-contract.md`
- `.scratch/hibiki-ios-mlx/issues/03-load-and-cache-model-artifacts.md`


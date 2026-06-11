# Implement Real Hugging Face Artifact Download Cache

Status: complete

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Replace the proof-of-concept `DemoModelArtifactProvider` with a real cache-first provider that can download the pinned Hibiki-Zero runtime artifacts from Hugging Face, store them in app-local cache storage, and return validated `ModelArtifactHandle` values to the existing preparation path.

This should keep the existing `ModelArtifactProviding` boundary and make the app's **Prepare** step capable of preparing real runtime files without requiring Python or repo-local helper scripts.

## Acceptance criteria

- [x] The app can discover cached model artifacts by filename and role before attempting a network download.
- [x] Missing runtime artifacts are downloaded from the pinned `ModelRuntimeManifest` repository and revision.
- [x] Download URLs use direct Hugging Face `resolve` URLs: `https://huggingface.co/{repo}/resolve/{revision}/{filename}`.
- [x] Downloaded artifacts are written atomically into an Application Support `Model Artifact Store` keyed by model repository and pinned revision.
- [x] Downloads are anonymous HTTP requests; this issue does not add Hugging Face token, login, or Keychain support.
- [x] Validation is limited to successful HTTP response, expected filename, and nonzero byte count; size and checksum validation are deferred.
- [x] Progress and clear failure states are surfaced through the existing Experiment Session preparation path.
- [x] The UI shows `Artifact Preparation Progress` while **Prepare** is fetching artifacts, including completed artifact count, current filename, and per-file percent when byte counts are available.
- [x] After real artifacts are prepared, **Start** may still run the deterministic runtime path, but the UI must clearly label that runtime path as deterministic/not real translation.
- [x] Partial downloads are written to temporary files, deleted on failure, and moved into place only after successful validation.
- [x] A later **Prepare** skips already-completed artifacts and retries only missing files; byte-range resume is out of scope.
- [x] User cancellation during **Prepare** is out of scope for this slice.
- [x] When HTTP content length is available, download checks available disk space before writing a large artifact.
- [x] Memory preflight is out of scope; issue 12 does not load model weights into memory.
- [x] The app's default **Prepare** path uses the real cache-first provider.
- [x] Required runtime files are limited to `config.json`, `hibiki.q4.safetensors`, `mimi-pytorch-e351c8d8@125.safetensors`, and `tokenizer_spm_48k_multi6_2.model`.
- [x] **Prepare** downloads or verifies all four required runtime files before reporting ready.
- [x] Development-only files such as `mlx_hibiki_patch.py` and `verify_mlx_q4.py` are not required by the app at runtime.
- [x] Tests cover cache hit, successful fake download, partial download cleanup, HTTP/download failure, and manifest filename mismatch.

## Implementation notes

- Added `HuggingFaceModelArtifactProvider` as the app default provider for **Prepare**.
- Runtime artifacts are stored under Application Support `ModelArtifacts/{repo}/{revision}` with safe repo/revision path components.
- Downloads use direct anonymous Hugging Face `resolve` URLs and stream into hidden `.download` temp files before moving into place.
- The downloader checks HTTP status, reports byte progress when `Content-Length` is available, and preflights disk space for known content lengths.
- The Experiment Session now streams `Artifact Preparation Progress` events during preparation so the UI can show current file, overall progress, and per-file percent.
- Existing deterministic Mimi/Hibiki/decode/playback components remain in the **Start** path until the MLX-backed runtime issues land.

## Notes

- Keep the demo provider for unit tests or preview-only flows if useful, but the app should default to the real provider so **Prepare** reflects real artifact readiness.
- Use Application Support, not Caches or Documents, for durable model artifacts.
- Assume the model repository is anonymously accessible for this slice.
- Do not call the Hugging Face metadata API in this slice; the `ModelRuntimeManifest` is the source of required filenames.
- Download progress should be visible enough that a multi-GB first run does not look frozen.
- Extend the current generic preparation progress event if needed; first-run download should not be represented only by a spinner.
- If the app is killed during download, the next **Prepare** should clean abandoned temporary files before retrying.
- Structure validation code so future size or checksum checks can be added without changing the provider boundary.
- Prefer a readable storage failure before starting a large download when the device clearly lacks enough free space.
- Avoid loading the model in this issue; this slice only proves artifact availability and cache semantics.
- Do not replace Mimi or Hibiki runtime components in this issue. That begins in issue 13.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/02-confirm-model-artifact-contract.md`
- `.scratch/hibiki-ios-mlx/issues/03-load-and-cache-model-artifacts.md`

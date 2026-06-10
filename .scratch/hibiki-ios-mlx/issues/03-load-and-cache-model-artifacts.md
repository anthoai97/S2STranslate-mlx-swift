# Load and Cache Model Artifacts

Status: complete

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Implement the first model artifact loading path for the experiment app. The app should discover cached artifacts first, prepare or download missing artifacts from the confirmed Hugging Face source, report progress to the experiment session, and surface clear user-visible errors when artifacts are missing, incompatible, inaccessible, corrupt, or too large.

This slice should be verifiable without requiring a real network download by using a fake artifact provider in tests.

Implemented in this slice:

- `S2STranslate/ModelArtifactPreparation.swift` defines the manifest model, required artifact roles, provider protocol, cache-first preparer, prepared artifact result, explicit artifact failure categories, and `ModelArtifactExperimentBackend`.
- `DemoModelArtifactProvider` keeps the proof-of-concept app network-free while exercising the same preparation path.
- `ContentView` now uses `ModelArtifactExperimentBackend` for **Prepare**, so the UI shows preparation progress and failure details through the same `ExperimentSession` state as future real providers.
- `Tests/S2STranslateCoreTests/ModelArtifactPreparationTests.swift` covers manifest decoding, cache hit, first-run preparation, distinct failures, and session integration.
- This slice intentionally does not download real large model files yet; the provider boundary is ready for a real cache/download implementation.

## Acceptance criteria

- [x] The session can request model preparation and receive progress updates.
- [x] Cached artifacts are used before attempting first-run download or preparation.
- [x] Missing, inaccessible, corrupt, and incompatible artifacts produce distinct error states.
- [x] Tests use a fake artifact provider to cover cache hit, first-run preparation, and failure paths.
- [x] The app presents first-run preparation progress and failure details in the proof-of-concept UI.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/01-create-experiment-session-skeleton.md`
- `.scratch/hibiki-ios-mlx/issues/02-confirm-model-artifact-contract.md`

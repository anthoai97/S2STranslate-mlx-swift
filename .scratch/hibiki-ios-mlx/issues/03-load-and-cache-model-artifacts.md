# Load and Cache Model Artifacts

Status: ready-for-agent

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Implement the first model artifact loading path for the experiment app. The app should discover cached artifacts first, prepare or download missing artifacts from the confirmed Hugging Face source, report progress to the experiment session, and surface clear user-visible errors when artifacts are missing, incompatible, inaccessible, corrupt, or too large.

This slice should be verifiable without requiring a real network download by using a fake artifact provider in tests.

## Acceptance criteria

- [ ] The session can request model preparation and receive progress updates.
- [ ] Cached artifacts are used before attempting first-run download or preparation.
- [ ] Missing, inaccessible, corrupt, and incompatible artifacts produce distinct error states.
- [ ] Tests use a fake artifact provider to cover cache hit, first-run preparation, and failure paths.
- [ ] The app presents first-run preparation progress and failure details in the proof-of-concept UI.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/01-create-experiment-session-skeleton.md`
- `.scratch/hibiki-ios-mlx/issues/02-confirm-model-artifact-contract.md`

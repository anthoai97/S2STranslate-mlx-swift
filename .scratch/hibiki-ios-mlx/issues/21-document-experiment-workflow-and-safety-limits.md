# Document Experiment Workflow and Safety Limits

Status: ready-for-agent

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Document how a contributor should set up, run, test, and interpret the experimental iOS Hibiki/Mimi app. The docs should explain first-run model artifact preparation, local cache behavior, file-based and live demos, test workflow, known limitations, licenses, and safety boundaries.

The documentation should make it clear that this repo is an experimentation harness, not a production translation app.

## Acceptance criteria

- [ ] Setup instructions explain required tools, model artifact access, and first-run preparation expectations.
- [ ] The file-based demo workflow is documented.
- [ ] The live microphone demo workflow and device-only limitations are documented.
- [ ] The test workflow explains how to run automated tests and how reference traces are used.
- [ ] Known limitations include French-to-English scope, performance uncertainty, model access constraints, and non-production status.
- [ ] License and model usage notes are visible.
- [ ] Safety notes explicitly rule out malicious voice impersonation or deceptive voice cloning use cases.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/19-run-real-file-based-french-to-english-smoke-test.md`
- `.scratch/hibiki-ios-mlx/issues/20-wire-live-microphone-translation-demo.md`

# Add Reference Trace Harness

Status: ready-for-agent

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Create a small reference trace harness for comparing Swift-side behavior against the Python reference. The harness should support storing or importing compact traces that describe expected tensor shapes, token sequences where deterministic comparison is practical, event order, and frame cadence.

The goal is not to run the full model in tests. The goal is to give future Mimi and Hibiki slices a stable way to check whether the Swift implementation is still following the reference pipeline.

## Acceptance criteria

- [ ] A compact trace format is defined for model, codec, or session events.
- [ ] Tests can load a reference trace and compare it to Swift-generated events.
- [ ] The harness supports deterministic token or shape comparisons when available.
- [ ] The harness supports cadence or event-order comparisons for streaming behavior.
- [ ] Documentation explains how traces should be generated from the Python reference.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/02-confirm-model-artifact-contract.md`

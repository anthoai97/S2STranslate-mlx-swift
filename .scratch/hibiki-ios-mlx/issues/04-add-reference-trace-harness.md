# Add Reference Trace Harness

Status: complete

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Create a small reference trace harness for comparing Swift-side behavior against the Python reference. The harness should support storing or importing compact traces that describe expected tensor shapes, token sequences where deterministic comparison is practical, event order, and frame cadence.

The goal is not to run the full model in tests. The goal is to give future Mimi and Hibiki slices a stable way to check whether the Swift implementation is still following the reference pipeline.

Implemented in this slice:

- `S2STranslate/ReferenceTrace.swift` defines a compact Codable trace format for session, codec, model, audio, and text events.
- `ReferenceTraceComparator` compares event count/order, deterministic shapes, deterministic tokens, and frame/time cadence with tolerances.
- `Tests/S2STranslateCoreTests/Fixtures/reference-trace-small.json` is the first bundled fixture.
- `Tests/S2STranslateCoreTests/ReferenceTraceTests.swift` covers fixture loading, identical traces, token/shape mismatches, event-order mismatches, and cadence tolerances.
- `docs/reference-traces.md` documents the trace format and how to generate compact traces from `ref/hibiki-zero-mlx/src/infer_mlx_fast.py`.

## Acceptance criteria

- [x] A compact trace format is defined for model, codec, or session events.
- [x] Tests can load a reference trace and compare it to Swift-generated events.
- [x] The harness supports deterministic token or shape comparisons when available.
- [x] The harness supports cadence or event-order comparisons for streaming behavior.
- [x] Documentation explains how traces should be generated from the Python reference.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/02-confirm-model-artifact-contract.md`

# Reference Traces

Reference traces are compact JSON fixtures used to compare Swift pipeline behavior against the Python Hibiki-Zero reference without running the full model in unit tests.

## Format

Each trace records:

- `schemaVersion`: currently `1`.
- `name`: a stable trace name.
- `source`: the reference script, reference commit, model revision, and input fixture name.
- `events`: ordered streaming events.

Each event may include:

- `stream`: `session`, `codec`, `model`, `audio`, or `text`.
- `name`: a stable event name such as `mimiEncodeStep`, `hibikiStep`, or `mimiDecodeStep`.
- `frameIndex`: the 12.5 Hz streaming frame index when known.
- `shape`: tensor or array shape when deterministic.
- `tokens`: token IDs when deterministic enough to compare.
- `cadenceMilliseconds`: expected cadence for streaming checks.

## Generating From Python

Use `ref/hibiki-zero-mlx/src/infer_mlx_fast.py` as the first reference. Instrument a small fixture run to emit one JSON event per meaningful step:

1. Record a `codec:mimiEncodeStep` event after `mimi_enc.encode_step`.
2. Record a `model:hibikiStep` event after `gen.step`.
3. Record `text` events for emitted or skipped text tokens.
4. Record an `audio:mimiDecodeStep` event when `gen.last_audio_tokens()` is decoded.
5. Record post-input stop behavior with frame indexes and tokens `0` or `3` when testing Text Pad Stop.

Keep traces small. Prefer a few representative frames over full audio outputs. Large waveform parity belongs in later integration tests, not this harness.

## Swift Comparison

Swift tests load trace JSON with `ReferenceTrace.decode(from:)` and compare expected vs actual events using `ReferenceTraceComparator.compare`.

The deterministic Mimi encode boundary can project its first emitted `MimiTokenFrame` into a `codec:mimiEncodeStep` trace event. This is a structural parity check for frame index, shape, token prefix, and cadence; it is not proof of neural Mimi token quality until the MLX-backed encoder replaces the deterministic implementation.

The deterministic Mimi decode boundary can project its first emitted `DecodedAudioChunk` into an `audio:mimiDecodeStep` trace event. This checks decoded chunk shape and cadence through the same trace comparator; it is not proof of perceptual audio quality until the MLX-backed decoder replaces the deterministic implementation.

The deterministic Hibiki inference boundary can project its first emitted `HibikiInferenceStep` into a `model:hibikiStep` trace event and its text token into a `text:*` trace event. This checks source-token-to-model event order, output shape, token prefix, and blank/padding skip behavior; it is not proof of translation quality until the MLX-backed inference session replaces the deterministic implementation.

The comparator can check:

- event count and event order,
- deterministic shapes,
- deterministic token sequences,
- frame cadence and millisecond cadence with tolerances.

The first fixture lives at `Tests/S2STranslateCoreTests/Fixtures/reference-trace-small.json`.

# Validate Mimi Runtime Metadata and Warmup

Status: ready-for-agent

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Add validation and a minimal warmup path for the loaded MLX Mimi Runtime before it is used by streaming encode or decode.

This issue should prove that the runtime has coherent metadata and can execute a tiny encode/decode-shaped path without becoming the default app translation path.

## Acceptance criteria

- [ ] Runtime validation confirms sample rate `24000`, frame rate `12.5`, codebook count `16`, quantizer bins `2048`, and samples per frame `1920`.
- [ ] Warmup creates a small zero PCM MLX array and exercises the Mimi runtime enough to catch obvious graph/parameter failures.
- [ ] Warmup failures surface through explicit Mimi load/runtime errors.
- [ ] The runtime exposes reset behavior for encoder and decoder streaming state.
- [ ] Tests cover metadata validation, reset forwarding, and warmup failure through fake MLX seams where full execution is impractical.
- [ ] The deterministic test doubles remain available and unchanged.

## Notes

- Moshi's reference `warmup()` uses `1920 * 4` zero samples. Use that as guidance, but keep memory/device cost reasonable for iOS.
- This issue still does not wire the MLX Mimi Runtime into the UI default.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/15-load-mimi-safetensors-with-key-mapping.md`

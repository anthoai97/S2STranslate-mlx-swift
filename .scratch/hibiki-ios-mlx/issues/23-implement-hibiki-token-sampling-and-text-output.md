# Implement Hibiki Token Sampling and Text Output

Status: ready-for-agent

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Complete the real Hibiki generation loop around the MLX model step. The session should sample text and audio tokens with the configured defaults, decode visible English text pieces through the tokenizer, skip blank/padding tokens where appropriate, and support offline/file input flushing after the source audio ends.

## Acceptance criteria

- [ ] Generation uses temperature `0.8` and top-k `250` defaults for text and audio streams unless explicitly configured otherwise.
- [ ] Text token output is decoded into visible English text pieces when available.
- [ ] Blank/padding text tokens such as `0` and `3` are handled according to the confirmed contract.
- [ ] Offline/file input end forces source-audio EOS behavior and continues sampling until the Text Pad Stop condition is reached.
- [ ] Target audio-token frames continue to stream while text is emitted.
- [ ] Session observations report text token count, visible text count, generated audio frame count, and sampling summary.
- [ ] Tests cover sampling configuration, blank/padding skipping, text accumulation, and post-input stop using fake logits/tokenizer seams.
- [ ] Reference trace comparison records event order and token-shape parity against the Python reference where possible.

## Notes

- Keep voice transfer disabled until a real supported control is implemented.
- This issue should not implement audio playback; it should only emit generated audio token frames to the existing decode path.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/22-implement-mlx-hibiki-model-load-and-step.md`

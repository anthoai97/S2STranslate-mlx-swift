# Confirm Model Artifact Contract

Status: ready-for-human

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Confirm the model artifact contract for `anquachdev/hbk-zero-3b-mlx-q4` before agents implement loading or inference against guessed filenames and tensor layouts. The output should be a short repo-local contract note that records the exact Hugging Face source, required files, configuration format, tokenizer or text assets, Mimi assets, quantization assumptions, tensor naming conventions, and expected parity targets against the Python reference.

This is a human-in-the-loop issue because the Hugging Face artifact may be private, newly published, or unavailable to automated browsing.

## Acceptance criteria

- [ ] The canonical model source is confirmed as `https://huggingface.co/anquachdev/hbk-zero-3b-mlx-q4`.
- [ ] Required artifact filenames and their roles are documented.
- [ ] Tokenizer, text, Mimi, and Hibiki configuration assets are identified or explicitly marked absent.
- [ ] Quantization format and MLX Swift loading expectations are documented.
- [ ] A minimal parity target is defined for the first implementation, such as shape checks, event order, token traces, or a small known input fixture.
- [ ] Any access requirements for private or gated model files are documented.

## Blocked by

None - can start immediately

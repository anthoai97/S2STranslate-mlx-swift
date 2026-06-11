# Run Real File-Based French-to-English Smoke Test

Status: in-progress

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Wire the real artifact provider, MLX Mimi encoder, MLX Hibiki inference session, MLX Mimi decoder, and playback sink into the file-based demo path. The app should run a French sample from the issue 9 catalog and stream real English text and generated voice output.

This is the first end-to-end real translation smoke test.

## Acceptance criteria

- [ ] The app can prepare real model artifacts and initialize real Mimi/Hibiki components.
- [ ] A selected French file input streams through real Mimi encode, real Hibiki generation, real Mimi decode, and playback.
- [ ] English text appears incrementally as it is produced.
- [ ] English voice output is routed to playback or a clearly inspectable sink.
- [ ] Metrics distinguish file decode, Mimi encode, Hibiki step/sampling, Mimi decode, and playback delivery.
- [ ] Failure states from artifact download, model load, encode, inference, decode, and playback remain user-visible.
- [x] A manual smoke-test document records device/simulator expectations, first-run model cache cost, and known limitations.
- [x] Automated tests exercise the orchestration path with fake real-component seams where full 3B inference is impractical.

## Notes

- Prefer `French Europarl short 1` as the first smoke input.
- This issue is not done if it only emits deterministic placeholder text or buffered silent audio.
- Current checkpoints unblock playback with `AVAudioPlaybackSink`, add the real decode boundary for zero/one/many decoded chunks, wire the executable MLX Mimi decode graph, add the MLX Hibiki session/load boundary with a fake model-step seam, add grouped-query/`rope_concat` transformer support plus Hibiki LM/Depformer topology parsing, add a real-file backend that constructs MLX Mimi encode/decode components after artifact preparation, add a tokenizer-decoding seam plus opt-in tail-silence flushing with blank/pad stop detection for file input, and record the manual smoke checklist in `docs/real-file-french-english-smoke-test.md`.
- Remaining blocker is the real Hibiki model step/token sampling graph before this can be marked done.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/12-implement-real-huggingface-artifact-download-cache.md`
- `.scratch/hibiki-ios-mlx/issues/17-implement-mlx-mimi-streaming-encode.md`
- `.scratch/hibiki-ios-mlx/issues/23-implement-hibiki-token-sampling-and-text-output.md`
- `.scratch/hibiki-ios-mlx/issues/24-add-av-audio-playback-sink.md`

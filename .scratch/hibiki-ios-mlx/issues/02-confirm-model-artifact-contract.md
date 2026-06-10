# Confirm Model Artifact Contract

Status: complete

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Confirm the model artifact contract for `anquachdev/hbk-zero-3b-mlx-q4` before agents implement loading or inference against guessed filenames and tensor layouts. The output should be a short repo-local contract note that records the exact Hugging Face source, required files, configuration format, tokenizer or text assets, Mimi assets, quantization assumptions, tensor naming conventions, and expected parity targets against the Python reference.

This is a human-in-the-loop issue because the Hugging Face artifact may be private, newly published, or unavailable to automated browsing.

Confirmed so far:

- Canonical source: `https://huggingface.co/anquachdev/hbk-zero-3b-mlx-q4`
- Confirmed Model Revision: `558daadd9272df9432642783b57b02756ff34d5b`
- Access: public, ungated
- License: MIT
- Confirmed files at that revision: `config.json`, `hibiki.q4.safetensors`, `mimi-pytorch-e351c8d8@125.safetensors`, `tokenizer_spm_48k_multi6_2.model`, `mlx_hibiki_patch.py`, `verify_mlx_q4.py`, `README.md`, `.gitattributes`
- `config.json` agrees with the file list: `moshi_name` is `hibiki.q4.safetensors`, `mimi_name` is `mimi-pytorch-e351c8d8@125.safetensors`, and `tokenizer_name` is `tokenizer_spm_48k_multi6_2.model`
- `config.json` is the Model Architecture Config. A future repo-local Model Runtime Manifest should be the app's control point for repository ID, pinned revision, required files, optional helper scripts, generation defaults, and loading policy.
- Initial Model Runtime Manifest: `S2STranslate/ModelRuntimeManifest.json`
- Generation defaults in the manifest are copied from `config.json.lm_gen_config`: temperature `0.8`, text temperature `0.8`, top-k `250`, text top-k `250`
- Runtime-required files are `config.json`, `hibiki.q4.safetensors`, `mimi-pytorch-e351c8d8@125.safetensors`, and `tokenizer_spm_48k_multi6_2.model`
- Quantization Contract: `hibiki.q4.safetensors` is a pre-quantized q4 MLX checkpoint and must be loaded with group size 32 semantics. Do not substitute group size 64 or a stock loader path that assumes a different q4 layout.
- Development-only helper files are `mlx_hibiki_patch.py` and `verify_mlx_q4.py`; the iOS app must not require Python files at runtime. `verify_mlx_q4.py` is not a parity authority for this project, only an optional debug/provenance helper.
- `mlx_hibiki_patch.py` is Implementation Provenance, not a runtime dependency. Swift implementation must account for the Hibiki-Zero MLX deltas it documents: `hidden_scale=6`, `kv_repeat=2`, `rope_concat`, and learned per-slice Depformer output LayerNorm (`depformer_norms.{i}`). The Depformer LayerNorm is required for audio quality; omitting it can leave text plausible while making generated audio babble or clip.
- The Model Runtime Manifest records those deltas in `architectureDeltas` so future Swift config and inference work does not accidentally treat the artifact as stock Moshi.
- Streaming Translation behavior: encode source audio with the streaming Mimi codec, feed source audio tokens to Hibiki-Zero incrementally, and decode generated output audio tokens incrementally.
- Input End Flush policy: offline/file paths should mark input end with source-audio EOS and continue sampling until the configured post-input Text Pad Stop condition is reached.
- Live Websocket Translation policy: the frontend path in `ref/hibiki-zero-mlx/frontend` sends browser Opus mic packets to `/api/chat`; the server decodes Opus to PCM, Mimi-encodes frames, steps Hibiki-Zero, and sends generated text/audio packets back. That path currently stops when the websocket closes and does not prove an explicit post-input flush contract.
- Text Pad Stop policy: follow the `infer_mlx_fast.py` observed stop rule until a true text EOS token is confirmed. Treat generated text tokens `0` and `3` as blank/padding after input end, and stop after 12 consecutive blank/padding frames. Token `3` matches `existing_text_padding_id` in `config.json`; token `0` is skipped by the Python reference when assembling text.
- Sampling policy: use temperature `0.8` and top-k `250` for all token streams.
- Python reference: `ref/hibiki-zero-mlx/src/infer_mlx_fast.py` from local reference clone commit `31a0f2b151f016e9347a9b5abd1d67f28c43448f`.
- Frontend/server reference: `ref/hibiki-zero-mlx/frontend/app/page.tsx`, `ref/hibiki-zero-mlx/frontend/app/useAudioProcessor.ts`, and `ref/hibiki-zero-mlx/hibiki_zero/inference.py`.
- First parity target: structural/token trace parity against the local Python reference, not perceptual audio parity. Use a small French audio fixture with the pinned model revision and record Mimi input token frame count, generated text token sequence, decoded text handling for tokens `0` and `3`, Text Pad Stop frame count, output event order, and tensor shapes. Audio quality parity belongs after the native Swift runtime is loading and stepping correctly.
- Swift implementation reference: `ref/moshi-swift/MoshiLib/`. Key reference files are `Mimi.swift` for streaming codec behavior, `LM.swift` for `LmConfig`, `LM`, `Depformer`, and `LMGen`, `Transformer.swift` for transformer configuration, `Streaming.swift` for streaming array mechanics, and `Quantization.swift` for Mimi quantizer behavior.

## Acceptance criteria

- [x] The canonical model source is confirmed as `https://huggingface.co/anquachdev/hbk-zero-3b-mlx-q4`.
- [x] Required artifact filenames and their roles are documented.
- [x] Tokenizer, text, Mimi, and Hibiki configuration assets are identified or explicitly marked absent.
- [x] Quantization format and MLX Swift loading expectations are documented: q4 MLX safetensors, group size 32.
- [x] A minimal parity target is defined for the first implementation: structural/token trace parity against a small Python-reference fixture before perceptual audio parity.
- [x] Any access requirements for private or gated model files are documented.
- [x] A repo-local Model Runtime Manifest exists for issue 3 to consume.

## Blocked by

None - can start immediately

# Wire File-Based Translation Demo

Status: complete

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Wire the first complete file-based translation demo through the app. A user should be able to prepare the model, choose or use a fixture French audio input, run the streaming pipeline, see English text events when available, hear or inspect generated audio output, and view basic metrics.

This is the first full vertical slice across artifact loading, audio input, Mimi encode, Hibiki generation, Mimi decode, output, UI state, and tests.

## Fixture candidates

Use French-to-English samples from `kyutai/hibiki-zero-samples` as the first bundled or downloadable file input candidates:

- Short-form Europarl-ST source:
  - `https://huggingface.co/spaces/kyutai/hibiki-zero-samples/resolve/main/data/europarl_st/fr/source/30ef344ae8687926.mp3`
  - `https://huggingface.co/spaces/kyutai/hibiki-zero-samples/resolve/main/data/europarl_st/fr/source/4539f03d07ce7fbf.mp3`
- Short-form Hibiki-Zero reference output:
  - `https://huggingface.co/spaces/kyutai/hibiki-zero-samples/resolve/main/data/europarl_st/fr/hibiki-zero/30ef344ae8687926.mp3`
  - `https://huggingface.co/spaces/kyutai/hibiki-zero-samples/resolve/main/data/europarl_st/fr/hibiki-zero/4539f03d07ce7fbf.mp3`
- Long-form Audio-NTREX-4L source:
  - `https://huggingface.co/spaces/kyutai/hibiki-zero-samples/resolve/main/data/audio_ntrex_4L/fr/source/ee67adf3f3768b1d_11labs.mp3`
  - `https://huggingface.co/spaces/kyutai/hibiki-zero-samples/resolve/main/data/audio_ntrex_4L/fr/source/f9fcfb48c566cfad_11labs.mp3`
- Long-form Hibiki-Zero reference output:
  - `https://huggingface.co/spaces/kyutai/hibiki-zero-samples/resolve/main/data/audio_ntrex_4L/fr/hibiki-zero/ee67adf3f3768b1d_11labs.mp3`
  - `https://huggingface.co/spaces/kyutai/hibiki-zero-samples/resolve/main/data/audio_ntrex_4L/fr/hibiki-zero/f9fcfb48c566cfad_11labs.mp3`

Prefer a short-form source first for simulator smoke testing, then add the long-form source once decode/resample and progress reporting are stable.

## Acceptance criteria

- [x] The app can run a translation session from a downloadable French file input.
- [x] The session streams through artifact loading, audio input, encode, inference, decode, and output without requiring microphone input.
- [x] Text output is displayed when emitted by the deterministic Hibiki boundary.
- [x] Generated audio output is routed to a buffered playback sink.
- [x] Basic frame cadence and resource-adjacent metrics are visible through session observations.
- [x] Tests cover the successful file-based flow using generated audio fixtures and deterministic model components.
- [x] Failure states from each major stage are surfaced in the UI through the existing Experiment Session failure path.

## Implementation notes

- Added `FileAudioInputSource` for decoding local audio files into mono 24 kHz PCM chunks.
- Added `RemoteAudioFileInputSource` for downloading selected Hugging Face MP3 fixtures into cache before decoding.
- Added `ConfigurableAudioInputSource` so the SwiftUI demo can switch French fixture inputs without rebuilding the session backend.
- The app still uses deterministic Mimi/Hibiki/decode/playback boundaries. This validates file input wiring, not real MLX translation quality.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/03-load-and-cache-model-artifacts.md`
- `.scratch/hibiki-ios-mlx/issues/05-build-streaming-audio-input-slice.md`
- `.scratch/hibiki-ios-mlx/issues/06-implement-mimi-streaming-encode-path.md`
- `.scratch/hibiki-ios-mlx/issues/07-implement-mimi-streaming-decode-and-playback-path.md`
- `.scratch/hibiki-ios-mlx/issues/08-implement-minimal-hibiki-inference-session.md`

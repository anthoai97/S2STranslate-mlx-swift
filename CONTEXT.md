# S2STranslate

S2STranslate is an experimental iOS context for running and inspecting streaming speech translation models on device. Its language centers on experiment lifecycles, streaming audio, model artifacts, and observable translation outputs.

## Language

**Experiment Session**:
One attempt to prepare and run the speech translation pipeline from an input source to observable outputs. It owns the user-visible lifecycle of the attempt.
_Avoid_: UI session, model session, audio session

**Unloaded**:
An Experiment Session state where no preparation work has started.
_Avoid_: Initial, empty

**Preparing**:
An Experiment Session state where work needed before running is in progress. Preparation may include artifact checks, runtime initialization, permission checks, fixture setup, or other prerequisites.
_Avoid_: Loading

**Stopped**:
A terminal Experiment Session state reached when the user intentionally ends a started attempt and active work has been released. Starting again creates a fresh attempt rather than resuming the stopped one.
_Avoid_: Paused, suspended, resumable

**Failed**:
A terminal Experiment Session state reached when the attempt cannot continue because an error occurred. Retrying starts a fresh attempt rather than resuming the failed one.
_Avoid_: Recovering, retrying, degraded

**New Session**:
A user intent that creates a fresh Experiment Session after a terminal state. It is not a retry or resume of the previous attempt.
_Avoid_: Retry, resume, restart

**Ready**:
An Experiment Session state where preparation is complete and the session can start running, but no active input, inference, or output work is happening.
_Avoid_: Loaded, idle

**Running**:
An Experiment Session state where active experiment work has started and the session is receiving progress events from the backend. Running does not by itself mean translation output has been produced.
_Avoid_: Translating, successful

**Audio Input Source**:
A component that emits timestamped PCM chunks into an Experiment Session. It may be fixture-backed, file-backed, or microphone-backed, but automated tests should prefer deterministic fixture or file sources.
_Avoid_: Recorder, microphone session

**File Audio Input Source**:
An Audio Input Source that decodes a local or cached audio file into mono PCM Chunks at the pipeline sample rate. It is the preferred reproducible UI input before live microphone support.
_Avoid_: Batch translation, uploaded file

**Remote Audio Fixture**:
A known sample audio file downloaded into the app cache before being decoded by a File Audio Input Source. The first fixtures are French-to-English examples from Kyutai's Hibiki-Zero sample Space.
_Avoid_: Model artifact, bundled asset

**PCM Chunk**:
A bounded slice of normalized audio samples with a sample rate, frame index, and timestamp. It is the input boundary before Mimi encoding.
_Avoid_: Audio token, waveform file

**Mimi Streaming Encoder**:
The stateful encode component that accepts PCM Chunks and emits Mimi Token Frames while preserving buffered samples and frame index across chunk boundaries. The current implementation is deterministic and protocol-backed; the future MLX-backed Mimi encoder should conform to the same boundary.
_Avoid_: Batch codec, stateless encoder

**Mimi Token Frame**:
One encoded Mimi timestep containing audio codebook tokens for a single frame cadence. It is the codec output boundary before Hibiki consumes source audio tokens.
_Avoid_: PCM frame, decoded audio frame

**Mimi Streaming Decoder**:
The stateful decode component that accepts Mimi Token Frames and emits Decoded Audio Chunks while preserving output frame index across token boundaries. The current implementation is deterministic and protocol-backed; the future MLX-backed Mimi decoder should conform to the same boundary.
_Avoid_: Batch decoder, audio player

**Decoded Audio Chunk**:
A bounded slice of PCM samples produced by Mimi decoding, with sample rate, frame index, timestamp, and source token frame metadata. It is the output boundary before playback.
_Avoid_: Mimi token, source PCM chunk

**Playback Sink**:
A playback-oriented output stream that accepts Decoded Audio Chunks. Automated tests use buffered sinks instead of device hardware; a future device sink can route chunks to actual audio playback behind the same boundary.
_Avoid_: Decoder, speaker

**Hibiki Inference Session**:
The stateful generation component that accepts source Mimi Token Frames and emits target text tokens plus generated target audio token frames. The current implementation is deterministic and protocol-backed; the future MLX-backed Hibiki session should conform to the same boundary.
_Avoid_: Codec, translator UI

**Generated Audio Token Frame**:
A Mimi Token Frame produced by Hibiki as target speech audio tokens. It is suitable for the Mimi Streaming Decoder and is distinct from source audio tokens emitted by the Mimi Streaming Encoder.
_Avoid_: Source token frame, decoded audio chunk

**Hibiki Generation Configuration**:
The explicit sampling and generation controls for a Hibiki Inference Session. The starting defaults are temperature `0.8` and top-k `250` for both text and audio streams, with voice transfer disabled until a real supported control exists.
_Avoid_: Hidden sampling defaults, runtime manifest

**Language Direction**:
The source-to-target language pair for an Experiment Session. French-to-English is the starting Language Direction, but the Experiment Session skeleton should not assume a specific pair.
_Avoid_: Language support, locale, localization

**Streaming Translation**:
A translation attempt where source audio is encoded into audio tokens incrementally, the model samples target tokens as the input arrives, and output audio is decoded incrementally.
_Avoid_: Batch translation, offline translation

**Input End Flush**:
The offline/file-mode behavior that marks the source audio end and continues sampling until the configured post-input stop condition is reached.
_Avoid_: Live disconnect, stop button

**Live Websocket Translation**:
The frontend/server mode where browser Opus mic packets stream over `/api/chat`, the server emits text and audio packets as they are produced, and the session currently ends when the websocket closes.
_Avoid_: Offline generation, file translation

**Text Pad Stop**:
The post-input stopping rule for Hibiki-Zero streaming translation. After the input audio stream has ended, sampling continues until the generated text stream emits sustained blank or padding tokens long enough to consider the translation flushed.
_Avoid_: Text Stream EOS, input finished, stop button

**Model Artifact Contract**:
A versioned record of the exact model repository revision, required files, file roles, loading assumptions, and minimum parity checks that implementation work must follow before loading or running a model.
_Avoid_: Model docs, file list, download notes

**Model Revision**:
The immutable source revision of a model repository that a Model Artifact Contract was confirmed against.
_Avoid_: Latest, main, current model

**Model Architecture Config**:
The model-provided configuration that describes architecture and generation defaults needed to construct the model correctly.
_Avoid_: App config, runtime manifest

**Model Runtime Manifest**:
A repo-local, machine-readable control file that tells the app which model repository, revision, files, defaults, and loading policy to use.
_Avoid_: Model architecture config, Hugging Face config

**Implementation Provenance**:
Reference material that explains how a model artifact was produced or patched, used to guide implementation but not loaded by the app at runtime.
_Avoid_: Runtime dependency, required artifact

**Quantization Contract**:
The model-loading requirement that the Hibiki weights are pre-quantized q4 MLX safetensors using group size 32. Loaders must preserve that format rather than trying alternate group sizes.
_Avoid_: Quantized model, smaller weights

**Depformer**:
The audio-token submodel that predicts dependent audio codebooks after the main transformer step in Hibiki/Moshi-style models.
_Avoid_: Decoder, Mimi

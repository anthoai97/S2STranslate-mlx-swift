# Real File French-to-English Smoke Test

## Scope

Use this checklist for issue 25 device/simulator smoke runs. The target flow is:

`French file input -> real MLX Mimi encode -> real MLX Hibiki generation -> real MLX Mimi decode -> playback`

## Current Expectations

- First-run preparation downloads/caches the pinned `hibiki.q4.safetensors`, Mimi safetensors, tokenizer, and `config.json` artifacts. Expect multi-GB disk use and a long first prepare step.
- Prefer `French Europarl short 1` for the first run.
- iOS Simulator runs are limited to Sample playback and UI diagnostics. Real MLX translation is disabled there because the current MLX Metal backend aborts during GPU initialization before Swift can surface a recoverable error.
- Device runs use `AVAudioPlaybackSink` and should route decoded PCM progressively through AVFoundation.
- The app demo path now constructs real MLX Mimi encode/decode components and a real MLX Hibiki session after artifacts prepare. It also appends a bounded tail-silence flush so delayed file-output text/audio can drain after source input ends.

## Known Limitations

- Real Mimi encode/decode and real Hibiki stepping are wired into the file demo path, with automated seam coverage for orchestration and opt-in artifact coverage for the local Mimi codec fixture.
- Full 3B q4 French-to-English translation quality and audible generated voice remain unverified until a device smoke run records real output.
- If real decode emits no chunks, playback must stay silent; do not inject placeholder silence.

## Smoke Checklist

1. Launch the app on a device with enough free disk for the model cache.
2. Select `French Europarl short 1`.
3. Tap **Prepare** and wait for all required artifacts to complete.
4. Tap **Start**.
5. Confirm observations distinguish file decode, Mimi encode, Hibiki inference, Mimi decode, and playback events.
6. Confirm visible English text appears incrementally once real Hibiki text output is enabled.
7. Confirm generated English voice is audible on device playback.
8. Confirm failures from artifact preparation, model load, encode, inference, decode, and playback are surfaced in the session state.

## Automated Smoke Command

Run the guarded smoke test only on a machine with the full local artifacts and enough memory for the 3B q4 graph:

```sh
S2S_RUN_REAL_FILE_SMOKE_TESTS=1 swift test --filter RealFileFrenchEnglishSmoke
```

By default it reads artifacts from `ref/hibiki-zero-mlx/weights` and downloads `French Europarl short 1` through the existing fixture cache. Override these paths when needed:

```sh
S2S_RUN_REAL_FILE_SMOKE_TESTS=1 \
S2S_REAL_FILE_SMOKE_WEIGHTS_DIR=/path/to/weights \
S2S_REAL_FILE_SMOKE_AUDIO_PATH=/path/to/french-europarl-short-1.mp3 \
swift test --filter RealFileFrenchEnglishSmoke
```

The test asserts real file decode, Mimi encode, Hibiki stepping/text, Mimi decode, and buffered playback counters are all nonzero.

When enabled, the test also writes inspectable artifacts:

- `.scratch/real-file-smoke/latest/translation.txt`
- `.scratch/real-file-smoke/latest/translation.wav`

Override the output directory with:

```sh
S2S_REAL_FILE_SMOKE_OUTPUT_DIR=/path/to/output
```

## Automated Benchmark Command

Run the guarded benchmark when you want to measure current model-flow production speed without AVAudio playback:

```sh
S2S_RUN_REAL_FILE_BENCHMARKS=1 swift test --filter RealFileFrenchEnglishSmokeTests
```

It uses the same defaults as the smoke test:

- weights: `ref/hibiki-zero-mlx/weights`
- audio: `French Europarl short 1`
- generation config: app real-file defaults, including tail flush

For a quick speed-only probe, cap the source and disable tail flush. This is useful for timing the model loop, but it may leave `translation.txt` empty because visible text can arrive after the first few seconds of source audio:

```sh
S2S_RUN_REAL_FILE_BENCHMARKS=1 \
S2S_REAL_FILE_SMOKE_WEIGHTS_DIR=/path/to/weights \
S2S_REAL_FILE_SMOKE_AUDIO_PATH=/path/to/french-europarl-short-1.mp3 \
S2S_REAL_FILE_BENCHMARK_OUTPUT_DIR=/path/to/output \
S2S_REAL_FILE_BENCHMARK_MAX_SOURCE_CHUNKS=20 \
S2S_REAL_FILE_BENCHMARK_INCLUDE_TAIL=0 \
swift test --filter RealFileFrenchEnglishSmokeTests
```

For a short text-output probe, keep tail flush enabled and use more source chunks:

```sh
S2S_RUN_REAL_FILE_BENCHMARKS=1 \
S2S_REAL_FILE_BENCHMARK_OUTPUT_DIR=.scratch/real-file-benchmark/text-check \
S2S_REAL_FILE_BENCHMARK_MAX_SOURCE_CHUNKS=40 \
swift test --filter RealFileFrenchEnglishSmokeTests
```

The benchmark writes:

- `.scratch/real-file-benchmark/latest/benchmark.json`
- `.scratch/real-file-benchmark/latest/benchmark.md`
- `.scratch/real-file-benchmark/latest/translation.txt`
- `.scratch/real-file-benchmark/latest/translation.wav`

Read `generatedRealtimeFactor` first. Values below `1.0` mean the model flow produces decoded audio slower than playback consumes it. The stage table then shows whether Mimi encode, Hibiki step, or Mimi decode dominates latency.

## Latest Verified Run

- 2026-06-11: `S2S_RUN_REAL_FILE_SMOKE_TESTS=1 swift test --filter RealFileFrenchEnglishSmoke` passed in 41.663 seconds with local `ref/hibiki-zero-mlx/weights`, `French Europarl short 1`, real MLX Mimi encode/decode, real MLX Hibiki generation, and `BufferedPlaybackSink`.

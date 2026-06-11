# Real File French-to-English Smoke Test

## Scope

Use this checklist for issue 25 device/simulator smoke runs. The target flow is:

`French file input -> real MLX Mimi encode -> real MLX Hibiki generation -> real MLX Mimi decode -> playback`

## Current Expectations

- First-run preparation downloads/caches the pinned `hibiki.q4.safetensors`, Mimi safetensors, tokenizer, and `config.json` artifacts. Expect multi-GB disk use and a long first prepare step.
- Prefer `French Europarl short 1` for the first run.
- Simulator runs may use the inspectable buffered sink when device audio is not available.
- Device runs use `AVAudioPlaybackSink` and should route decoded PCM progressively through AVFoundation.
- The app demo path now constructs real MLX Mimi encode/decode components and a real MLX Hibiki session after artifacts prepare. It also appends a bounded tail-silence flush so delayed file-output text/audio can drain after source input ends.

## Known Limitations

- Real Mimi encode/decode and real Hibiki stepping are wired into the file demo path, with automated seam coverage for orchestration and opt-in artifact coverage for the local Mimi codec fixture.
- Full 3B q4 French-to-English translation quality and audible generated voice remain unverified until a device/simulator smoke run records real output.
- If real decode emits no chunks, playback must stay silent; do not inject placeholder silence.

## Smoke Checklist

1. Launch the app on a device or simulator with enough free disk for the model cache.
2. Select `French Europarl short 1`.
3. Tap **Prepare** and wait for all required artifacts to complete.
4. Tap **Start**.
5. Confirm observations distinguish file decode, Mimi encode, Hibiki inference, Mimi decode, and playback events.
6. Confirm visible English text appears incrementally once real Hibiki text output is enabled.
7. Confirm generated English voice is audible on device playback, or inspect decoded/playback chunk metrics when using a buffered sink.
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

## Latest Verified Run

- 2026-06-11: `S2S_RUN_REAL_FILE_SMOKE_TESTS=1 swift test --filter RealFileFrenchEnglishSmoke` passed in 42.794 seconds with local `ref/hibiki-zero-mlx/weights`, `French Europarl short 1`, real MLX Mimi encode/decode, real MLX Hibiki generation, and `BufferedPlaybackSink`.

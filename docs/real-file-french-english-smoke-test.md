# Real File French-to-English Smoke Test

## Scope

Use this checklist for issue 25 device/simulator smoke runs. The target flow is:

`French file input -> real MLX Mimi encode -> real MLX Hibiki generation -> real MLX Mimi decode -> playback`

## Current Expectations

- First-run preparation downloads/caches the pinned `hibiki.q4.safetensors`, Mimi safetensors, tokenizer, and `config.json` artifacts. Expect multi-GB disk use and a long first prepare step.
- Prefer `French Europarl short 1` for the first run.
- Simulator runs may use the inspectable buffered sink when device audio is not available.
- Device runs use `AVAudioPlaybackSink` and should route decoded PCM progressively through AVFoundation.

## Known Limitations

- Real Mimi encode is available and covered by an opt-in local artifact fixture.
- The public Mimi decode boundary now supports zero, one, or many decoded chunks per token frame, and the AV playback sink can schedule decoded chunks progressively.
- Full real Mimi decode graph, real Hibiki model stepping, token sampling/text output, and bit-for-bit end-to-end translation are still separate blockers before issue 25 can be marked done.
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

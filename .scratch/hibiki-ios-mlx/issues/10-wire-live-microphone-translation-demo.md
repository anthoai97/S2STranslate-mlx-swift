# Wire Live Microphone Translation Demo

Status: ready-for-agent

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Wire live microphone input through the same translation path proven by the file-based demo. The user should be able to start a live French speech session, stream input through the model, receive English text and generated audio output when available, inspect metrics, and stop cleanly.

This slice should keep automated coverage focused on simulated sources while documenting or supporting a device smoke test for real microphone and playback behavior.

## Acceptance criteria

- [ ] The app can start a live microphone-backed translation session on a capable device.
- [ ] Live audio uses the same downstream streaming path as the file-based demo.
- [ ] Stop behavior cleanly releases input, inference, and output resources.
- [ ] User-visible metrics update during the live session.
- [ ] Automated tests cover the live path with simulated sources and sinks.
- [ ] Device smoke-test guidance covers microphone permission, capture, playback, and expected limitations.

## Blocked by

- `.scratch/hibiki-ios-mlx/issues/09-wire-file-based-translation-demo.md`

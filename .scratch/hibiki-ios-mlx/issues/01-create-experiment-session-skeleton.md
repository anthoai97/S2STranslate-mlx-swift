# Create Experiment Session Skeleton

Status: ready-for-agent

Implementation status: complete

## Parent

`.scratch/hibiki-ios-mlx/PRD.md`

## What to build

Create the first end-to-end experiment slice: the app can present an experiment session, drive it with a fake backend, show user-visible session status, and validate the state transitions with automated tests. This does not need real MLX inference yet; it establishes the shape that model loading, audio input, translation, playback, metrics, and errors will plug into.

The slice should be demoable as a proof-of-concept app flow: unloaded, preparing, ready, running, stopped, and failed states are visible and reachable through controlled fake behavior.

For this slice, the fake backend should fake the whole future pipeline behind one boundary. It should emit scripted experiment events such as preparation progress, ready, running metric ticks, stopped, and failed. It should not create fake Mimi, Hibiki, audio input, or playback subsystems yet.

Experiment Session state should live outside the SwiftUI view. The view should render session state and send user intents such as prepare, start, stop, trigger failure-demo, and new session; it should not own model, audio, translation, playback, or metrics lifecycle state directly.

The fake backend should be deterministic at its core: tests should be able to provide an exact script of experiment events and assert exact state transitions. A UI demo adapter may add small delays so the proof-of-concept screen visibly changes, but timer behavior should not be required for core tests.

Metrics in this slice should be placeholder experiment observations only, such as elapsed time, event count, last event name, or progress fraction. Do not present fake latency, memory, frame cadence, token count, or audio chunk metrics until the relevant subsystems exist.

## Completion notes

This slice was implemented with a Moshi-inspired UI shape from the local `ref/moshi-swift/Moshi/ModelView.swift` reference: an information panel before activity, an output panel once the fake run emits text, a status strip during activity, a centered primary action button, and a settings popover. The UI remains generic and fake-backed; it does not name a specific Language Direction or model target yet.

The fake backend emits deterministic scripts for preparation and running. Run observations append placeholder output text so the running screen can be exercised without introducing fake Mimi, Hibiki, microphone, playback, or model-loading subsystems.

## Acceptance criteria

- [x] The app exposes a visible experiment session state instead of the starter "Hello, world!" screen.
- [x] A fake backend can drive preparing, running, stopping, and failure states without real model weights.
- [x] The fake backend represents the whole future pipeline behind one boundary rather than separate fake model/audio/codec subsystems.
- [x] Tests can drive the fake backend with a deterministic event script without relying on timers.
- [x] Any displayed metrics are clearly placeholder observations and do not imply real model, audio, codec, latency, or memory measurement.
- [x] Session state is separated from the SwiftUI view so later model/audio work can attach without UI rewrites.
- [x] The SwiftUI view observes Experiment Session state and sends user intents rather than owning experiment lifecycle state directly.
- [x] Automated tests cover the main state transitions and at least one failure path.
- [x] The UI clearly communicates that real model inference is not wired in this slice.
- [x] The repo has an automated test target or equivalent automated Swift test setup for the Experiment Session lifecycle.
- [x] Tests cover the Experiment Session state machine, deterministic fake backend scripts, and one failure path without asserting SwiftUI pixel/layout details.
- [x] After stopped or failed terminal states, the UI provides a New Session intent that creates a fresh attempt rather than retrying or resuming the previous one.

## Blocked by

None - can start immediately

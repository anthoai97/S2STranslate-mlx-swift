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

**Language Direction**:
The source-to-target language pair for an Experiment Session. French-to-English is the starting Language Direction, but the Experiment Session skeleton should not assume a specific pair.
_Avoid_: Language support, locale, localization

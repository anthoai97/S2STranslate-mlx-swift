# Microphone Smoke Test

This project does not wire live microphone capture into the app yet. Issue 5 uses a simulator-friendly `FixtureAudioInputSource` so automated tests can prove the experiment session can start, consume timestamped PCM chunks, stop cleanly, and report stream metrics without physical hardware.

Use this checklist when the real `AVAudioEngine` source is added:

1. Run the app on a physical iOS device.
2. Grant microphone permission when prompted.
3. Tap **Prepare** and wait for the session to become ready.
4. Tap **Start** and speak for several seconds.
5. Confirm the observation panel shows `streaming`, a 24 kHz sample rate after resampling, increasing chunk count, and increasing duration.
6. Tap **Stop** and confirm the session reaches `Stopped` without extra chunks after stop.
7. Deny microphone permission in Settings and confirm the UI surfaces an audio input failure instead of silently hanging.

The automated test suite should keep using fixture or file-backed sources. Physical-device microphone behavior belongs in manual smoke testing or later UI/integration tests.

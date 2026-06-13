import Testing

@testable import S2STranslateCore

@Suite("Experiment Session")
struct ExperimentSessionTests {
    @Test("prepare moves an unloaded Experiment Session through preparing to ready")
    @MainActor
    func prepareMovesSessionToReady() async {
        let backend = ScriptedExperimentBackend(events: [
            .preparationProgress(0.25),
            .preparationProgress(1.0),
            .ready,
        ])
        let session = ExperimentSession(backend: backend)

        await session.prepare()

        #expect(session.state == .ready)
        #expect(session.observations.progress == 1.0)
        #expect(session.observations.eventCount == 3)
        #expect(session.observations.lastEventName == "ready")
    }

    @Test("start moves a ready Experiment Session to running and records placeholder observations")
    @MainActor
    func startMovesReadySessionToRunning() async {
        let backend = ScriptedExperimentBackend(
            prepareEvents: [.ready],
            runEvents: [
                .observation("demo tick"),
                .observation("second tick"),
            ]
        )
        let session = ExperimentSession(backend: backend)

        await session.prepare()
        await session.start()

        #expect(session.state == .running)
        #expect(session.observations.eventCount == 3)
        #expect(session.observations.lastEventName == "second tick")
        #expect(session.observations.output == "demo tick\nsecond tick")
    }

    @Test("stop from running reaches terminal stopped")
    @MainActor
    func stopFromRunningReachesStopped() async {
        let backend = ScriptedExperimentBackend(
            prepareEvents: [.ready],
            runEvents: [.observation("running")]
        )
        let session = ExperimentSession(backend: backend)

        await session.prepare()
        await session.start()
        session.stop()

        #expect(session.state == .stopped)
        #expect(session.observations.lastEventName == "stopped")
    }

    @Test("scripted backend failure reaches terminal failed with readable message")
    @MainActor
    func scriptedFailureReachesFailed() async {
        let backend = ScriptedExperimentBackend(
            prepareEvents: [.ready],
            runEvents: [.failure("Artifact missing")]
        )
        let session = ExperimentSession(backend: backend)

        await session.prepare()
        await session.start()

        #expect(session.state == .failed("Artifact missing"))
        #expect(session.observations.lastEventName == "failed")
    }

    @Test("new session after a terminal state creates a fresh unloaded attempt")
    @MainActor
    func newSessionAfterTerminalStateCreatesFreshAttempt() async {
        let backend = ScriptedExperimentBackend(
            prepareEvents: [.ready],
            runEvents: [.observation("running")]
        )
        let session = ExperimentSession(backend: backend)

        await session.prepare()
        await session.start()
        session.stop()
        session.newSession()

        #expect(session.state == .unloaded)
        #expect(session.observations == ExperimentObservations())
    }

    @Test("failure demo intent reaches terminal failed without real inference")
    @MainActor
    func failureDemoIntentReachesFailed() {
        let session = ExperimentSession(backend: ScriptedExperimentBackend(events: []))

        session.triggerFailureDemo()

        #expect(session.state == .failed("Fake backend failure"))
        #expect(session.observations.lastEventName == "failed")
    }

    @Test("playback diagnostics reach Experiment Session observations")
    @MainActor
    func playbackDiagnosticsReachExperimentSessionObservations() async {
        let snapshot = PlaybackDiagnosticsSnapshot(
            sampleRate: 24_000,
            playbackStarted: true,
            scheduledBufferCount: 4,
            completedBufferCount: 2,
            pendingBufferCount: 2,
            scheduledSampleCount: 7_680,
            completedSampleCount: 3_840,
            pendingSampleCount: 3_840,
            pendingDurationMilliseconds: 160,
            lastScheduleGapMilliseconds: 95,
            underrunCount: 1,
            elapsedMilliseconds: 1_000
        )
        let backend = ScriptedExperimentBackend(
            prepareEvents: [.ready],
            runEvents: [.playback(.diagnostics(snapshot))]
        )
        let session = ExperimentSession(backend: backend)

        await session.prepare()
        await session.start()

        #expect(session.observations.playbackStatus == "streaming")
        #expect(session.observations.playbackScheduledDurationMilliseconds == 320)
        #expect(session.observations.playbackCompletedDurationMilliseconds == 160)
        #expect(session.observations.playbackPendingDurationMilliseconds == 160)
        #expect(session.observations.playbackScheduleGapMilliseconds == 95)
        #expect(session.observations.playbackUnderrunCount == 1)
    }
}

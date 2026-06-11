import Testing

@testable import S2STranslateCore

@Suite("Streaming Audio Input")
struct StreamingAudioInputTests {
    @Test("fixture source emits timestamped PCM chunks")
    func fixtureSourceEmitsTimestampedChunks() async throws {
        let source = FixtureAudioInputSource(sampleRate: 24_000, chunkSampleCount: 1_920, chunkCount: 3)

        let description = await source.description()
        let chunks = try await source.chunks()

        #expect(description.sampleRate == 24_000)
        #expect(description.chunkSampleCount == 1_920)
        #expect(chunks.count == 3)
        #expect(chunks[0].frameIndex == 0)
        #expect(chunks[1].timestampMilliseconds == 80)
        #expect(chunks[2].durationMilliseconds == 80)
    }

    @Test("stopped fixture source emits no chunks")
    func stoppedFixtureSourceEmitsNoChunks() async throws {
        let source = FixtureAudioInputSource(sampleRate: 24_000, chunkSampleCount: 1_920, chunkCount: 3)

        source.stop()
        let chunks = try await source.chunks()

        #expect(chunks.isEmpty)
    }

    @Test("audio backend reports chunk timing through Experiment Session observations")
    @MainActor
    func audioBackendReportsChunkTiming() async {
        let session = ExperimentSession(
            backend: AudioInputExperimentBackend(
                source: FixtureAudioInputSource(sampleRate: 24_000, chunkSampleCount: 1_920, chunkCount: 2)
            )
        )

        await session.prepare()
        await session.start()

        #expect(session.state == .running)
        #expect(session.observations.audioInputStatus == "stopped")
        #expect(session.observations.audioChunkCount == 2)
        #expect(session.observations.audioSampleRate == 24_000)
        #expect(session.observations.audioDurationMilliseconds == 160)
        #expect(session.observations.lastAudioFrameIndex == 1)
        #expect(session.observations.lastEventName == "audioInput:streamStopped")
    }

    @Test("source playback backend plays decoded input samples directly")
    @MainActor
    func sourcePlaybackBackendPlaysInputSamplesDirectly() async {
        let session = ExperimentSession(
            backend: SourceAudioPlaybackExperimentBackend(
                source: FixtureAudioInputSource(sampleRate: 24_000, chunkSampleCount: 1_920, chunkCount: 2),
                playbackSink: BufferedPlaybackSink()
            )
        )

        await session.prepare()
        await session.start()

        #expect(session.state == .running)
        #expect(session.observations.audioInputStatus == "stopped")
        #expect(session.observations.audioChunkCount == 2)
        #expect(session.observations.playbackStatus == "stopped")
        #expect(session.observations.playbackChunkCount == 2)
        #expect(session.observations.mimiEncodedFrameCount == 0)
        #expect(session.observations.hibikiStepCount == 0)
        #expect(session.observations.decodedAudioChunkCount == 0)
    }

    @Test("audio backend propagates input failures to Experiment Session")
    @MainActor
    func audioBackendPropagatesInputFailure() async {
        let session = ExperimentSession(
            backend: AudioInputExperimentBackend(
                source: FailingAudioInputSource(error: .unavailable("microphone permission denied"))
            )
        )

        await session.prepare()
        await session.start()

        #expect(session.state == .failed("Audio input unavailable: microphone permission denied"))
        #expect(session.observations.audioInputStatus == "failed: Audio input unavailable: microphone permission denied")
    }

    @Test("session stop asks the audio backend to stop the source")
    @MainActor
    func sessionStopAsksAudioBackendToStopSource() async throws {
        let source = FixtureAudioInputSource(sampleRate: 24_000, chunkSampleCount: 1_920, chunkCount: 3)
        let session = ExperimentSession(backend: AudioInputExperimentBackend(source: source))

        await session.prepare()
        await session.start()
        session.stop()
        let chunksAfterStop = try await source.chunks()

        #expect(session.state == .stopped)
        #expect(chunksAfterStop.isEmpty)
    }
}

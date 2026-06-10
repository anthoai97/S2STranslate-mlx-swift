import Foundation
import Testing

@testable import S2STranslateCore

@Suite("Streaming Mimi Decode")
struct StreamingMimiDecodeTests {
    @Test("deterministic decoder preserves state across token frames")
    func deterministicDecoderPreservesStateAcrossTokenFrames() async throws {
        let decoder = DeterministicMimiStreamingDecoder()
        let firstFrame = MimiTokenFrame(
            frameIndex: 0,
            timestampMilliseconds: 0,
            codebookCount: 16,
            tokens: Array(101..<117),
            sourceAudioFrameIndex: 0
        )
        let secondFrame = MimiTokenFrame(
            frameIndex: 1,
            timestampMilliseconds: 80,
            codebookCount: 16,
            tokens: Array(117..<133),
            sourceAudioFrameIndex: 1
        )

        let firstChunk = try await decoder.decode(firstFrame)
        let secondChunk = try await decoder.decode(secondFrame)

        #expect(firstChunk.frameIndex == 0)
        #expect(secondChunk.frameIndex == 1)
        #expect(firstChunk.samples.count == 1_920)
        #expect(secondChunk.samples.count == 1_920)
        #expect(secondChunk.timestampMilliseconds == 80)
        #expect(secondChunk.sourceTokenFrameIndex == 1)
    }

    @Test("codec playback backend reports decode and playback metrics")
    @MainActor
    func codecPlaybackBackendReportsDecodeAndPlaybackMetrics() async {
        let sink = BufferedPlaybackSink()
        let session = ExperimentSession(
            backend: MimiCodecPlaybackExperimentBackend(
                source: FixtureAudioInputSource(sampleRate: 24_000, chunkSampleCount: 1_920, chunkCount: 2),
                encoder: DeterministicMimiStreamingEncoder(),
                decoder: DeterministicMimiStreamingDecoder(),
                playbackSink: sink
            )
        )

        await session.prepare()
        await session.start()

        #expect(session.state == .running)
        #expect(session.observations.mimiDecodeStatus == "stopped")
        #expect(session.observations.decodedAudioChunkCount == 2)
        #expect(session.observations.decodedAudioSampleRate == 24_000)
        #expect(session.observations.decodedAudioDurationMilliseconds == 160)
        #expect(session.observations.lastDecodedAudioFrameIndex == 1)
        #expect(session.observations.playbackStatus == "stopped")
        #expect(session.observations.playbackChunkCount == 2)
        #expect(session.observations.playbackDurationMilliseconds == 160)
        #expect(session.observations.lastPlaybackFrameIndex == 1)
        #expect(session.observations.lastEventName == "playback:streamStopped")
        #expect(sink.bufferedChunks().map(\.frameIndex) == [0, 1])
    }

    @Test("session stop asks playback sink to stop")
    @MainActor
    func sessionStopAsksPlaybackSinkToStop() async throws {
        let sink = BufferedPlaybackSink()
        let session = ExperimentSession(
            backend: MimiCodecPlaybackExperimentBackend(
                source: FixtureAudioInputSource(sampleRate: 24_000, chunkSampleCount: 1_920, chunkCount: 1),
                encoder: DeterministicMimiStreamingEncoder(),
                decoder: DeterministicMimiStreamingDecoder(),
                playbackSink: sink
            )
        )

        await session.prepare()
        await session.start()
        session.stop()

        do {
            try await sink.receive(
                DecodedAudioChunk(
                    frameIndex: 99,
                    timestampMilliseconds: 7_920,
                    sampleRate: 24_000,
                    samples: Array(repeating: 0, count: 1_920),
                    sourceTokenFrameIndex: 99
                )
            )
            Issue.record("Expected stopped playback sink to reject chunks")
        } catch let error as PlaybackSinkError {
            #expect(error == .stopped)
        }
    }

    @Test("decode failures reach session failure state")
    @MainActor
    func decodeFailuresReachSessionFailureState() async {
        let session = ExperimentSession(
            backend: MimiCodecPlaybackExperimentBackend(
                source: FixtureAudioInputSource(sampleRate: 24_000, chunkSampleCount: 1_920, chunkCount: 1),
                encoder: DeterministicMimiStreamingEncoder(),
                decoder: FailingMimiStreamingDecoder(error: .unavailable("decoder weights missing")),
                playbackSink: BufferedPlaybackSink()
            )
        )

        await session.prepare()
        await session.start()

        #expect(session.state == .failed("Mimi decoder unavailable: decoder weights missing"))
        #expect(session.observations.mimiDecodeStatus == "failed: Mimi decoder unavailable: decoder weights missing")
    }

    @Test("playback failures reach session failure state")
    @MainActor
    func playbackFailuresReachSessionFailureState() async {
        let session = ExperimentSession(
            backend: MimiCodecPlaybackExperimentBackend(
                source: FixtureAudioInputSource(sampleRate: 24_000, chunkSampleCount: 1_920, chunkCount: 1),
                encoder: DeterministicMimiStreamingEncoder(),
                decoder: DeterministicMimiStreamingDecoder(),
                playbackSink: FailingPlaybackSink(error: .unavailable("audio route unavailable"))
            )
        )

        await session.prepare()
        await session.start()

        #expect(session.state == .failed("Playback sink unavailable: audio route unavailable"))
        #expect(session.observations.playbackStatus == "failed: Playback sink unavailable: audio route unavailable")
    }

    @Test("decoded chunk matches reference trace audio event")
    func decodedChunkMatchesReferenceTraceAudioEvent() async throws {
        let expected = try loadFixtureTrace()
        let decoder = DeterministicMimiStreamingDecoder()
        let frame = MimiTokenFrame(
            frameIndex: 0,
            timestampMilliseconds: 0,
            codebookCount: 16,
            tokens: Array(101..<117),
            sourceAudioFrameIndex: 0
        )

        let chunk = try await decoder.decode(frame)
        let expectedAudio = ReferenceTrace(
            name: expected.name,
            source: expected.source,
            events: [expected.events[4]]
        )
        let actualAudio = ReferenceTrace(
            name: expected.name,
            source: expected.source,
            events: [chunk.referenceTraceEvent]
        )

        let mismatches = ReferenceTraceComparator.compare(expected: expectedAudio, actual: actualAudio)

        #expect(mismatches.isEmpty)
    }
}

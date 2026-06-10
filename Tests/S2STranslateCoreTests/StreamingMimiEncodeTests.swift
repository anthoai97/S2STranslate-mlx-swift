import Foundation
import Testing

@testable import S2STranslateCore

@Suite("Streaming Mimi Encode")
struct StreamingMimiEncodeTests {
    @Test("deterministic encoder preserves state across chunk boundaries")
    func deterministicEncoderPreservesStateAcrossChunks() async throws {
        let encoder = DeterministicMimiStreamingEncoder()
        let firstHalf = PCMChunk(
            frameIndex: 0,
            timestampMilliseconds: 0,
            sampleRate: 24_000,
            samples: Array(repeating: 0.1, count: 960)
        )
        let secondHalf = PCMChunk(
            frameIndex: 1,
            timestampMilliseconds: 40,
            sampleRate: 24_000,
            samples: Array(repeating: 0.2, count: 960)
        )
        let fullFrame = PCMChunk(
            frameIndex: 2,
            timestampMilliseconds: 80,
            sampleRate: 24_000,
            samples: Array(repeating: 0.3, count: 1_920)
        )

        let firstFrames = try await encoder.encode(firstHalf)
        let secondFrames = try await encoder.encode(secondHalf)
        let thirdFrames = try await encoder.encode(fullFrame)

        #expect(firstFrames.isEmpty)
        #expect(secondFrames.map(\.frameIndex) == [0])
        #expect(thirdFrames.map(\.frameIndex) == [1])
        #expect(secondFrames[0].tokens.prefix(4) == [101, 102, 103, 104])
        #expect(thirdFrames[0].tokens.prefix(4) == [117, 118, 119, 120])
    }

    @Test("Mimi encode backend reports frame cadence through Experiment Session observations")
    @MainActor
    func mimiEncodeBackendReportsFrameCadence() async {
        let session = ExperimentSession(
            backend: MimiEncodeExperimentBackend(
                source: FixtureAudioInputSource(sampleRate: 24_000, chunkSampleCount: 1_920, chunkCount: 2),
                encoder: DeterministicMimiStreamingEncoder()
            )
        )

        await session.prepare()
        await session.start()

        #expect(session.state == .running)
        #expect(session.observations.audioChunkCount == 2)
        #expect(session.observations.mimiEncodeStatus == "stopped")
        #expect(session.observations.mimiEncodedFrameCount == 2)
        #expect(session.observations.mimiCodebookCount == 16)
        #expect(session.observations.mimiTokenCount == 32)
        #expect(session.observations.mimiFrameDurationMilliseconds == 80)
        #expect(session.observations.lastMimiFrameIndex == 1)
        #expect(session.observations.lastEventName == "codec:streamStopped")
    }

    @Test("Mimi encode failures reach session failure state")
    @MainActor
    func mimiEncodeFailuresReachSessionFailureState() async {
        let source = FixtureAudioInputSource(
            chunks: [
                PCMChunk(
                    frameIndex: 0,
                    timestampMilliseconds: 0,
                    sampleRate: 16_000,
                    samples: Array(repeating: 0.1, count: 1_920)
                ),
            ],
            sampleRate: 16_000,
            chunkSampleCount: 1_920
        )
        let session = ExperimentSession(
            backend: MimiEncodeExperimentBackend(
                source: source,
                encoder: DeterministicMimiStreamingEncoder()
            )
        )

        await session.prepare()
        await session.start()

        #expect(session.state == .failed("Mimi encoder sample rate unsupported: 16000 Hz"))
        #expect(session.observations.mimiEncodeStatus == "failed: Mimi encoder sample rate unsupported: 16000 Hz")
    }

    @Test("unavailable codec assets reach session failure state")
    @MainActor
    func unavailableCodecAssetsReachSessionFailureState() async {
        let session = ExperimentSession(
            backend: MimiEncodeExperimentBackend(
                source: FixtureAudioInputSource(sampleRate: 24_000, chunkSampleCount: 1_920, chunkCount: 1),
                encoder: FailingMimiStreamingEncoder(error: .unavailable("mimi weights missing"))
            )
        )

        await session.prepare()
        await session.start()

        #expect(session.state == .failed("Mimi encoder unavailable: mimi weights missing"))
        #expect(session.observations.mimiEncodeStatus == "failed: Mimi encoder unavailable: mimi weights missing")
    }

    @Test("first encoded frame matches reference trace codec event")
    func firstEncodedFrameMatchesReferenceTraceCodecEvent() async throws {
        let expected = try loadFixtureTrace()
        let encoder = DeterministicMimiStreamingEncoder()
        let chunk = PCMChunk(
            frameIndex: 0,
            timestampMilliseconds: 0,
            sampleRate: 24_000,
            samples: Array(repeating: 0.1, count: 1_920)
        )

        let frame = try #require(try await encoder.encode(chunk).first)
        let actual = ReferenceTrace(
            name: expected.name,
            source: expected.source,
            events: [
                expected.events[0],
                frame.referenceTraceEvent,
            ]
        )
        let expectedPrefix = ReferenceTrace(
            name: expected.name,
            source: expected.source,
            events: Array(expected.events.prefix(2))
        )

        let mismatches = ReferenceTraceComparator.compare(expected: expectedPrefix, actual: actual)

        #expect(mismatches.isEmpty)
    }
}

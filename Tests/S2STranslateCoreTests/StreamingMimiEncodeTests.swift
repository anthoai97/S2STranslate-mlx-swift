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

    @Test("MLX Mimi encoder passes PCM chunks to runtime without resetting state")
    func mlxMimiEncoderPassesPCMChunksToRuntimeWithoutResettingState() async throws {
        let engine = FakeStreamingMimiRuntimeEngine(
            encodedFramesByCall: [
                [],
                [MLXMimiEncodedFrame(tokens: Array(200..<216))],
                [MLXMimiEncodedFrame(tokens: Array(216..<232))],
            ]
        )
        let encoder = MLXMimiStreamingEncoder(
            runtime: MLXMimiRuntime(
                artifact: try makePreparedMimiArtifact(),
                engine: engine
            )
        )
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
        #expect(secondFrames == [
            MimiTokenFrame(
                frameIndex: 0,
                timestampMilliseconds: 0,
                codebookCount: 16,
                tokens: Array(200..<216),
                sourceAudioFrameIndex: 1
            ),
        ])
        #expect(thirdFrames == [
            MimiTokenFrame(
                frameIndex: 1,
                timestampMilliseconds: 80,
                codebookCount: 16,
                tokens: Array(216..<232),
                sourceAudioFrameIndex: 2
            ),
        ])
        #expect(engine.encodeRequests.map(\.pcmShape) == [
            [1, 1, 960],
            [1, 1, 960],
            [1, 1, 1_920],
        ])
        #expect(engine.resetEncodeCount == 0)
    }

    @Test("MLX Mimi encoder validates chunk sample rate and shape")
    func mlxMimiEncoderValidatesChunkSampleRateAndShape() async throws {
        let encoder = MLXMimiStreamingEncoder(
            runtime: MLXMimiRuntime(
                artifact: try makePreparedMimiArtifact(),
                engine: FakeStreamingMimiRuntimeEngine()
            )
        )

        await #expect(throws: MimiEncodeError.unsupportedSampleRate(16_000)) {
            try await encoder.encode(
                PCMChunk(
                    frameIndex: 0,
                    timestampMilliseconds: 0,
                    sampleRate: 16_000,
                    samples: Array(repeating: 0, count: 1_920)
                )
            )
        }
        await #expect(throws: MimiEncodeError.malformedChunk("empty chunk at audio frame 1")) {
            try await encoder.encode(
                PCMChunk(
                    frameIndex: 1,
                    timestampMilliseconds: 80,
                    sampleRate: 24_000,
                    samples: []
                )
            )
        }
    }

    @Test("MLX Mimi encoder reports invalid runtime token shape")
    func mlxMimiEncoderReportsInvalidRuntimeTokenShape() async throws {
        let encoder = MLXMimiStreamingEncoder(
            runtime: MLXMimiRuntime(
                artifact: try makePreparedMimiArtifact(),
                engine: FakeStreamingMimiRuntimeEngine(
                    encodedFramesByCall: [
                        [MLXMimiEncodedFrame(tokens: [1, 2, 3])],
                    ]
                )
            )
        )

        await #expect(
            throws: MimiEncodeError.malformedChunk(
                "encoded frame has 3 tokens, expected 16"
            )
        ) {
            try await encoder.encode(
                PCMChunk(
                    frameIndex: 0,
                    timestampMilliseconds: 0,
                    sampleRate: 24_000,
                    samples: Array(repeating: 0, count: 1_920)
                )
            )
        }
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

private func makePreparedMimiArtifact() throws -> PreparedModelArtifact {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let weightsURL = directory.appendingPathComponent("mimi.safetensors")
    try Data("fake mimi weights".utf8).write(to: weightsURL)
    return PreparedModelArtifact(
        role: "mimiWeights",
        fileName: "mimi.safetensors",
        location: weightsURL.path,
        source: .prepared
    )
}

private final class FakeStreamingMimiRuntimeEngine: MLXMimiRuntimeEngine {
    var resetEncodeCount = 0
    var resetDecodeCount = 0
    var warmupRequests: [MLXMimiWarmupRequest] = []
    var encodeRequests: [MLXMimiPCMInput] = []
    var encodedFramesByCall: [[MLXMimiEncodedFrame]]

    init(encodedFramesByCall: [[MLXMimiEncodedFrame]] = []) {
        self.encodedFramesByCall = encodedFramesByCall
    }

    func resetEncodeState() {
        resetEncodeCount += 1
    }

    func resetDecodeState() {
        resetDecodeCount += 1
    }

    func warmup(request: MLXMimiWarmupRequest) throws {
        warmupRequests.append(request)
    }

    func encode(_ input: MLXMimiPCMInput) throws -> [MLXMimiEncodedFrame] {
        encodeRequests.append(input)
        guard !encodedFramesByCall.isEmpty else { return [] }
        return encodedFramesByCall.removeFirst()
    }

    func decode(_ input: MLXMimiTokenInput) throws -> [MLXMimiDecodedChunk] {
        []
    }
}

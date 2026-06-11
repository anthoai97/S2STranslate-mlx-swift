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

        let firstChunk = try #require(try await decoder.decode(firstFrame).first)
        let secondChunk = try #require(try await decoder.decode(secondFrame).first)

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

        let chunk = try #require(try await decoder.decode(frame).first)
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

    @Test("codec playback backend skips empty decode output without placeholder playback")
    @MainActor
    func codecPlaybackBackendSkipsEmptyDecodeOutputWithoutPlaceholderPlayback() async {
        let session = ExperimentSession(
            backend: MimiCodecPlaybackExperimentBackend(
                source: FixtureAudioInputSource(sampleRate: 24_000, chunkSampleCount: 1_920, chunkCount: 2),
                encoder: DeterministicMimiStreamingEncoder(),
                decoder: EmptyMimiStreamingDecoder(),
                playbackSink: BufferedPlaybackSink()
            )
        )

        await session.prepare()
        await session.start()

        #expect(session.state == .running)
        #expect(session.observations.mimiEncodedFrameCount == 2)
        #expect(session.observations.decodedAudioChunkCount == 0)
        #expect(session.observations.playbackChunkCount == 0)
        #expect(session.observations.mimiDecodeStatus == "stopped")
    }

    @Test("MLX Mimi decoder wrapper maps runtime chunks and empty outputs")
    func mlxMimiDecoderWrapperMapsRuntimeChunksAndEmptyOutputs() async throws {
        let engine = DecodeOnlyMimiRuntimeEngine()
        engine.decodedChunks = [
            MLXMimiDecodedChunk(samples: Array(repeating: 0.25, count: 960)),
            MLXMimiDecodedChunk(samples: Array(repeating: -0.25, count: 960)),
        ]
        let decoder = MLXMimiStreamingDecoder(runtime: try makeDecodeRuntime(engine: engine))
        let frame = MimiTokenFrame(
            frameIndex: 7,
            timestampMilliseconds: 560,
            codebookCount: 16,
            tokens: Array(0..<16),
            sourceAudioFrameIndex: 3
        )

        let chunks = try await decoder.decode(frame)

        #expect(chunks.map(\.frameIndex) == [0, 1])
        #expect(chunks.map(\.sourceTokenFrameIndex) == [7, 7])
        #expect(chunks.map(\.samples.count) == [960, 960])
        engine.decodedChunks = []
        #expect(try await decoder.decode(frame).isEmpty)
    }

    @Test("MLX Mimi decoder wrapper surfaces runtime failures")
    func mlxMimiDecoderWrapperSurfacesRuntimeFailures() async throws {
        let engine = DecodeOnlyMimiRuntimeEngine()
        engine.decodeError = MimiRuntimeError.loadFailed("decode graph unavailable")
        let decoder = MLXMimiStreamingDecoder(runtime: try makeDecodeRuntime(engine: engine))
        let frame = MimiTokenFrame(
            frameIndex: 0,
            timestampMilliseconds: 0,
            codebookCount: 16,
            tokens: Array(0..<16),
            sourceAudioFrameIndex: 0
        )

        await #expect(throws: MimiDecodeError.unavailable("Mimi runtime load failed: decode graph unavailable")) {
            _ = try await decoder.decode(frame)
        }
    }
}

private struct EmptyMimiStreamingDecoder: MimiStreamingDecoder {
    func description() async -> MimiDecoderDescription {
        MimiDecoderDescription()
    }

    func decode(_ frame: MimiTokenFrame) async throws -> [DecodedAudioChunk] {
        []
    }

    func reset() {}
}

private func makeDecodeRuntime(engine: MLXMimiRuntimeEngine) throws -> MLXMimiRuntime {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let weightsURL = directory.appendingPathComponent("mimi.safetensors")
    try Data("fake".utf8).write(to: weightsURL)
    return MLXMimiRuntime(
        artifact: PreparedModelArtifact(
            role: "mimiWeights",
            fileName: "mimi.safetensors",
            location: weightsURL.path,
            source: .prepared
        ),
        engine: engine
    )
}

private final class DecodeOnlyMimiRuntimeEngine: MLXMimiRuntimeEngine {
    var decodedChunks: [MLXMimiDecodedChunk] = []
    var decodeError: Error?

    func resetEncodeState() {}
    func resetDecodeState() {}
    func warmup(request: MLXMimiWarmupRequest) throws {}
    func encode(_ input: MLXMimiPCMInput) throws -> [MLXMimiEncodedFrame] { [] }

    func decode(_ input: MLXMimiTokenInput) throws -> [MLXMimiDecodedChunk] {
        if let decodeError {
            throw decodeError
        }
        return decodedChunks
    }
}

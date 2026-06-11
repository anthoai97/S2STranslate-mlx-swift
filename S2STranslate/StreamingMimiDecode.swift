import Foundation
import Synchronization

private let twoPi = Float.pi * 2

public struct MimiDecoderDescription: Equatable, Sendable {
    public var sampleRate: Int
    public var samplesPerFrame: Int
    public var codebookCount: Int
    public var frameRate: Double

    nonisolated public init(
        sampleRate: Int = 24_000,
        samplesPerFrame: Int = 1_920,
        codebookCount: Int = 16,
        frameRate: Double = 12.5
    ) {
        self.sampleRate = sampleRate
        self.samplesPerFrame = samplesPerFrame
        self.codebookCount = codebookCount
        self.frameRate = frameRate
    }

    public var frameDurationMilliseconds: Double {
        Double(samplesPerFrame) / Double(sampleRate) * 1000
    }
}

public struct DecodedAudioChunk: Equatable, Sendable {
    public var frameIndex: Int
    public var timestampMilliseconds: Double
    public var sampleRate: Int
    public var samples: [Float]
    public var sourceTokenFrameIndex: Int

    nonisolated public init(
        frameIndex: Int,
        timestampMilliseconds: Double,
        sampleRate: Int,
        samples: [Float],
        sourceTokenFrameIndex: Int
    ) {
        self.frameIndex = frameIndex
        self.timestampMilliseconds = timestampMilliseconds
        self.sampleRate = sampleRate
        self.samples = samples
        self.sourceTokenFrameIndex = sourceTokenFrameIndex
    }

    public var durationMilliseconds: Double {
        guard sampleRate > 0 else { return 0 }
        return Double(samples.count) / Double(sampleRate) * 1000
    }

    public var referenceTraceEvent: ReferenceTraceEvent {
        ReferenceTraceEvent(
            stream: .audio,
            name: "mimiDecodeStep",
            frameIndex: frameIndex,
            shape: [1, 1, samples.count],
            cadenceMilliseconds: durationMilliseconds
        )
    }
}

public enum MimiDecodeEvent: Equatable, Sendable {
    case streamStarted(MimiDecoderDescription)
    case chunk(DecodedAudioChunk)
    case streamStopped
    case streamFailed(String)

    var name: String {
        switch self {
        case .streamStarted:
            "decodeStreamStarted"
        case .chunk:
            "mimiDecodeStep"
        case .streamStopped:
            "decodeStreamStopped"
        case .streamFailed:
            "decodeStreamFailed"
        }
    }

    var isStreamBoundary: Bool {
        switch self {
        case .streamStarted, .streamStopped, .streamFailed:
            true
        case .chunk:
            false
        }
    }
}

public enum PlaybackEvent: Equatable, Sendable {
    case streamStarted(sampleRate: Int)
    case chunk(DecodedAudioChunk)
    case streamStopped
    case streamFailed(String)

    var name: String {
        switch self {
        case .streamStarted:
            "streamStarted"
        case .chunk:
            "chunk"
        case .streamStopped:
            "streamStopped"
        case .streamFailed:
            "streamFailed"
        }
    }

    var isStreamBoundary: Bool {
        switch self {
        case .streamStarted, .streamStopped, .streamFailed:
            true
        case .chunk:
            false
        }
    }
}

public enum MimiDecodeError: Error, Equatable, Sendable {
    case unavailable(String)
    case malformedTokenFrame(String)
    case unsupportedCodebookCount(Int)

    public var userVisibleMessage: String {
        switch self {
        case let .unavailable(message):
            "Mimi decoder unavailable: \(message)"
        case let .malformedTokenFrame(message):
            "Mimi decoder token frame malformed: \(message)"
        case let .unsupportedCodebookCount(count):
            "Mimi decoder codebook count unsupported: \(count)"
        }
    }
}

public enum PlaybackSinkError: Error, Equatable, Sendable {
    case unavailable(String)
    case stopped

    public var userVisibleMessage: String {
        switch self {
        case let .unavailable(message):
            "Playback sink unavailable: \(message)"
        case .stopped:
            "Playback sink stopped"
        }
    }
}

public protocol MimiStreamingDecoder: Sendable {
    func description() async -> MimiDecoderDescription
    func decode(_ frame: MimiTokenFrame) async throws -> [DecodedAudioChunk]
    func reset()
}

public protocol PlaybackSink: Sendable {
    func start(sampleRate: Int) async throws
    func receive(_ chunk: DecodedAudioChunk) async throws
    func stop()
    func reset()
}

public final class DeterministicMimiStreamingDecoder: MimiStreamingDecoder, @unchecked Sendable {
    private let decoderDescription: MimiDecoderDescription
    private let state = Mutex(DeterministicMimiStreamingDecoderState())

    public init(description: MimiDecoderDescription = MimiDecoderDescription()) {
        self.decoderDescription = description
    }

    public func description() async -> MimiDecoderDescription {
        decoderDescription
    }

    public func decode(_ frame: MimiTokenFrame) async throws -> [DecodedAudioChunk] {
        guard frame.codebookCount == decoderDescription.codebookCount else {
            throw MimiDecodeError.unsupportedCodebookCount(frame.codebookCount)
        }

        guard frame.tokens.count == decoderDescription.codebookCount else {
            throw MimiDecodeError.malformedTokenFrame(
                "expected \(decoderDescription.codebookCount) tokens, got \(frame.tokens.count)"
            )
        }

        return state.withLock { state in
            let outputFrameIndex = state.nextFrameIndex
            let tokenOffset = Float((frame.tokens.first ?? 0) % 12)
            let frequency = Float(220) * pow(Float(2), tokenOffset / 12)
            let amplitude = Float(0.08)
            let samples = (0..<decoderDescription.samplesPerFrame).map { sampleIndex in
                let absoluteSampleIndex = state.nextSampleIndex + sampleIndex
                let phase = twoPi * frequency * Float(absoluteSampleIndex) / Float(decoderDescription.sampleRate)
                let fadeIn = min(1, Float(sampleIndex) / 128)
                let fadeOut = min(1, Float(decoderDescription.samplesPerFrame - sampleIndex - 1) / 128)
                let envelope = min(fadeIn, fadeOut)
                return sin(phase) * amplitude * envelope
            }
            state.nextFrameIndex += 1
            state.nextSampleIndex += decoderDescription.samplesPerFrame
            let chunk = DecodedAudioChunk(
                frameIndex: outputFrameIndex,
                timestampMilliseconds: Double(outputFrameIndex) * decoderDescription.frameDurationMilliseconds,
                sampleRate: decoderDescription.sampleRate,
                samples: samples,
                sourceTokenFrameIndex: frame.frameIndex
            )
            return [chunk]
        }
    }

    public func reset() {
        state.withLock { state in
            state.nextFrameIndex = 0
            state.nextSampleIndex = 0
        }
    }
}

private struct DeterministicMimiStreamingDecoderState: Sendable {
    var nextFrameIndex = 0
    var nextSampleIndex = 0
}

public final class MLXMimiStreamingDecoder: MimiStreamingDecoder, @unchecked Sendable {
    private let runtime: MLXMimiRuntime
    private let decoderDescription: MimiDecoderDescription
    private let state = Mutex(MLXMimiStreamingDecoderState())

    public init(runtime: MLXMimiRuntime) {
        self.runtime = runtime
        self.decoderDescription = MimiDecoderDescription(
            sampleRate: runtime.configuration.sampleRate,
            samplesPerFrame: runtime.configuration.samplesPerFrame,
            codebookCount: runtime.configuration.codebookCount,
            frameRate: runtime.configuration.frameRate
        )
    }

    public func description() async -> MimiDecoderDescription {
        decoderDescription
    }

    public func decode(_ frame: MimiTokenFrame) async throws -> [DecodedAudioChunk] {
        guard frame.codebookCount == decoderDescription.codebookCount else {
            throw MimiDecodeError.unsupportedCodebookCount(frame.codebookCount)
        }

        guard frame.tokens.count == decoderDescription.codebookCount else {
            throw MimiDecodeError.malformedTokenFrame(
                "expected \(decoderDescription.codebookCount) tokens, got \(frame.tokens.count)"
            )
        }

        let decodedChunks: [MLXMimiDecodedChunk]
        do {
            decodedChunks = try runtime.decode(
                MLXMimiTokenInput(tokens: frame.tokens, codebookCount: frame.codebookCount)
            )
        } catch let error as MimiRuntimeError {
            throw MimiDecodeError.unavailable(error.userVisibleMessage)
        } catch {
            throw MimiDecodeError.unavailable(String(describing: error))
        }

        guard !decodedChunks.isEmpty else { return [] }

        return state.withLock { state in
            decodedChunks.map { decodedChunk in
                let frameIndex = state.nextFrameIndex
                state.nextFrameIndex += 1
                return DecodedAudioChunk(
                    frameIndex: frameIndex,
                    timestampMilliseconds: Double(frameIndex) * decoderDescription.frameDurationMilliseconds,
                    sampleRate: decoderDescription.sampleRate,
                    samples: decodedChunk.samples,
                    sourceTokenFrameIndex: frame.frameIndex
                )
            }
        }
    }

    public func reset() {
        runtime.resetDecodeState()
        state.withLock { state in
            state.nextFrameIndex = 0
        }
    }
}

private struct MLXMimiStreamingDecoderState: Sendable {
    var nextFrameIndex = 0
}

public struct FailingMimiStreamingDecoder: MimiStreamingDecoder, Sendable {
    private let decoderDescription: MimiDecoderDescription
    private let error: MimiDecodeError

    public init(
        description: MimiDecoderDescription = MimiDecoderDescription(),
        error: MimiDecodeError
    ) {
        self.decoderDescription = description
        self.error = error
    }

    public func description() async -> MimiDecoderDescription {
        decoderDescription
    }

    public func decode(_ frame: MimiTokenFrame) async throws -> [DecodedAudioChunk] {
        throw error
    }

    public func reset() {}
}

public final class BufferedPlaybackSink: PlaybackSink, @unchecked Sendable {
    private let state = Mutex(BufferedPlaybackSinkState())

    public init() {}

    public func start(sampleRate: Int) async throws {
        state.withLock { state in
            state.sampleRate = sampleRate
            state.stopped = false
            state.chunks.removeAll()
        }
    }

    public func receive(_ chunk: DecodedAudioChunk) async throws {
        try state.withLock { state in
            if state.stopped {
                throw PlaybackSinkError.stopped
            }
            state.chunks.append(chunk)
        }
    }

    public func stop() {
        state.withLock { state in
            state.stopped = true
        }
    }

    public func reset() {
        state.withLock { state in
            state.sampleRate = nil
            state.stopped = false
            state.chunks.removeAll()
        }
    }

    public func bufferedChunks() -> [DecodedAudioChunk] {
        state.withLock { state in
            state.chunks
        }
    }
}

private struct BufferedPlaybackSinkState: Sendable {
    var sampleRate: Int?
    var stopped = false
    var chunks: [DecodedAudioChunk] = []
}

public struct FailingPlaybackSink: PlaybackSink, Sendable {
    private let error: PlaybackSinkError

    public init(error: PlaybackSinkError) {
        self.error = error
    }

    public func start(sampleRate: Int) async throws {
        throw error
    }

    public func receive(_ chunk: DecodedAudioChunk) async throws {
        throw error
    }

    public func stop() {}
    public func reset() {}
}

public struct MimiCodecPlaybackExperimentBackend: ExperimentBackend, Sendable {
    private let source: any AudioInputSource
    private let encoder: any MimiStreamingEncoder
    private let decoder: any MimiStreamingDecoder
    private let playbackSink: any PlaybackSink

    nonisolated public init(
        source: any AudioInputSource,
        encoder: any MimiStreamingEncoder,
        decoder: any MimiStreamingDecoder,
        playbackSink: any PlaybackSink
    ) {
        self.source = source
        self.encoder = encoder
        self.decoder = decoder
        self.playbackSink = playbackSink
    }

    public func prepareEvents() async -> [ExperimentEvent] {
        [.ready]
    }

    public func runEvents() async -> [ExperimentEvent] {
        encoder.reset()
        decoder.reset()
        playbackSink.reset()

        let audioDescription = await source.description()
        let encoderDescription = await encoder.description()
        let decoderDescription = await decoder.description()
        var events: [ExperimentEvent] = [
            .audioInput(.streamStarted(sampleRate: audioDescription.sampleRate)),
            .mimiEncode(.streamStarted(encoderDescription)),
            .mimiDecode(.streamStarted(decoderDescription)),
        ]

        do {
            try await playbackSink.start(sampleRate: decoderDescription.sampleRate)
            events.append(.playback(.streamStarted(sampleRate: decoderDescription.sampleRate)))

            for chunk in try await source.chunks() {
                events.append(.audioInput(.chunk(chunk)))
                for frame in try await encoder.encode(chunk) {
                    events.append(.mimiEncode(.frame(frame)))
                    for decoded in try await decoder.decode(frame) {
                        events.append(.mimiDecode(.chunk(decoded)))
                        try await playbackSink.receive(decoded)
                        events.append(.playback(.chunk(decoded)))
                    }
                }
            }

            events.append(.audioInput(.streamStopped))
            events.append(.mimiEncode(.streamStopped))
            events.append(.mimiDecode(.streamStopped))
            events.append(.playback(.streamStopped))
        } catch let error as AudioInputError {
            events.append(.audioInput(.streamFailed(error.userVisibleMessage)))
            events.append(.failure(error.userVisibleMessage))
        } catch let error as MimiEncodeError {
            events.append(.mimiEncode(.streamFailed(error.userVisibleMessage)))
            events.append(.failure(error.userVisibleMessage))
        } catch let error as MimiDecodeError {
            events.append(.mimiDecode(.streamFailed(error.userVisibleMessage)))
            events.append(.failure(error.userVisibleMessage))
        } catch let error as PlaybackSinkError {
            events.append(.playback(.streamFailed(error.userVisibleMessage)))
            events.append(.failure(error.userVisibleMessage))
        } catch {
            let message = PlaybackSinkError.unavailable(String(describing: error)).userVisibleMessage
            events.append(.playback(.streamFailed(message)))
            events.append(.failure(message))
        }

        return events
    }

    public func stop() {
        source.stop()
        encoder.reset()
        decoder.reset()
        playbackSink.stop()
    }
}

public struct ArtifactAudioMimiPlaybackExperimentBackend: ExperimentBackend, Sendable {
    private let artifactBackend: ModelArtifactExperimentBackend
    private let codecBackend: MimiCodecPlaybackExperimentBackend

    public init(
        artifactPreparer: ModelArtifactPreparer,
        audioSource: any AudioInputSource,
        mimiEncoder: any MimiStreamingEncoder,
        mimiDecoder: any MimiStreamingDecoder,
        playbackSink: any PlaybackSink
    ) {
        self.artifactBackend = ModelArtifactExperimentBackend(preparer: artifactPreparer)
        self.codecBackend = MimiCodecPlaybackExperimentBackend(
            source: audioSource,
            encoder: mimiEncoder,
            decoder: mimiDecoder,
            playbackSink: playbackSink
        )
    }

    public func prepareEvents() async -> [ExperimentEvent] {
        await artifactBackend.prepareEvents()
    }

    public func runEvents() async -> [ExperimentEvent] {
        await codecBackend.runEvents()
    }

    public func stop() {
        codecBackend.stop()
    }
}

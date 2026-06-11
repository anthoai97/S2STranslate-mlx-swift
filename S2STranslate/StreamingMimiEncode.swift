import Foundation
import Synchronization

public struct MimiEncoderDescription: Equatable, Sendable {
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

public struct MimiTokenFrame: Equatable, Sendable {
    public var frameIndex: Int
    public var timestampMilliseconds: Double
    public var codebookCount: Int
    public var tokens: [Int]
    public var sourceAudioFrameIndex: Int

    nonisolated public init(
        frameIndex: Int,
        timestampMilliseconds: Double,
        codebookCount: Int,
        tokens: [Int],
        sourceAudioFrameIndex: Int
    ) {
        self.frameIndex = frameIndex
        self.timestampMilliseconds = timestampMilliseconds
        self.codebookCount = codebookCount
        self.tokens = tokens
        self.sourceAudioFrameIndex = sourceAudioFrameIndex
    }

    public var referenceTraceEvent: ReferenceTraceEvent {
        ReferenceTraceEvent(
            stream: .codec,
            name: "mimiEncodeStep",
            frameIndex: frameIndex,
            shape: [1, codebookCount],
            tokens: Array(tokens.prefix(4)),
            cadenceMilliseconds: 80
        )
    }
}

public enum MimiEncodeEvent: Equatable, Sendable {
    case streamStarted(MimiEncoderDescription)
    case frame(MimiTokenFrame)
    case streamStopped
    case streamFailed(String)

    var name: String {
        switch self {
        case .streamStarted:
            "streamStarted"
        case .frame:
            "mimiEncodeStep"
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
        case .frame:
            false
        }
    }
}

public enum MimiEncodeError: Error, Equatable, Sendable {
    case unavailable(String)
    case unsupportedSampleRate(Int)
    case malformedChunk(String)

    public var userVisibleMessage: String {
        switch self {
        case let .unavailable(message):
            "Mimi encoder unavailable: \(message)"
        case let .unsupportedSampleRate(sampleRate):
            "Mimi encoder sample rate unsupported: \(sampleRate) Hz"
        case let .malformedChunk(message):
            "Mimi encoder chunk malformed: \(message)"
        }
    }
}

public protocol MimiStreamingEncoder: Sendable {
    func description() async -> MimiEncoderDescription
    func encode(_ chunk: PCMChunk) async throws -> [MimiTokenFrame]
    func reset()
}

public final class DeterministicMimiStreamingEncoder: MimiStreamingEncoder, @unchecked Sendable {
    private let encoderDescription: MimiEncoderDescription
    private let tokenBase: Int
    private let state = Mutex(DeterministicMimiStreamingEncoderState())

    public init(
        description: MimiEncoderDescription = MimiEncoderDescription(),
        tokenBase: Int = 101
    ) {
        self.encoderDescription = description
        self.tokenBase = tokenBase
    }

    public func description() async -> MimiEncoderDescription {
        encoderDescription
    }

    public func encode(_ chunk: PCMChunk) async throws -> [MimiTokenFrame] {
        guard chunk.sampleRate == encoderDescription.sampleRate else {
            throw MimiEncodeError.unsupportedSampleRate(chunk.sampleRate)
        }

        guard !chunk.samples.isEmpty else {
            throw MimiEncodeError.malformedChunk("empty chunk at audio frame \(chunk.frameIndex)")
        }

        return state.withLock { state in
            state.bufferedSampleCount += chunk.samples.count
            var frames: [MimiTokenFrame] = []

            while state.bufferedSampleCount >= encoderDescription.samplesPerFrame {
                let frameIndex = state.nextFrameIndex
                let tokenOffset = frameIndex * encoderDescription.codebookCount
                let tokens = (0..<encoderDescription.codebookCount).map { tokenBase + tokenOffset + $0 }
                frames.append(
                    MimiTokenFrame(
                        frameIndex: frameIndex,
                        timestampMilliseconds: Double(frameIndex) * encoderDescription.frameDurationMilliseconds,
                        codebookCount: encoderDescription.codebookCount,
                        tokens: tokens,
                        sourceAudioFrameIndex: chunk.frameIndex
                    )
                )
                state.nextFrameIndex += 1
                state.bufferedSampleCount -= encoderDescription.samplesPerFrame
            }

            return frames
        }
    }

    public func reset() {
        state.withLock { state in
            state.bufferedSampleCount = 0
            state.nextFrameIndex = 0
        }
    }
}

public final class MLXMimiStreamingEncoder: MimiStreamingEncoder, @unchecked Sendable {
    private let runtime: MLXMimiRuntime
    private let encoderDescription: MimiEncoderDescription
    private let state = Mutex(MLXMimiStreamingEncoderState())

    public init(runtime: MLXMimiRuntime) {
        self.runtime = runtime
        self.encoderDescription = MimiEncoderDescription(
            sampleRate: runtime.configuration.sampleRate,
            samplesPerFrame: runtime.configuration.samplesPerFrame,
            codebookCount: runtime.configuration.codebookCount,
            frameRate: runtime.configuration.frameRate
        )
    }

    public func description() async -> MimiEncoderDescription {
        encoderDescription
    }

    public func encode(_ chunk: PCMChunk) async throws -> [MimiTokenFrame] {
        guard chunk.sampleRate == encoderDescription.sampleRate else {
            throw MimiEncodeError.unsupportedSampleRate(chunk.sampleRate)
        }

        guard !chunk.samples.isEmpty else {
            throw MimiEncodeError.malformedChunk("empty chunk at audio frame \(chunk.frameIndex)")
        }

        let encodedFrames: [MLXMimiEncodedFrame]
        do {
            encodedFrames = try runtime.encode(
                MLXMimiPCMInput(
                    samples: chunk.samples,
                    sampleRate: chunk.sampleRate,
                    pcmShape: [1, 1, chunk.samples.count]
                )
            )
        } catch let error as MimiRuntimeError {
            throw MimiEncodeError.unavailable(error.userVisibleMessage)
        } catch {
            throw MimiEncodeError.unavailable(String(describing: error))
        }

        guard !encodedFrames.isEmpty else { return [] }

        return try state.withLock { state in
            try encodedFrames.map { encodedFrame in
                guard encodedFrame.tokens.count == encoderDescription.codebookCount else {
                    throw MimiEncodeError.malformedChunk(
                        "encoded frame has \(encodedFrame.tokens.count) tokens, expected \(encoderDescription.codebookCount)"
                    )
                }

                let frameIndex = state.nextFrameIndex
                state.nextFrameIndex += 1
                return MimiTokenFrame(
                    frameIndex: frameIndex,
                    timestampMilliseconds: Double(frameIndex) * encoderDescription.frameDurationMilliseconds,
                    codebookCount: encoderDescription.codebookCount,
                    tokens: encodedFrame.tokens,
                    sourceAudioFrameIndex: chunk.frameIndex
                )
            }
        }
    }

    public func reset() {
        runtime.resetEncodeState()
        state.withLock { state in
            state.nextFrameIndex = 0
        }
    }
}

private struct MLXMimiStreamingEncoderState: Sendable {
    var nextFrameIndex = 0
}

private struct DeterministicMimiStreamingEncoderState: Sendable {
    var bufferedSampleCount = 0
    var nextFrameIndex = 0
}

public struct FailingMimiStreamingEncoder: MimiStreamingEncoder, Sendable {
    private let encoderDescription: MimiEncoderDescription
    private let error: MimiEncodeError

    public init(
        description: MimiEncoderDescription = MimiEncoderDescription(),
        error: MimiEncodeError
    ) {
        self.encoderDescription = description
        self.error = error
    }

    public func description() async -> MimiEncoderDescription {
        encoderDescription
    }

    public func encode(_ chunk: PCMChunk) async throws -> [MimiTokenFrame] {
        throw error
    }

    public func reset() {}
}

public struct MimiEncodeExperimentBackend: ExperimentBackend, Sendable {
    private let source: any AudioInputSource
    private let encoder: any MimiStreamingEncoder

    nonisolated public init(
        source: any AudioInputSource,
        encoder: any MimiStreamingEncoder
    ) {
        self.source = source
        self.encoder = encoder
    }

    public func prepareEvents() async -> [ExperimentEvent] {
        [.ready]
    }

    public func runEvents() async -> [ExperimentEvent] {
        source.reset()
        encoder.reset()

        let audioDescription = await source.description()
        let encoderDescription = await encoder.description()
        var events: [ExperimentEvent] = [
            .audioInput(.streamStarted(sampleRate: audioDescription.sampleRate)),
            .mimiEncode(.streamStarted(encoderDescription)),
        ]

        do {
            for chunk in try await source.chunks() {
                events.append(.audioInput(.chunk(chunk)))
                let frames = try await encoder.encode(chunk)
                events.append(contentsOf: frames.map { .mimiEncode(.frame($0)) })
            }
            events.append(.audioInput(.streamStopped))
            events.append(.mimiEncode(.streamStopped))
        } catch let error as AudioInputError {
            events.append(.audioInput(.streamFailed(error.userVisibleMessage)))
            events.append(.failure(error.userVisibleMessage))
        } catch let error as MimiEncodeError {
            events.append(.mimiEncode(.streamFailed(error.userVisibleMessage)))
            events.append(.failure(error.userVisibleMessage))
        } catch {
            let message = MimiEncodeError.unavailable(String(describing: error)).userVisibleMessage
            events.append(.mimiEncode(.streamFailed(message)))
            events.append(.failure(message))
        }

        return events
    }

    public func stop() {
        source.stop()
        encoder.reset()
    }
}

public struct ArtifactAudioMimiExperimentBackend: ExperimentBackend, Sendable {
    private let artifactBackend: ModelArtifactExperimentBackend
    private let encodeBackend: MimiEncodeExperimentBackend

    public init(
        artifactPreparer: ModelArtifactPreparer,
        audioSource: any AudioInputSource,
        mimiEncoder: any MimiStreamingEncoder
    ) {
        self.artifactBackend = ModelArtifactExperimentBackend(preparer: artifactPreparer)
        self.encodeBackend = MimiEncodeExperimentBackend(source: audioSource, encoder: mimiEncoder)
    }

    public func prepareEvents() async -> [ExperimentEvent] {
        await artifactBackend.prepareEvents()
    }

    public func runEvents() async -> [ExperimentEvent] {
        await encodeBackend.runEvents()
    }

    public func stop() {
        encodeBackend.stop()
    }
}

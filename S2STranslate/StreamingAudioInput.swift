import Foundation

public struct PCMChunk: Equatable, Sendable {
    public var frameIndex: Int
    public var timestampMilliseconds: Double
    public var sampleRate: Int
    public var samples: [Float]

    nonisolated public init(
        frameIndex: Int,
        timestampMilliseconds: Double,
        sampleRate: Int,
        samples: [Float]
    ) {
        self.frameIndex = frameIndex
        self.timestampMilliseconds = timestampMilliseconds
        self.sampleRate = sampleRate
        self.samples = samples
    }

    public var durationMilliseconds: Double {
        guard sampleRate > 0 else { return 0 }
        return Double(samples.count) / Double(sampleRate) * 1000
    }
}

public enum AudioInputEvent: Equatable, Sendable {
    case streamStarted(sampleRate: Int)
    case chunk(PCMChunk)
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

public struct AudioInputDescription: Equatable, Sendable {
    public var sampleRate: Int
    public var chunkSampleCount: Int

    nonisolated public init(sampleRate: Int, chunkSampleCount: Int) {
        self.sampleRate = sampleRate
        self.chunkSampleCount = chunkSampleCount
    }
}

public enum AudioInputError: Error, Equatable, Sendable {
    case unavailable(String)
    case unsupportedSampleRate(Int)
    case malformedChunk(String)

    public var userVisibleMessage: String {
        switch self {
        case let .unavailable(message):
            "Audio input unavailable: \(message)"
        case let .unsupportedSampleRate(sampleRate):
            "Audio input sample rate unsupported: \(sampleRate) Hz"
        case let .malformedChunk(message):
            "Audio input chunk malformed: \(message)"
        }
    }
}

public protocol AudioInputSource: Sendable {
    func description() async -> AudioInputDescription
    func chunks() async throws -> [PCMChunk]
    func stop()
    func reset()
}

public extension AudioInputSource {
    func reset() {}
}

public final class FixtureAudioInputSource: AudioInputSource, @unchecked Sendable {
    private let sourceDescription: AudioInputDescription
    private let plannedChunks: [PCMChunk]
    private let lock = NSLock()
    private var stopped = false

    public init(sampleRate: Int = 24_000, chunkSampleCount: Int = 1_920, chunkCount: Int = 4) {
        self.sourceDescription = AudioInputDescription(
            sampleRate: sampleRate,
            chunkSampleCount: chunkSampleCount
        )
        self.plannedChunks = (0..<chunkCount).map { frameIndex in
            let samples = (0..<chunkSampleCount).map { sampleIndex in
                Float((frameIndex + sampleIndex) % 17) / 16
            }
            return PCMChunk(
                frameIndex: frameIndex,
                timestampMilliseconds: Double(frameIndex * chunkSampleCount) / Double(sampleRate) * 1000,
                sampleRate: sampleRate,
                samples: samples
            )
        }
    }

    public init(chunks: [PCMChunk], sampleRate: Int, chunkSampleCount: Int) {
        self.sourceDescription = AudioInputDescription(
            sampleRate: sampleRate,
            chunkSampleCount: chunkSampleCount
        )
        self.plannedChunks = chunks
    }

    public func description() async -> AudioInputDescription {
        sourceDescription
    }

    public func chunks() async throws -> [PCMChunk] {
        var emittedChunks: [PCMChunk] = []

        for chunk in plannedChunks {
            guard !isStopped else { break }
            guard chunk.sampleRate == sourceDescription.sampleRate else {
                throw AudioInputError.unsupportedSampleRate(chunk.sampleRate)
            }
            guard !chunk.samples.isEmpty else {
                throw AudioInputError.malformedChunk("empty chunk at frame \(chunk.frameIndex)")
            }
            emittedChunks.append(chunk)
        }

        return emittedChunks
    }

    public func stop() {
        lock.lock()
        defer { lock.unlock() }
        stopped = true
    }

    public func reset() {
        lock.lock()
        defer { lock.unlock() }
        stopped = false
    }

    private var isStopped: Bool {
        lock.lock()
        defer { lock.unlock() }
        return stopped
    }
}

public final class FailingAudioInputSource: AudioInputSource, @unchecked Sendable {
    private let sourceDescription: AudioInputDescription
    private let error: AudioInputError

    public init(error: AudioInputError) {
        self.sourceDescription = AudioInputDescription(sampleRate: 24_000, chunkSampleCount: 1_920)
        self.error = error
    }

    public func description() async -> AudioInputDescription {
        sourceDescription
    }

    public func chunks() async throws -> [PCMChunk] {
        throw error
    }

    public func stop() {}
}

public struct AudioInputExperimentBackend: ExperimentBackend, Sendable {
    private let source: any AudioInputSource

    nonisolated public init(source: any AudioInputSource) {
        self.source = source
    }

    public func prepareEvents() async -> [ExperimentEvent] {
        [.ready]
    }

    public func runEvents() async -> [ExperimentEvent] {
        source.reset()

        let description = await source.description()
        var events: [ExperimentEvent] = [
            .audioInput(.streamStarted(sampleRate: description.sampleRate)),
        ]

        do {
            let chunks = try await source.chunks()
            events.append(contentsOf: chunks.map { .audioInput(.chunk($0)) })
            events.append(.audioInput(.streamStopped))
        } catch let error as AudioInputError {
            events.append(.audioInput(.streamFailed(error.userVisibleMessage)))
            events.append(.failure(error.userVisibleMessage))
        } catch {
            let message = AudioInputError.unavailable(String(describing: error)).userVisibleMessage
            events.append(.audioInput(.streamFailed(message)))
            events.append(.failure(message))
        }

        return events
    }

    public func stop() {
        source.stop()
    }
}

public struct SourceAudioPlaybackExperimentBackend: ExperimentBackend, Sendable {
    private let source: any AudioInputSource
    private let playbackSink: any PlaybackSink

    public init(
        source: any AudioInputSource,
        playbackSink: any PlaybackSink
    ) {
        self.source = source
        self.playbackSink = playbackSink
    }

    public func prepareEvents() async -> [ExperimentEvent] {
        [.ready]
    }

    public func runEvents() async -> [ExperimentEvent] {
        source.reset()
        playbackSink.reset()

        let description = await source.description()
        var events: [ExperimentEvent] = [
            .audioInput(.streamStarted(sampleRate: description.sampleRate)),
        ]

        do {
            try await playbackSink.start(sampleRate: description.sampleRate)
            events.append(.playback(.streamStarted(sampleRate: description.sampleRate)))

            for chunk in try await source.chunks() {
                events.append(.audioInput(.chunk(chunk)))
                let decoded = DecodedAudioChunk(
                    frameIndex: chunk.frameIndex,
                    timestampMilliseconds: chunk.timestampMilliseconds,
                    sampleRate: chunk.sampleRate,
                    samples: chunk.samples,
                    sourceTokenFrameIndex: chunk.frameIndex
                )
                try await playbackSink.receive(decoded)
                events.append(.playback(.chunk(decoded)))
            }

            events.append(.audioInput(.streamStopped))
            try await playbackSink.finish()
            events.append(.playback(.streamStopped))
        } catch let error as AudioInputError {
            events.append(.audioInput(.streamFailed(error.userVisibleMessage)))
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
        playbackSink.stop()
    }
}

public struct ArtifactAndAudioExperimentBackend: ExperimentBackend, Sendable {
    private let artifactBackend: ModelArtifactExperimentBackend
    private let audioBackend: AudioInputExperimentBackend

    public init(
        artifactPreparer: ModelArtifactPreparer,
        audioSource: any AudioInputSource
    ) {
        self.artifactBackend = ModelArtifactExperimentBackend(preparer: artifactPreparer)
        self.audioBackend = AudioInputExperimentBackend(source: audioSource)
    }

    public func prepareEvents() async -> [ExperimentEvent] {
        await artifactBackend.prepareEvents()
    }

    public func runEvents() async -> [ExperimentEvent] {
        await audioBackend.runEvents()
    }

    public func stop() {
        audioBackend.stop()
    }
}

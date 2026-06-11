import Foundation
import OSLog
import Synchronization

private let hibikiBackendLogger = Logger(subsystem: "andyq.S2STranslate", category: "HibikiBackend")

public struct HibikiGenerationConfiguration: Equatable, Sendable {
    public var temperature: Double
    public var textTemperature: Double
    public var topK: Int
    public var textTopK: Int
    public var voiceTransferEnabled: Bool
    public var tailSilenceFrameCount: Int
    public var postInputPaddingStopFrameCount: Int

    nonisolated public init(
        temperature: Double = 0.8,
        textTemperature: Double = 0.8,
        topK: Int = 250,
        textTopK: Int = 250,
        voiceTransferEnabled: Bool = false,
        tailSilenceFrameCount: Int = 0,
        postInputPaddingStopFrameCount: Int = 12
    ) {
        self.temperature = temperature
        self.textTemperature = textTemperature
        self.topK = topK
        self.textTopK = textTopK
        self.voiceTransferEnabled = voiceTransferEnabled
        self.tailSilenceFrameCount = tailSilenceFrameCount
        self.postInputPaddingStopFrameCount = postInputPaddingStopFrameCount
    }
}

public struct HibikiInferenceDescription: Equatable, Sendable {
    public var modelRevision: String
    public var artifactCount: Int
    public var configuration: HibikiGenerationConfiguration

    nonisolated public init(
        modelRevision: String,
        artifactCount: Int,
        configuration: HibikiGenerationConfiguration
    ) {
        self.modelRevision = modelRevision
        self.artifactCount = artifactCount
        self.configuration = configuration
    }
}

public struct HibikiTextOutput: Equatable, Sendable {
    public var frameIndex: Int
    public var token: Int
    public var piece: String?
    public var candidateTokens: [Int]

    nonisolated public init(
        frameIndex: Int,
        token: Int,
        piece: String? = nil,
        candidateTokens: [Int] = []
    ) {
        self.frameIndex = frameIndex
        self.token = token
        self.piece = piece
        self.candidateTokens = candidateTokens
    }

    public var isVisible: Bool {
        piece?.isEmpty == false
    }

    public var isBlankOrPadding: Bool {
        HibikiTextTokenContract.isBlankOrPadding(token)
    }

    public var referenceTraceEvent: ReferenceTraceEvent {
        ReferenceTraceEvent(
            stream: .text,
            name: isVisible ? "textToken" : "skipBlankOrPadding",
            frameIndex: frameIndex,
            tokens: [token]
        )
    }
}

public enum HibikiTextTokenContract {
    public static let blankOrPaddingTokens: Set<Int> = [0, 3]

    public static func isBlankOrPadding(_ token: Int) -> Bool {
        blankOrPaddingTokens.contains(token)
    }

    public static func normalizeSentencePiece(_ piece: String) -> String {
        piece.replacingOccurrences(of: "\u{2581}", with: " ")
    }
}

public protocol HibikiTextTokenDecoding: Sendable {
    func piece(for token: Int) -> String?
}

public struct EmptyHibikiTextTokenDecoder: HibikiTextTokenDecoding {
    public init() {}

    public func piece(for token: Int) -> String? {
        nil
    }
}

public struct DictionaryHibikiTextTokenDecoder: HibikiTextTokenDecoding {
    private let piecesByToken: [Int: String]

    public init(piecesByToken: [Int: String]) {
        self.piecesByToken = piecesByToken
    }

    public func piece(for token: Int) -> String? {
        guard !HibikiTextTokenContract.isBlankOrPadding(token),
              let piece = piecesByToken[token] else {
            return nil
        }
        return HibikiTextTokenContract.normalizeSentencePiece(piece)
    }
}

public struct HibikiSampledToken: Equatable, Sendable {
    public var token: Int
    public var candidateTokens: [Int]

    public init(token: Int, candidateTokens: [Int]) {
        self.token = token
        self.candidateTokens = candidateTokens
    }
}

public enum HibikiTokenSamplingError: Error, Equatable, Sendable {
    case emptyLogits
    case invalidTemperature(Double)
    case invalidRandomValue(Double)
}

public struct HibikiTopKTokenSampler: Sendable {
    public init() {}

    public func sample(
        logits: [Float],
        temperature: Double,
        topK: Int,
        randomUnit: Double
    ) throws -> HibikiSampledToken {
        guard !logits.isEmpty else {
            throw HibikiTokenSamplingError.emptyLogits
        }
        guard temperature >= 0 else {
            throw HibikiTokenSamplingError.invalidTemperature(temperature)
        }
        guard randomUnit >= 0, randomUnit <= 1 else {
            throw HibikiTokenSamplingError.invalidRandomValue(randomUnit)
        }

        let ranked = logits.enumerated().sorted { lhs, rhs in
            if lhs.element == rhs.element {
                return lhs.offset < rhs.offset
            }
            return lhs.element > rhs.element
        }
        let candidateCount = topK > 0 ? min(topK, ranked.count) : ranked.count
        let candidates = Array(ranked.prefix(candidateCount))
        let candidateTokens = candidates.map(\.offset)

        guard temperature > 0 else {
            return HibikiSampledToken(token: candidates[0].offset, candidateTokens: candidateTokens)
        }

        let scaled = candidates.map { Double($0.element) / temperature }
        let maxScaled = scaled.max() ?? 0
        let weights = scaled.map { exp($0 - maxScaled) }
        let totalWeight = weights.reduce(0, +)
        let threshold = randomUnit * totalWeight
        var cumulative = 0.0
        for (index, weight) in weights.enumerated() {
            cumulative += weight
            if threshold <= cumulative {
                return HibikiSampledToken(token: candidates[index].offset, candidateTokens: candidateTokens)
            }
        }

        return HibikiSampledToken(token: candidates[candidates.count - 1].offset, candidateTokens: candidateTokens)
    }
}

public struct HibikiInferenceStep: Equatable, Sendable {
    public var frameIndex: Int
    public var sourceAudioTokens: MimiTokenFrame
    public var text: HibikiTextOutput
    public var generatedAudioTokens: MimiTokenFrame

    nonisolated public init(
        frameIndex: Int,
        sourceAudioTokens: MimiTokenFrame,
        text: HibikiTextOutput,
        generatedAudioTokens: MimiTokenFrame
    ) {
        self.frameIndex = frameIndex
        self.sourceAudioTokens = sourceAudioTokens
        self.text = text
        self.generatedAudioTokens = generatedAudioTokens
    }

    public var referenceTraceEvent: ReferenceTraceEvent {
        ReferenceTraceEvent(
            stream: .model,
            name: "hibikiStep",
            frameIndex: frameIndex,
            shape: [1, sourceAudioTokens.codebookCount + 1],
            tokens: [text.token] + Array(generatedAudioTokens.tokens.prefix(2)),
            cadenceMilliseconds: 80
        )
    }
}

public enum HibikiInferenceEvent: Equatable, Sendable {
    case streamStarted(HibikiInferenceDescription)
    case sourceTokens(MimiTokenFrame)
    case step(HibikiInferenceStep)
    case text(HibikiTextOutput)
    case generatedAudio(MimiTokenFrame)
    case streamStopped
    case streamFailed(String)

    var name: String {
        switch self {
        case .streamStarted:
            "streamStarted"
        case .sourceTokens:
            "sourceAudioTokens"
        case .step:
            "hibikiStep"
        case .text:
            "textToken"
        case .generatedAudio:
            "generatedAudioTokens"
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
        case .sourceTokens, .step, .text, .generatedAudio:
            false
        }
    }
}

public enum HibikiInferenceError: Error, Equatable, Sendable {
    case unavailable(String)
    case notInitialized
    case invalidArtifacts(String)
    case unsupportedCodebookCount(Int)

    public var userVisibleMessage: String {
        switch self {
        case let .unavailable(message):
            "Hibiki inference unavailable: \(message)"
        case .notInitialized:
            "Hibiki inference not initialized"
        case let .invalidArtifacts(message):
            "Hibiki inference artifacts invalid: \(message)"
        case let .unsupportedCodebookCount(count):
            "Hibiki inference codebook count unsupported: \(count)"
        }
    }
}

public protocol HibikiInferenceSession: Sendable {
    func initialize(
        artifacts: PreparedModelArtifacts,
        configuration: HibikiGenerationConfiguration
    ) async throws -> HibikiInferenceDescription
    func step(sourceAudioTokens: MimiTokenFrame) async throws -> HibikiInferenceStep
    func reset()
}

public final class DeterministicHibikiInferenceSession: HibikiInferenceSession, @unchecked Sendable {
    private let visibleTextPieces: [String]
    private let state = Mutex(DeterministicHibikiInferenceState())

    public init(visibleTextPieces: [String] = [" hello", " world"]) {
        self.visibleTextPieces = visibleTextPieces
    }

    public func initialize(
        artifacts: PreparedModelArtifacts,
        configuration: HibikiGenerationConfiguration
    ) async throws -> HibikiInferenceDescription {
        let roles = Set(artifacts.files.map(\.role))
        let requiredRoles: Set<String> = ["architectureConfig", "hibikiWeights", "mimiWeights", "tokenizer"]
        guard requiredRoles.isSubset(of: roles) else {
            throw HibikiInferenceError.invalidArtifacts("missing required model roles")
        }

        state.withLock { state in
            state.initialized = true
            state.nextFrameIndex = 0
        }

        return HibikiInferenceDescription(
            modelRevision: artifacts.manifest.revision,
            artifactCount: artifacts.files.count,
            configuration: configuration
        )
    }

    public func step(sourceAudioTokens: MimiTokenFrame) async throws -> HibikiInferenceStep {
        guard sourceAudioTokens.codebookCount > 0 else {
            throw HibikiInferenceError.unsupportedCodebookCount(sourceAudioTokens.codebookCount)
        }

        return try state.withLock { state in
            guard state.initialized else {
                throw HibikiInferenceError.notInitialized
            }

            let frameIndex = state.nextFrameIndex
            let textOutput = makeTextOutput(frameIndex: frameIndex)
            let tokenOffset = frameIndex * sourceAudioTokens.codebookCount
            let audioTokens = (0..<sourceAudioTokens.codebookCount).map { 501 + tokenOffset + $0 }
            let generatedAudio = MimiTokenFrame(
                frameIndex: frameIndex,
                timestampMilliseconds: sourceAudioTokens.timestampMilliseconds,
                codebookCount: sourceAudioTokens.codebookCount,
                tokens: audioTokens,
                sourceAudioFrameIndex: sourceAudioTokens.sourceAudioFrameIndex
            )
            state.nextFrameIndex += 1

            return HibikiInferenceStep(
                frameIndex: frameIndex,
                sourceAudioTokens: sourceAudioTokens,
                text: textOutput,
                generatedAudioTokens: generatedAudio
            )
        }
    }

    public func reset() {
        state.withLock { state in
            state.initialized = false
            state.nextFrameIndex = 0
        }
    }

    private func makeTextOutput(frameIndex: Int) -> HibikiTextOutput {
        guard frameIndex > 0 else {
            return HibikiTextOutput(frameIndex: frameIndex, token: 0)
        }

        let pieceIndex = frameIndex - 1
        if visibleTextPieces.indices.contains(pieceIndex) {
            return HibikiTextOutput(
                frameIndex: frameIndex,
                token: 501 + pieceIndex,
                piece: visibleTextPieces[pieceIndex]
            )
        }

        return HibikiTextOutput(frameIndex: frameIndex, token: 3)
    }
}

private struct DeterministicHibikiInferenceState: Sendable {
    var initialized = false
    var nextFrameIndex = 0
}

public struct FailingHibikiInferenceSession: HibikiInferenceSession, Sendable {
    private let error: HibikiInferenceError

    public init(error: HibikiInferenceError) {
        self.error = error
    }

    public func initialize(
        artifacts: PreparedModelArtifacts,
        configuration: HibikiGenerationConfiguration
    ) async throws -> HibikiInferenceDescription {
        throw error
    }

    public func step(sourceAudioTokens: MimiTokenFrame) async throws -> HibikiInferenceStep {
        throw error
    }

    public func reset() {}
}

public struct HibikiTranslationExperimentBackend: ExperimentBackend, Sendable {
    private let artifactPreparer: ModelArtifactPreparer
    private let source: any AudioInputSource
    private let encoder: any MimiStreamingEncoder
    private let inferenceSession: any HibikiInferenceSession
    private let decoder: any MimiStreamingDecoder
    private let playbackSink: any PlaybackSink
    private let generationConfiguration: HibikiGenerationConfiguration

    public init(
        artifactPreparer: ModelArtifactPreparer,
        audioSource: any AudioInputSource,
        mimiEncoder: any MimiStreamingEncoder,
        inferenceSession: any HibikiInferenceSession,
        mimiDecoder: any MimiStreamingDecoder,
        playbackSink: any PlaybackSink,
        generationConfiguration: HibikiGenerationConfiguration = HibikiGenerationConfiguration()
    ) {
        self.artifactPreparer = artifactPreparer
        self.source = audioSource
        self.encoder = mimiEncoder
        self.inferenceSession = inferenceSession
        self.decoder = mimiDecoder
        self.playbackSink = playbackSink
        self.generationConfiguration = generationConfiguration
    }

    public func prepareEvents() async -> [ExperimentEvent] {
        let result = await artifactPreparer.prepare()
        var events = artifactEvents(from: result)

        events.append(contentsOf: await terminalPrepareEvents(from: result))
        return events
    }

    public func prepareEvents(send: @escaping @MainActor (ExperimentEvent) -> Void) async {
        hibikiBackendLogger.info("deterministic prepare artifact begin")
        let result = await artifactPreparer.prepare { artifactProgress in
            await MainActor.run {
                send(.artifactPreparationProgress(artifactProgress))
                send(.preparationProgress(artifactProgress.overallFractionCompleted))
            }
        }
        hibikiBackendLogger.info("deterministic prepare artifact end files=\(result.artifacts?.files.count ?? 0, privacy: .public) failure=\(result.failure == nil ? "none" : "present", privacy: .public)")

        for event in await terminalPrepareEvents(from: result) {
            send(event)
        }
    }

    private func artifactEvents(from result: ModelArtifactPreparationResult) -> [ExperimentEvent] {
        let events = result.artifactProgressEvents.flatMap { artifactProgress in
            [
                ExperimentEvent.artifactPreparationProgress(artifactProgress),
                ExperimentEvent.preparationProgress(artifactProgress.overallFractionCompleted),
            ]
        }
        if events.isEmpty {
            return result.progressEvents.map(ExperimentEvent.preparationProgress)
        }
        return events
    }

    private func terminalPrepareEvents(from result: ModelArtifactPreparationResult) async -> [ExperimentEvent] {
        if let failure = result.failure {
            hibikiBackendLogger.error("deterministic prepare artifact failed: \(failure.userVisibleMessage, privacy: .public)")
            return [.failure(failure.userVisibleMessage)]
        }

        guard let artifacts = result.artifacts else {
            hibikiBackendLogger.error("deterministic prepare missing prepared artifacts")
            return [.failure(HibikiInferenceError.invalidArtifacts("missing prepared artifacts").userVisibleMessage)]
        }

        do {
            hibikiBackendLogger.info("deterministic hibiki initialize begin")
            let description = try await inferenceSession.initialize(
                artifacts: artifacts,
                configuration: generationConfiguration
            )
            hibikiBackendLogger.info("deterministic hibiki initialize end revision=\(description.modelRevision, privacy: .public)")
            return [
                .hibikiInference(.streamStarted(description)),
                .ready,
            ]
        } catch let error as HibikiInferenceError {
            hibikiBackendLogger.error("deterministic hibiki initialize failed: \(error.userVisibleMessage, privacy: .public)")
            return [
                .hibikiInference(.streamFailed(error.userVisibleMessage)),
                .failure(error.userVisibleMessage),
            ]
        } catch {
            let message = HibikiInferenceError.unavailable(String(describing: error)).userVisibleMessage
            hibikiBackendLogger.error("deterministic hibiki initialize failed: \(message, privacy: .public)")
            return [
                .hibikiInference(.streamFailed(message)),
                .failure(message),
            ]
        }
    }

    public func runEvents() async -> [ExperimentEvent] {
        hibikiBackendLogger.info("deterministic run begin")
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

            var nextAudioFrameIndex = 0
            for chunk in try await source.chunks() {
                events.append(.audioInput(.chunk(chunk)))
                nextAudioFrameIndex = max(nextAudioFrameIndex, chunk.frameIndex + 1)
                for sourceTokens in try await encoder.encode(chunk) {
                    _ = try await appendHibikiGeneratedFrameEvents(
                        sourceTokens: sourceTokens,
                        inferenceSession: inferenceSession,
                        decoder: decoder,
                        playbackSink: playbackSink,
                        events: &events
                    )
                }
            }

            events.append(.audioInput(.streamStopped))
            try await appendHibikiTailFlushEvents(
                encoderDescription: encoderDescription,
                generationConfiguration: generationConfiguration,
                startAudioFrameIndex: nextAudioFrameIndex,
                encoder: encoder,
                inferenceSession: inferenceSession,
                decoder: decoder,
                playbackSink: playbackSink,
                events: &events
            )
            events.append(.mimiEncode(.streamStopped))
            events.append(.hibikiInference(.streamStopped))
            events.append(.mimiDecode(.streamStopped))
            events.append(.playback(.streamStopped))
            hibikiBackendLogger.info("deterministic run end events=\(events.count, privacy: .public)")
        } catch let error as AudioInputError {
            hibikiBackendLogger.error("deterministic audio input failed: \(error.userVisibleMessage, privacy: .public)")
            events.append(.audioInput(.streamFailed(error.userVisibleMessage)))
            events.append(.failure(error.userVisibleMessage))
        } catch let error as MimiEncodeError {
            hibikiBackendLogger.error("deterministic mimi encode failed: \(error.userVisibleMessage, privacy: .public)")
            events.append(.mimiEncode(.streamFailed(error.userVisibleMessage)))
            events.append(.failure(error.userVisibleMessage))
        } catch let error as HibikiInferenceError {
            hibikiBackendLogger.error("deterministic hibiki step failed: \(error.userVisibleMessage, privacy: .public)")
            events.append(.hibikiInference(.streamFailed(error.userVisibleMessage)))
            events.append(.failure(error.userVisibleMessage))
        } catch let error as MimiDecodeError {
            hibikiBackendLogger.error("deterministic mimi decode failed: \(error.userVisibleMessage, privacy: .public)")
            events.append(.mimiDecode(.streamFailed(error.userVisibleMessage)))
            events.append(.failure(error.userVisibleMessage))
        } catch let error as PlaybackSinkError {
            hibikiBackendLogger.error("deterministic playback failed: \(error.userVisibleMessage, privacy: .public)")
            events.append(.playback(.streamFailed(error.userVisibleMessage)))
            events.append(.failure(error.userVisibleMessage))
        } catch {
            let message = HibikiInferenceError.unavailable(String(describing: error)).userVisibleMessage
            hibikiBackendLogger.error("deterministic run failed: \(message, privacy: .public)")
            events.append(.hibikiInference(.streamFailed(message)))
            events.append(.failure(message))
        }

        return events
    }

    public func stop() {
        source.stop()
        encoder.reset()
        inferenceSession.reset()
        decoder.reset()
        playbackSink.stop()
    }
}

public struct RealFileHibikiTranslationExperimentBackend: ExperimentBackend, Sendable {
    public typealias MimiRuntimeLoader = @Sendable (PreparedModelArtifacts) throws -> MLXMimiRuntime

    private let artifactPreparer: ModelArtifactPreparer
    private let source: any AudioInputSource
    private let inferenceSession: any HibikiInferenceSession
    private let playbackSink: any PlaybackSink
    private let generationConfiguration: HibikiGenerationConfiguration
    private let mimiRuntimeLoader: MimiRuntimeLoader
    private let components = RealFileHibikiTranslationComponentStore()

    public init(
        artifactPreparer: ModelArtifactPreparer,
        audioSource: any AudioInputSource,
        inferenceSession: any HibikiInferenceSession = MLXHibikiInferenceSession(),
        playbackSink: any PlaybackSink,
        generationConfiguration: HibikiGenerationConfiguration = HibikiGenerationConfiguration(),
        mimiRuntimeLoader: @escaping MimiRuntimeLoader = { artifacts in
            try MLXMimiRuntimeLoader().load(from: artifacts)
        }
    ) {
        self.artifactPreparer = artifactPreparer
        self.source = audioSource
        self.inferenceSession = inferenceSession
        self.playbackSink = playbackSink
        self.generationConfiguration = generationConfiguration
        self.mimiRuntimeLoader = mimiRuntimeLoader
    }

    public func prepareEvents() async -> [ExperimentEvent] {
        let result = await artifactPreparer.prepare()
        var events = artifactEvents(from: result)
        events.append(contentsOf: await terminalPrepareEvents(from: result))
        return events
    }

    public func prepareEvents(send: @escaping @MainActor (ExperimentEvent) -> Void) async {
        hibikiBackendLogger.info("real-file prepare artifact begin")
        let result = await artifactPreparer.prepare { artifactProgress in
            await MainActor.run {
                send(.artifactPreparationProgress(artifactProgress))
                send(.preparationProgress(artifactProgress.overallFractionCompleted))
            }
        }
        hibikiBackendLogger.info("real-file prepare artifact end files=\(result.artifacts?.files.count ?? 0, privacy: .public) failure=\(result.failure == nil ? "none" : "present", privacy: .public)")

        for event in await terminalPrepareEvents(from: result) {
            send(event)
        }
    }

    private func artifactEvents(from result: ModelArtifactPreparationResult) -> [ExperimentEvent] {
        let events = result.artifactProgressEvents.flatMap { artifactProgress in
            [
                ExperimentEvent.artifactPreparationProgress(artifactProgress),
                ExperimentEvent.preparationProgress(artifactProgress.overallFractionCompleted),
            ]
        }
        if events.isEmpty {
            return result.progressEvents.map(ExperimentEvent.preparationProgress)
        }
        return events
    }

    private func terminalPrepareEvents(from result: ModelArtifactPreparationResult) async -> [ExperimentEvent] {
        if let failure = result.failure {
            hibikiBackendLogger.error("real-file prepare artifact failed: \(failure.userVisibleMessage, privacy: .public)")
            return [.failure(failure.userVisibleMessage)]
        }

        guard let artifacts = result.artifacts else {
            hibikiBackendLogger.error("real-file prepare missing prepared artifacts")
            return [.failure(HibikiInferenceError.invalidArtifacts("missing prepared artifacts").userVisibleMessage)]
        }

        do {
            hibikiBackendLogger.info("real-file mimi runtime load begin")
            let runtime = try mimiRuntimeLoader(artifacts)
            hibikiBackendLogger.info("real-file mimi runtime load end")
            let encoder = MLXMimiStreamingEncoder(runtime: runtime)
            let decoder = MLXMimiStreamingDecoder(runtime: runtime)
            hibikiBackendLogger.info("real-file hibiki initialize begin")
            let description = try await inferenceSession.initialize(
                artifacts: artifacts,
                configuration: generationConfiguration
            )
            hibikiBackendLogger.info("real-file hibiki initialize end revision=\(description.modelRevision, privacy: .public)")
            components.store(
                RealFileHibikiTranslationComponents(
                    encoder: encoder,
                    decoder: decoder
                )
            )
            return [
                .hibikiInference(.streamStarted(description)),
                .ready,
            ]
        } catch let error as MimiRuntimeError {
            hibikiBackendLogger.error("real-file mimi runtime failed: \(error.userVisibleMessage, privacy: .public)")
            return [
                .mimiEncode(.streamFailed(error.userVisibleMessage)),
                .mimiDecode(.streamFailed(error.userVisibleMessage)),
                .failure(error.userVisibleMessage),
            ]
        } catch let error as MimiEncodeError {
            hibikiBackendLogger.error("real-file mimi encode prepare failed: \(error.userVisibleMessage, privacy: .public)")
            return [
                .mimiEncode(.streamFailed(error.userVisibleMessage)),
                .failure(error.userVisibleMessage),
            ]
        } catch let error as HibikiInferenceError {
            hibikiBackendLogger.error("real-file hibiki initialize failed: \(error.userVisibleMessage, privacy: .public)")
            return [
                .hibikiInference(.streamFailed(error.userVisibleMessage)),
                .failure(error.userVisibleMessage),
            ]
        } catch {
            let message = HibikiInferenceError.unavailable(String(describing: error)).userVisibleMessage
            hibikiBackendLogger.error("real-file prepare failed: \(message, privacy: .public)")
            return [
                .hibikiInference(.streamFailed(message)),
                .failure(message),
            ]
        }
    }

    public func runEvents() async -> [ExperimentEvent] {
        guard let prepared = components.load() else {
            hibikiBackendLogger.error("real-file run missing prepared components")
            return [.failure(HibikiInferenceError.notInitialized.userVisibleMessage)]
        }

        hibikiBackendLogger.info("real-file run begin")
        let encoder = prepared.encoder
        let decoder = prepared.decoder
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

            var nextAudioFrameIndex = 0
            for chunk in try await source.chunks() {
                events.append(.audioInput(.chunk(chunk)))
                nextAudioFrameIndex = max(nextAudioFrameIndex, chunk.frameIndex + 1)
                for sourceTokens in try await encoder.encode(chunk) {
                    _ = try await appendHibikiGeneratedFrameEvents(
                        sourceTokens: sourceTokens,
                        inferenceSession: inferenceSession,
                        decoder: decoder,
                        playbackSink: playbackSink,
                        events: &events
                    )
                }
            }

            events.append(.audioInput(.streamStopped))
            try await appendHibikiTailFlushEvents(
                encoderDescription: encoderDescription,
                generationConfiguration: generationConfiguration,
                startAudioFrameIndex: nextAudioFrameIndex,
                encoder: encoder,
                inferenceSession: inferenceSession,
                decoder: decoder,
                playbackSink: playbackSink,
                events: &events
            )
            events.append(.mimiEncode(.streamStopped))
            events.append(.hibikiInference(.streamStopped))
            events.append(.mimiDecode(.streamStopped))
            events.append(.playback(.streamStopped))
            hibikiBackendLogger.info("real-file run end events=\(events.count, privacy: .public)")
        } catch let error as AudioInputError {
            hibikiBackendLogger.error("real-file audio input failed: \(error.userVisibleMessage, privacy: .public)")
            events.append(.audioInput(.streamFailed(error.userVisibleMessage)))
            events.append(.failure(error.userVisibleMessage))
        } catch let error as MimiEncodeError {
            hibikiBackendLogger.error("real-file mimi encode failed: \(error.userVisibleMessage, privacy: .public)")
            events.append(.mimiEncode(.streamFailed(error.userVisibleMessage)))
            events.append(.failure(error.userVisibleMessage))
        } catch let error as HibikiInferenceError {
            hibikiBackendLogger.error("real-file hibiki step failed: \(error.userVisibleMessage, privacy: .public)")
            events.append(.hibikiInference(.streamFailed(error.userVisibleMessage)))
            events.append(.failure(error.userVisibleMessage))
        } catch let error as MimiDecodeError {
            hibikiBackendLogger.error("real-file mimi decode failed: \(error.userVisibleMessage, privacy: .public)")
            events.append(.mimiDecode(.streamFailed(error.userVisibleMessage)))
            events.append(.failure(error.userVisibleMessage))
        } catch let error as PlaybackSinkError {
            hibikiBackendLogger.error("real-file playback failed: \(error.userVisibleMessage, privacy: .public)")
            events.append(.playback(.streamFailed(error.userVisibleMessage)))
            events.append(.failure(error.userVisibleMessage))
        } catch {
            let message = HibikiInferenceError.unavailable(String(describing: error)).userVisibleMessage
            hibikiBackendLogger.error("real-file run failed: \(message, privacy: .public)")
            events.append(.hibikiInference(.streamFailed(message)))
            events.append(.failure(message))
        }

        return events
    }

    public func stop() {
        source.stop()
        inferenceSession.reset()
        components.reset()
        playbackSink.stop()
    }
}

private final class RealFileHibikiTranslationComponentStore: @unchecked Sendable {
    private let components = Mutex<RealFileHibikiTranslationComponents?>(nil)

    func store(_ newComponents: RealFileHibikiTranslationComponents) {
        components.withLock { components in
            components = newComponents
        }
    }

    func load() -> RealFileHibikiTranslationComponents? {
        components.withLock { $0 }
    }

    func reset() {
        components.withLock { components in
            components?.encoder.reset()
            components?.decoder.reset()
            components = nil
        }
    }
}

private struct RealFileHibikiTranslationComponents: Sendable {
    var encoder: any MimiStreamingEncoder
    var decoder: any MimiStreamingDecoder
}

private func appendHibikiGeneratedFrameEvents(
    sourceTokens: MimiTokenFrame,
    inferenceSession: any HibikiInferenceSession,
    decoder: any MimiStreamingDecoder,
    playbackSink: any PlaybackSink,
    events: inout [ExperimentEvent]
) async throws -> HibikiTextOutput {
    events.append(.mimiEncode(.frame(sourceTokens)))
    events.append(.hibikiInference(.sourceTokens(sourceTokens)))

    let step = try await inferenceSession.step(sourceAudioTokens: sourceTokens)
    events.append(.hibikiInference(.step(step)))
    events.append(.hibikiInference(.text(step.text)))
    events.append(.hibikiInference(.generatedAudio(step.generatedAudioTokens)))

    for decoded in try await decoder.decode(step.generatedAudioTokens) {
        events.append(.mimiDecode(.chunk(decoded)))
        try await playbackSink.receive(decoded)
        events.append(.playback(.chunk(decoded)))
    }

    return step.text
}

private func appendHibikiTailFlushEvents(
    encoderDescription: MimiEncoderDescription,
    generationConfiguration: HibikiGenerationConfiguration,
    startAudioFrameIndex: Int,
    encoder: any MimiStreamingEncoder,
    inferenceSession: any HibikiInferenceSession,
    decoder: any MimiStreamingDecoder,
    playbackSink: any PlaybackSink,
    events: inout [ExperimentEvent]
) async throws {
    guard generationConfiguration.tailSilenceFrameCount > 0 else { return }

    var stopDetector = HibikiPostInputStopDetector(
        requiredBlankOrPaddingFrameCount: generationConfiguration.postInputPaddingStopFrameCount
    )
    for tailFrameIndex in 0..<generationConfiguration.tailSilenceFrameCount {
        let chunkFrameIndex = startAudioFrameIndex + tailFrameIndex
        let chunk = PCMChunk(
            frameIndex: chunkFrameIndex,
            timestampMilliseconds: Double(chunkFrameIndex * encoderDescription.samplesPerFrame)
                / Double(encoderDescription.sampleRate) * 1000,
            sampleRate: encoderDescription.sampleRate,
            samples: Array(repeating: 0, count: encoderDescription.samplesPerFrame)
        )
        for sourceTokens in try await encoder.encode(chunk) {
            let text = try await appendHibikiGeneratedFrameEvents(
                sourceTokens: sourceTokens,
                inferenceSession: inferenceSession,
                decoder: decoder,
                playbackSink: playbackSink,
                events: &events
            )
            if stopDetector.shouldStop(after: text) {
                return
            }
        }
    }
}

private struct HibikiPostInputStopDetector {
    private let requiredBlankOrPaddingFrameCount: Int
    private var blankOrPaddingRun = 0

    init(requiredBlankOrPaddingFrameCount: Int) {
        self.requiredBlankOrPaddingFrameCount = requiredBlankOrPaddingFrameCount
    }

    mutating func shouldStop(after text: HibikiTextOutput) -> Bool {
        guard requiredBlankOrPaddingFrameCount > 0 else { return false }
        if text.isBlankOrPadding {
            blankOrPaddingRun += 1
        } else {
            blankOrPaddingRun = 0
        }
        return blankOrPaddingRun >= requiredBlankOrPaddingFrameCount
    }
}

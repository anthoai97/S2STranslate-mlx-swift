import Foundation
import Synchronization

public struct HibikiGenerationConfiguration: Equatable, Sendable {
    public var temperature: Double
    public var textTemperature: Double
    public var topK: Int
    public var textTopK: Int
    public var voiceTransferEnabled: Bool

    nonisolated public init(
        temperature: Double = 0.8,
        textTemperature: Double = 0.8,
        topK: Int = 250,
        textTopK: Int = 250,
        voiceTransferEnabled: Bool = false
    ) {
        self.temperature = temperature
        self.textTemperature = textTemperature
        self.topK = topK
        self.textTopK = textTopK
        self.voiceTransferEnabled = voiceTransferEnabled
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

    public var referenceTraceEvent: ReferenceTraceEvent {
        ReferenceTraceEvent(
            stream: .text,
            name: isVisible ? "textToken" : "skipBlankOrPadding",
            frameIndex: frameIndex,
            tokens: [token]
        )
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
        let result = await artifactPreparer.prepare { artifactProgress in
            await MainActor.run {
                send(.artifactPreparationProgress(artifactProgress))
                send(.preparationProgress(artifactProgress.overallFractionCompleted))
            }
        }

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
            return [.failure(failure.userVisibleMessage)]
        }

        guard let artifacts = result.artifacts else {
            return [.failure(HibikiInferenceError.invalidArtifacts("missing prepared artifacts").userVisibleMessage)]
        }

        do {
            let description = try await inferenceSession.initialize(
                artifacts: artifacts,
                configuration: generationConfiguration
            )
            return [
                .hibikiInference(.streamStarted(description)),
                .ready,
            ]
        } catch let error as HibikiInferenceError {
            return [
                .hibikiInference(.streamFailed(error.userVisibleMessage)),
                .failure(error.userVisibleMessage),
            ]
        } catch {
            let message = HibikiInferenceError.unavailable(String(describing: error)).userVisibleMessage
            return [
                .hibikiInference(.streamFailed(message)),
                .failure(message),
            ]
        }
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
                for sourceTokens in try await encoder.encode(chunk) {
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
                }
            }

            events.append(.audioInput(.streamStopped))
            events.append(.mimiEncode(.streamStopped))
            events.append(.hibikiInference(.streamStopped))
            events.append(.mimiDecode(.streamStopped))
            events.append(.playback(.streamStopped))
        } catch let error as AudioInputError {
            events.append(.audioInput(.streamFailed(error.userVisibleMessage)))
            events.append(.failure(error.userVisibleMessage))
        } catch let error as MimiEncodeError {
            events.append(.mimiEncode(.streamFailed(error.userVisibleMessage)))
            events.append(.failure(error.userVisibleMessage))
        } catch let error as HibikiInferenceError {
            events.append(.hibikiInference(.streamFailed(error.userVisibleMessage)))
            events.append(.failure(error.userVisibleMessage))
        } catch let error as MimiDecodeError {
            events.append(.mimiDecode(.streamFailed(error.userVisibleMessage)))
            events.append(.failure(error.userVisibleMessage))
        } catch let error as PlaybackSinkError {
            events.append(.playback(.streamFailed(error.userVisibleMessage)))
            events.append(.failure(error.userVisibleMessage))
        } catch {
            let message = HibikiInferenceError.unavailable(String(describing: error)).userVisibleMessage
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
        let result = await artifactPreparer.prepare { artifactProgress in
            await MainActor.run {
                send(.artifactPreparationProgress(artifactProgress))
                send(.preparationProgress(artifactProgress.overallFractionCompleted))
            }
        }

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
            return [.failure(failure.userVisibleMessage)]
        }

        guard let artifacts = result.artifacts else {
            return [.failure(HibikiInferenceError.invalidArtifacts("missing prepared artifacts").userVisibleMessage)]
        }

        do {
            let runtime = try mimiRuntimeLoader(artifacts)
            let encoder = MLXMimiStreamingEncoder(runtime: runtime)
            let decoder = MLXMimiStreamingDecoder(runtime: runtime)
            let description = try await inferenceSession.initialize(
                artifacts: artifacts,
                configuration: generationConfiguration
            )
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
            return [.failure(error.userVisibleMessage)]
        } catch let error as MimiEncodeError {
            return [.failure(error.userVisibleMessage)]
        } catch let error as HibikiInferenceError {
            return [
                .hibikiInference(.streamFailed(error.userVisibleMessage)),
                .failure(error.userVisibleMessage),
            ]
        } catch {
            let message = HibikiInferenceError.unavailable(String(describing: error)).userVisibleMessage
            return [
                .hibikiInference(.streamFailed(message)),
                .failure(message),
            ]
        }
    }

    public func runEvents() async -> [ExperimentEvent] {
        guard let prepared = components.load() else {
            return [.failure(HibikiInferenceError.notInitialized.userVisibleMessage)]
        }

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

            for chunk in try await source.chunks() {
                events.append(.audioInput(.chunk(chunk)))
                for sourceTokens in try await encoder.encode(chunk) {
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
                }
            }

            events.append(.audioInput(.streamStopped))
            events.append(.mimiEncode(.streamStopped))
            events.append(.hibikiInference(.streamStopped))
            events.append(.mimiDecode(.streamStopped))
            events.append(.playback(.streamStopped))
        } catch let error as AudioInputError {
            events.append(.audioInput(.streamFailed(error.userVisibleMessage)))
            events.append(.failure(error.userVisibleMessage))
        } catch let error as MimiEncodeError {
            events.append(.mimiEncode(.streamFailed(error.userVisibleMessage)))
            events.append(.failure(error.userVisibleMessage))
        } catch let error as HibikiInferenceError {
            events.append(.hibikiInference(.streamFailed(error.userVisibleMessage)))
            events.append(.failure(error.userVisibleMessage))
        } catch let error as MimiDecodeError {
            events.append(.mimiDecode(.streamFailed(error.userVisibleMessage)))
            events.append(.failure(error.userVisibleMessage))
        } catch let error as PlaybackSinkError {
            events.append(.playback(.streamFailed(error.userVisibleMessage)))
            events.append(.failure(error.userVisibleMessage))
        } catch {
            let message = HibikiInferenceError.unavailable(String(describing: error)).userVisibleMessage
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

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

public struct MLXMimiRuntimeConfiguration: Equatable, Sendable {
    public var sampleRate: Int
    public var frameRate: Double
    public var codebookCount: Int
    public var quantizerBins: Int
    public var samplesPerFrame: Int

    nonisolated public init(
        sampleRate: Int,
        frameRate: Double,
        codebookCount: Int,
        samplesPerFrame: Int,
        quantizerBins: Int = 2_048
    ) {
        self.sampleRate = sampleRate
        self.frameRate = frameRate
        self.codebookCount = codebookCount
        self.quantizerBins = quantizerBins
        self.samplesPerFrame = samplesPerFrame
    }

    public static let mimi202407 = MLXMimiRuntimeConfiguration(
        sampleRate: 24_000,
        frameRate: 12.5,
        codebookCount: 16,
        samplesPerFrame: 1_920,
        quantizerBins: 2_048
    )

    public static func mimi202407(codebookCount: Int) -> MLXMimiRuntimeConfiguration {
        MLXMimiRuntimeConfiguration(
            sampleRate: 24_000,
            frameRate: 12.5,
            codebookCount: codebookCount,
            samplesPerFrame: 1_920,
            quantizerBins: 2_048
        )
    }
}

public enum MimiRuntimeError: Error, Equatable, Sendable {
    case missingArtifactRole(String)
    case missingArtifactFile(String)
    case incompatibleConfiguration(String)
    case loadFailed(String)
    case warmupFailed(String)

    public var userVisibleMessage: String {
        switch self {
        case let .missingArtifactRole(role):
            "Mimi runtime artifact role missing: \(role)"
        case let .missingArtifactFile(fileName):
            "Mimi runtime artifact file missing: \(fileName)"
        case let .incompatibleConfiguration(message):
            "Mimi runtime configuration incompatible: \(message)"
        case let .loadFailed(message):
            "Mimi runtime load failed: \(message)"
        case let .warmupFailed(message):
            "Mimi runtime warmup failed: \(message)"
        }
    }
}

public struct MLXMimiWarmupRequest: Equatable, Sendable {
    public var pcmShape: [Int]
    public var sampleCount: Int
    public var frameCount: Int

    nonisolated public init(pcmShape: [Int], sampleCount: Int, frameCount: Int) {
        self.pcmShape = pcmShape
        self.sampleCount = sampleCount
        self.frameCount = frameCount
    }
}

public struct MLXMimiPCMInput: Equatable, Sendable {
    public var samples: [Float]
    public var sampleRate: Int
    public var pcmShape: [Int]

    nonisolated public init(samples: [Float], sampleRate: Int, pcmShape: [Int]) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.pcmShape = pcmShape
    }
}

public struct MLXMimiEncodedFrame: Equatable, Sendable {
    public var tokens: [Int]

    nonisolated public init(tokens: [Int]) {
        self.tokens = tokens
    }
}

public struct MLXMimiTokenInput: Equatable, Sendable {
    public var tokens: [Int]
    public var codebookCount: Int

    nonisolated public init(tokens: [Int], codebookCount: Int) {
        self.tokens = tokens
        self.codebookCount = codebookCount
    }
}

public struct MLXMimiDecodedChunk: Equatable, Sendable {
    public var samples: [Float]

    nonisolated public init(samples: [Float]) {
        self.samples = samples
    }
}

public protocol MLXMimiRuntimeEngine: AnyObject {
    func resetEncodeState()
    func resetDecodeState()
    func warmup(request: MLXMimiWarmupRequest) throws
    func encode(_ input: MLXMimiPCMInput) throws -> [MLXMimiEncodedFrame]
    func decode(_ input: MLXMimiTokenInput) throws -> [MLXMimiDecodedChunk]
}

public final class MLXMimiDefaultRuntimeEngine: MLXMimiRuntimeEngine {
    private let model: MLXMimiModel
    private let inputBuilder: (MLXMimiPCMInput) throws -> MLXMimiStreamArray
    private let encodeGraphStep: (MLXMimiStreamArray) throws -> MLXMimiStreamArray
    private let tokenExtractor: (MLXMimiStreamArray) throws -> [MLXMimiEncodedFrame]
    private let decodeInputBuilder: (MLXMimiTokenInput) throws -> MLXMimiStreamArray
    private let decodeGraphStep: (MLXMimiStreamArray) throws -> MLXMimiStreamArray
    private let decodedChunkExtractor: (MLXMimiStreamArray) throws -> [MLXMimiDecodedChunk]

    public init(model: MLXMimiModel = MLXMimiModel()) {
        self.model = model
        self.inputBuilder = { input in
            MLXMimiStreamArray(MLXArray(input.samples)[.newAxis, .newAxis])
        }
        self.encodeGraphStep = { [model] stream in
            var stream = model.encoder.step(stream)
            stream = model.encoderTransformer.step(stream, cache: model.encoderTransformerCache)
            stream = model.downsample.step(stream)
            return model.quantizer.encode(stream)
        }
        self.tokenExtractor = { [model] stream in
            try MLXMimiTokenFrameExtractor(
                codebookCount: model.configuration.quantizerCodebookCount
            ).frames(from: stream)
        }
        self.decodeInputBuilder = { input in
            try MLXMimiTokenInputBuilder().stream(from: input)
        }
        self.decodeGraphStep = { [model] codes in
            var stream = codes.map { model.quantizer.decode($0) }
            stream = model.upsample.step(stream)
            stream = model.decoderTransformer.step(stream, cache: model.decoderTransformerCache)
            return model.decoder.step(stream)
        }
        self.decodedChunkExtractor = { stream in
            try MLXMimiDecodedChunkExtractor().chunks(from: stream)
        }
    }

    init(
        model: MLXMimiModel = MLXMimiModel(),
        inputBuilder: @escaping (MLXMimiPCMInput) throws -> MLXMimiStreamArray,
        encodeGraphStep: @escaping (MLXMimiStreamArray) throws -> MLXMimiStreamArray,
        tokenExtractor: @escaping (MLXMimiStreamArray) throws -> [MLXMimiEncodedFrame],
        decodeInputBuilder: @escaping (MLXMimiTokenInput) throws -> MLXMimiStreamArray = { _ in MLXMimiStreamArray() },
        decodeGraphStep: @escaping (MLXMimiStreamArray) throws -> MLXMimiStreamArray = { _ in MLXMimiStreamArray() },
        decodedChunkExtractor: @escaping (MLXMimiStreamArray) throws -> [MLXMimiDecodedChunk] = { _ in [] }
    ) {
        self.model = model
        self.inputBuilder = inputBuilder
        self.encodeGraphStep = encodeGraphStep
        self.tokenExtractor = tokenExtractor
        self.decodeInputBuilder = decodeInputBuilder
        self.decodeGraphStep = decodeGraphStep
        self.decodedChunkExtractor = decodedChunkExtractor
    }

    public func resetEncodeState() {
        model.resetEncodeState()
    }

    public func resetDecodeState() {
        model.resetDecodeState()
    }

    public func warmup(request: MLXMimiWarmupRequest) throws {
        _ = MLXArray.zeros(request.pcmShape, type: Float32.self)
        model.resetState()
    }

    public func encode(_ input: MLXMimiPCMInput) throws -> [MLXMimiEncodedFrame] {
        let stream = try inputBuilder(input)
        let codes = try encodeGraphStep(stream)
        return try tokenExtractor(codes)
    }

    public func decode(_ input: MLXMimiTokenInput) throws -> [MLXMimiDecodedChunk] {
        let codes = try decodeInputBuilder(input)
        let pcm = try decodeGraphStep(codes)
        return try decodedChunkExtractor(pcm)
    }
}

struct MLXMimiTokenInputBuilder {
    func stream(from input: MLXMimiTokenInput) throws -> MLXMimiStreamArray {
        guard input.codebookCount > 0 else {
            throw MimiRuntimeError.loadFailed("Mimi token input codebooks malformed: expected positive count")
        }
        guard input.tokens.count == input.codebookCount else {
            throw MimiRuntimeError.loadFailed(
                "Mimi token input malformed: expected \(input.codebookCount) tokens, got \(input.tokens.count)"
            )
        }

        let tokens = input.tokens.map(Int32.init)
        return MLXMimiStreamArray(MLXArray(tokens).reshaped([1, input.codebookCount, 1]))
    }
}

struct MLXMimiTokenFrameExtractor {
    var codebookCount: Int

    func frames(from stream: MLXMimiStreamArray) throws -> [MLXMimiEncodedFrame] {
        guard let array = stream.asArray() else { return [] }

        let tokens = array.asArray(Int32.self).map(Int.init)
        return try frames(flattenedTokens: tokens, shape: array.shape)
    }

    func frames(flattenedTokens: [Int], shape: [Int]) throws -> [MLXMimiEncodedFrame] {
        guard shape.count == 3 else {
            throw MimiRuntimeError.loadFailed(
                "Mimi token output shape malformed: expected [batch, codebook, time], got \(shape)"
            )
        }
        guard shape[0] == 1 else {
            throw MimiRuntimeError.loadFailed(
                "Mimi token output batch unsupported: expected 1, got \(shape[0])"
            )
        }
        guard shape[1] == codebookCount else {
            throw MimiRuntimeError.loadFailed(
                "Mimi token output codebooks malformed: expected \(codebookCount), got \(shape[1])"
            )
        }

        let frameCount = shape[2]
        guard flattenedTokens.count == shape.reduce(1, *) else {
            throw MimiRuntimeError.loadFailed(
                "Mimi token output buffer malformed: expected \(shape.reduce(1, *)) tokens, got \(flattenedTokens.count)"
            )
        }
        guard frameCount > 0 else { return [] }

        return (0..<frameCount).map { frameIndex in
            let tokens = (0..<codebookCount).map { codebookIndex in
                flattenedTokens[codebookIndex * frameCount + frameIndex]
            }
            return MLXMimiEncodedFrame(tokens: tokens)
        }
    }
}

struct MLXMimiDecodedChunkExtractor {
    func chunks(from stream: MLXMimiStreamArray) throws -> [MLXMimiDecodedChunk] {
        guard let array = stream.asArray() else { return [] }
        return try chunks(flattenedSamples: array.asArray(Float.self), shape: array.shape)
    }

    func chunks(flattenedSamples: [Float], shape: [Int]) throws -> [MLXMimiDecodedChunk] {
        guard shape.count == 3 else {
            throw MimiRuntimeError.loadFailed(
                "Mimi decoded PCM shape malformed: expected [batch, channel, time], got \(shape)"
            )
        }
        guard shape[0] == 1 else {
            throw MimiRuntimeError.loadFailed(
                "Mimi decoded PCM batch unsupported: expected 1, got \(shape[0])"
            )
        }
        guard shape[1] == 1 else {
            throw MimiRuntimeError.loadFailed(
                "Mimi decoded PCM channel unsupported: expected 1, got \(shape[1])"
            )
        }
        guard flattenedSamples.count == shape.reduce(1, *) else {
            throw MimiRuntimeError.loadFailed(
                "Mimi decoded PCM buffer malformed: expected \(shape.reduce(1, *)) samples, got \(flattenedSamples.count)"
            )
        }
        guard shape[2] > 0 else { return [] }
        return [MLXMimiDecodedChunk(samples: flattenedSamples)]
    }
}

public final class MLXMimiRuntime: @unchecked Sendable {
    public let artifact: PreparedModelArtifact
    public let configuration: MLXMimiRuntimeConfiguration
    private let engine: MLXMimiRuntimeEngine

    public init(
        artifact: PreparedModelArtifact,
        configuration: MLXMimiRuntimeConfiguration = .mimi202407,
        engine: MLXMimiRuntimeEngine? = nil
    ) {
        self.artifact = artifact
        self.configuration = configuration
        self.engine = engine ?? MLXMimiDefaultRuntimeEngine(
            model: MLXMimiModel(
                configuration: .mimi202407(codebookCount: configuration.codebookCount),
                batchSize: 1
            )
        )
    }

    public func validateMetadata() throws {
        guard configuration.sampleRate == 24_000 else {
            throw MimiRuntimeError.incompatibleConfiguration(
                "sampleRate expected 24000, got \(configuration.sampleRate)"
            )
        }
        guard configuration.frameRate == 12.5 else {
            throw MimiRuntimeError.incompatibleConfiguration(
                "frameRate expected 12.5, got \(configuration.frameRate)"
            )
        }
        guard (1...16).contains(configuration.codebookCount) else {
            throw MimiRuntimeError.incompatibleConfiguration(
                "codebookCount expected 1...16, got \(configuration.codebookCount)"
            )
        }
        guard configuration.quantizerBins == 2_048 else {
            throw MimiRuntimeError.incompatibleConfiguration(
                "quantizerBins expected 2048, got \(configuration.quantizerBins)"
            )
        }
        guard configuration.samplesPerFrame == 1_920 else {
            throw MimiRuntimeError.incompatibleConfiguration(
                "samplesPerFrame expected 1920, got \(configuration.samplesPerFrame)"
            )
        }
    }

    public func warmup(frameCount: Int = 4) throws {
        try validateMetadata()
        let sampleCount = configuration.samplesPerFrame * frameCount
        let request = MLXMimiWarmupRequest(
            pcmShape: [1, 1, sampleCount],
            sampleCount: sampleCount,
            frameCount: frameCount
        )

        do {
            try engine.warmup(request: request)
        } catch let error as MimiRuntimeError {
            throw error
        } catch {
            throw MimiRuntimeError.warmupFailed(String(describing: error))
        }
    }

    public func encode(_ input: MLXMimiPCMInput) throws -> [MLXMimiEncodedFrame] {
        try validateMetadata()
        do {
            return try engine.encode(input)
        } catch let error as MimiRuntimeError {
            throw error
        } catch {
            throw MimiRuntimeError.loadFailed(String(describing: error))
        }
    }

    public func decode(_ input: MLXMimiTokenInput) throws -> [MLXMimiDecodedChunk] {
        try validateMetadata()
        do {
            return try engine.decode(input)
        } catch let error as MimiRuntimeError {
            throw error
        } catch {
            throw MimiRuntimeError.loadFailed(String(describing: error))
        }
    }

    public func resetEncodeState() {
        engine.resetEncodeState()
    }

    public func resetDecodeState() {
        engine.resetDecodeState()
    }

    public func resetState() {
        resetEncodeState()
        resetDecodeState()
    }
}

public struct MLXMimiRuntimeLoader {
    private let configuration: MLXMimiRuntimeConfiguration
    private let weightLoader: MLXMimiWeightLoader
    private let graphParameterApplier: MLXMimiGraphParameterApplier

    nonisolated public init(
        configuration: MLXMimiRuntimeConfiguration = .mimi202407,
        weightLoader: MLXMimiWeightLoader = MLXMimiWeightLoader(),
        graphParameterApplier: MLXMimiGraphParameterApplier = MLXMimiGraphParameterApplier()
    ) {
        self.configuration = configuration
        self.weightLoader = weightLoader
        self.graphParameterApplier = graphParameterApplier
    }

    public func load(from artifacts: PreparedModelArtifacts) throws -> MLXMimiRuntime {
        guard let artifact = artifacts.files.first(where: { $0.role == "mimiWeights" }) else {
            throw MimiRuntimeError.missingArtifactRole("mimiWeights")
        }

        guard FileManager.default.fileExists(atPath: artifact.location) else {
            throw MimiRuntimeError.missingArtifactFile(artifact.fileName)
        }

        let model = MLXMimiModel(
            configuration: .mimi202407(codebookCount: configuration.codebookCount),
            batchSize: 1
        )
        do {
            let weights = try weightLoader.load(from: artifacts)
            try graphParameterApplier.apply(weights, to: model)
        } catch let error as MLXMimiWeightLoadError {
            throw MimiRuntimeError.loadFailed(error.userVisibleMessage)
        } catch {
            throw MimiRuntimeError.loadFailed(String(describing: error))
        }

        let runtime = MLXMimiRuntime(
            artifact: artifact,
            configuration: configuration,
            engine: MLXMimiDefaultRuntimeEngine(model: model)
        )
        try runtime.validateMetadata()
        return runtime
    }
}

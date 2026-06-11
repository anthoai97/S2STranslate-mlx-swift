import Foundation
import MLX
import Synchronization

public struct MLXHibikiRuntimeConfiguration: Equatable, Sendable {
    public var hiddenScale: Int
    public var kvRepeat: Int
    public var positionalEmbedding: String
    public var depformerOutputLayerNorm: Bool
    public var quantizationBits: Int
    public var quantizationGroupSize: Int
    public var codebookCount: Int

    nonisolated public init(
        hiddenScale: Int = 6,
        kvRepeat: Int = 2,
        positionalEmbedding: String = "rope_concat",
        depformerOutputLayerNorm: Bool = true,
        quantizationBits: Int = 4,
        quantizationGroupSize: Int = 32,
        codebookCount: Int = 16
    ) {
        self.hiddenScale = hiddenScale
        self.kvRepeat = kvRepeat
        self.positionalEmbedding = positionalEmbedding
        self.depformerOutputLayerNorm = depformerOutputLayerNorm
        self.quantizationBits = quantizationBits
        self.quantizationGroupSize = quantizationGroupSize
        self.codebookCount = codebookCount
    }
}

public struct MLXHibikiWeightsSummary: Equatable, Sendable {
    public var tensorCount: Int
    public var hasDepformerOutputLayerNorms: Bool

    nonisolated public init(tensorCount: Int, hasDepformerOutputLayerNorms: Bool) {
        self.tensorCount = tensorCount
        self.hasDepformerOutputLayerNorms = hasDepformerOutputLayerNorms
    }
}

public struct MLXHibikiLoadRequest: Equatable, Sendable {
    public var configURL: URL
    public var weightsURL: URL
    public var tokenizerURL: URL
    public var artifactCount: Int
    public var modelRevision: String
    public var runtimeConfiguration: MLXHibikiRuntimeConfiguration

    nonisolated public init(
        configURL: URL,
        weightsURL: URL,
        tokenizerURL: URL,
        artifactCount: Int,
        modelRevision: String,
        runtimeConfiguration: MLXHibikiRuntimeConfiguration
    ) {
        self.configURL = configURL
        self.weightsURL = weightsURL
        self.tokenizerURL = tokenizerURL
        self.artifactCount = artifactCount
        self.modelRevision = modelRevision
        self.runtimeConfiguration = runtimeConfiguration
    }
}

public struct MLXHibikiStepOutput: Equatable, Sendable {
    public var textToken: Int
    public var textPiece: String?
    public var textCandidateTokens: [Int]
    public var audioTokens: [Int]

    nonisolated public init(
        textToken: Int,
        textPiece: String? = nil,
        textCandidateTokens: [Int] = [],
        audioTokens: [Int]
    ) {
        self.textToken = textToken
        self.textPiece = textPiece
        self.textCandidateTokens = textCandidateTokens
        self.audioTokens = audioTokens
    }
}

public protocol MLXHibikiRuntimeEngine: AnyObject {
    func load(request: MLXHibikiLoadRequest) throws
    func step(
        sourceAudioTokens: MimiTokenFrame,
        frameIndex: Int,
        configuration: HibikiGenerationConfiguration
    ) throws -> MLXHibikiStepOutput
    func reset()
}

public final class MLXHibikiDefaultRuntimeEngine: MLXHibikiRuntimeEngine {
    public typealias WeightsReader = (URL) throws -> MLXHibikiWeightsSummary

    private let weightsReader: WeightsReader
    private var loaded = false

    public init(
        weightsReader: @escaping WeightsReader = { url in
            let arrays = try loadArrays(url: url)
            let keys = Set(arrays.keys)
            let hasNorms = keys.contains { key in
                key.hasPrefix("depformer_norms.")
                    || (key.hasPrefix("depformer.slices.") && key.contains(".norm."))
            }
            return MLXHibikiWeightsSummary(
                tensorCount: arrays.count,
                hasDepformerOutputLayerNorms: hasNorms
            )
        }
    ) {
        self.weightsReader = weightsReader
    }

    public func load(request: MLXHibikiLoadRequest) throws {
        let config = try MLXHibikiModelConfig.load(from: request.configURL)
        try config.validate(against: request.runtimeConfiguration)

        let summary = try weightsReader(request.weightsURL)
        guard summary.tensorCount > 0 else {
            throw HibikiInferenceError.invalidArtifacts("hibiki weights contained no tensors")
        }
        if request.runtimeConfiguration.depformerOutputLayerNorm && !summary.hasDepformerOutputLayerNorms {
            throw HibikiInferenceError.invalidArtifacts("hibiki weights missing depformer output LayerNorm tensors")
        }

        loaded = true
    }

    public func step(
        sourceAudioTokens: MimiTokenFrame,
        frameIndex: Int,
        configuration: HibikiGenerationConfiguration
    ) throws -> MLXHibikiStepOutput {
        guard loaded else {
            throw HibikiInferenceError.notInitialized
        }
        throw HibikiInferenceError.unavailable("Hibiki MLX model step graph not implemented")
    }

    public func reset() {
        loaded = false
    }
}

public final class MLXHibikiInferenceSession: HibikiInferenceSession, @unchecked Sendable {
    private let runtimeConfiguration: MLXHibikiRuntimeConfiguration
    private let engine: MLXHibikiRuntimeEngine
    private let textTokenDecoder: any HibikiTextTokenDecoding
    private let state = Mutex(MLXHibikiInferenceState())

    public init(
        runtimeConfiguration: MLXHibikiRuntimeConfiguration = MLXHibikiRuntimeConfiguration(),
        engine: MLXHibikiRuntimeEngine = MLXHibikiDefaultRuntimeEngine(),
        textTokenDecoder: any HibikiTextTokenDecoding = EmptyHibikiTextTokenDecoder()
    ) {
        self.runtimeConfiguration = runtimeConfiguration
        self.engine = engine
        self.textTokenDecoder = textTokenDecoder
    }

    public func initialize(
        artifacts: PreparedModelArtifacts,
        configuration: HibikiGenerationConfiguration
    ) async throws -> HibikiInferenceDescription {
        let request = try loadRequest(from: artifacts)

        do {
            try engine.load(request: request)
        } catch let error as HibikiInferenceError {
            throw error
        } catch {
            throw HibikiInferenceError.unavailable(String(describing: error))
        }

        state.withLock { state in
            state.initialized = true
            state.nextFrameIndex = 0
            state.generationConfiguration = configuration
        }

        return HibikiInferenceDescription(
            modelRevision: artifacts.manifest.revision,
            artifactCount: artifacts.files.count,
            configuration: configuration
        )
    }

    public func step(sourceAudioTokens: MimiTokenFrame) async throws -> HibikiInferenceStep {
        guard sourceAudioTokens.codebookCount == runtimeConfiguration.codebookCount else {
            throw HibikiInferenceError.unsupportedCodebookCount(sourceAudioTokens.codebookCount)
        }
        guard sourceAudioTokens.tokens.count == runtimeConfiguration.codebookCount else {
            throw HibikiInferenceError.invalidArtifacts(
                "source token frame malformed: expected \(runtimeConfiguration.codebookCount) tokens, got \(sourceAudioTokens.tokens.count)"
            )
        }

        let (frameIndex, generationConfiguration) = try state.withLock { state in
            guard state.initialized else {
                throw HibikiInferenceError.notInitialized
            }
            let frameIndex = state.nextFrameIndex
            state.nextFrameIndex += 1
            return (frameIndex, state.generationConfiguration)
        }

        let output = try engine.step(
            sourceAudioTokens: sourceAudioTokens,
            frameIndex: frameIndex,
            configuration: generationConfiguration
        )
        guard output.audioTokens.count == runtimeConfiguration.codebookCount else {
            throw HibikiInferenceError.invalidArtifacts(
                "generated audio token frame malformed: expected \(runtimeConfiguration.codebookCount) tokens, got \(output.audioTokens.count)"
            )
        }

        let text = HibikiTextOutput(
            frameIndex: frameIndex,
            token: output.textToken,
            piece: output.textPiece ?? textTokenDecoder.piece(for: output.textToken),
            candidateTokens: output.textCandidateTokens
        )
        let generatedAudio = MimiTokenFrame(
            frameIndex: frameIndex,
            timestampMilliseconds: sourceAudioTokens.timestampMilliseconds,
            codebookCount: runtimeConfiguration.codebookCount,
            tokens: output.audioTokens,
            sourceAudioFrameIndex: sourceAudioTokens.sourceAudioFrameIndex
        )

        return HibikiInferenceStep(
            frameIndex: frameIndex,
            sourceAudioTokens: sourceAudioTokens,
            text: text,
            generatedAudioTokens: generatedAudio
        )
    }

    public func reset() {
        engine.reset()
        state.withLock { state in
            state.initialized = false
            state.nextFrameIndex = 0
            state.generationConfiguration = HibikiGenerationConfiguration()
        }
    }

    private func loadRequest(from artifacts: PreparedModelArtifacts) throws -> MLXHibikiLoadRequest {
        let config = try artifact(role: "architectureConfig", in: artifacts)
        let weights = try artifact(role: "hibikiWeights", in: artifacts)
        let tokenizer = try artifact(role: "tokenizer", in: artifacts)

        return MLXHibikiLoadRequest(
            configURL: URL(fileURLWithPath: config.location),
            weightsURL: URL(fileURLWithPath: weights.location),
            tokenizerURL: URL(fileURLWithPath: tokenizer.location),
            artifactCount: artifacts.files.count,
            modelRevision: artifacts.manifest.revision,
            runtimeConfiguration: runtimeConfiguration
        )
    }

    private func artifact(role: String, in artifacts: PreparedModelArtifacts) throws -> PreparedModelArtifact {
        guard let artifact = artifacts.files.first(where: { $0.role == role }) else {
            throw HibikiInferenceError.invalidArtifacts("missing \(role)")
        }
        guard FileManager.default.fileExists(atPath: artifact.location) else {
            throw HibikiInferenceError.invalidArtifacts("missing \(artifact.fileName)")
        }
        return artifact
    }
}

private struct MLXHibikiInferenceState: Sendable {
    var initialized = false
    var nextFrameIndex = 0
    var generationConfiguration = HibikiGenerationConfiguration()
}

struct MLXHibikiModelConfig: Equatable {
    var hiddenScale: Int
    var kvRepeat: Int
    var positionalEmbedding: String
    var topology: MLXHibikiModelTopology

    static func load(from url: URL) throws -> MLXHibikiModelConfig {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw HibikiInferenceError.invalidArtifacts("config unreadable: \(url.lastPathComponent)")
        }

        let object: Any
        do {
            object = try JSONSerialization.jsonObject(with: data)
        } catch {
            throw HibikiInferenceError.invalidArtifacts("config JSON malformed: \(url.lastPathComponent)")
        }
        guard let dictionary = object as? [String: Any] else {
            throw HibikiInferenceError.invalidArtifacts("config JSON root malformed: \(url.lastPathComponent)")
        }

        return MLXHibikiModelConfig(
            hiddenScale: try intValue("hidden_scale", in: dictionary),
            kvRepeat: try intValue("kv_repeat", in: dictionary),
            positionalEmbedding: try positionalEmbedding(in: dictionary),
            topology: try MLXHibikiModelTopology.load(from: dictionary)
        )
    }

    func validate(against expected: MLXHibikiRuntimeConfiguration) throws {
        guard expected.quantizationBits == 4, expected.quantizationGroupSize == 32 else {
            throw HibikiInferenceError.invalidArtifacts(
                "unsupported q4 quantization: bits \(expected.quantizationBits), group size \(expected.quantizationGroupSize)"
            )
        }
        guard hiddenScale == expected.hiddenScale else {
            throw HibikiInferenceError.invalidArtifacts(
                "hidden_scale expected \(expected.hiddenScale), got \(hiddenScale)"
            )
        }
        guard kvRepeat == expected.kvRepeat else {
            throw HibikiInferenceError.invalidArtifacts(
                "kv_repeat expected \(expected.kvRepeat), got \(kvRepeat)"
            )
        }
        guard positionalEmbedding == expected.positionalEmbedding else {
            throw HibikiInferenceError.invalidArtifacts(
                "positional_embedding expected \(expected.positionalEmbedding), got \(positionalEmbedding)"
            )
        }
        try topology.validate(against: expected)
    }

    fileprivate static func intValue(_ key: String, in dictionary: [String: Any]) throws -> Int {
        guard let value = dictionary[key] as? Int else {
            throw HibikiInferenceError.invalidArtifacts("config missing \(key)")
        }
        return value
    }

    fileprivate static func optionalIntValue(_ key: String, in dictionary: [String: Any]) throws -> Int? {
        guard let value = dictionary[key], !(value is NSNull) else {
            return nil
        }
        guard let intValue = value as? Int else {
            throw HibikiInferenceError.invalidArtifacts("config \(key) malformed")
        }
        return intValue
    }

    fileprivate static func boolValue(_ key: String, in dictionary: [String: Any]) throws -> Bool {
        guard let value = dictionary[key] as? Bool else {
            throw HibikiInferenceError.invalidArtifacts("config missing \(key)")
        }
        return value
    }

    fileprivate static func optionalBoolValue(_ key: String, in dictionary: [String: Any]) throws -> Bool? {
        guard let value = dictionary[key], !(value is NSNull) else {
            return nil
        }
        guard let boolValue = value as? Bool else {
            throw HibikiInferenceError.invalidArtifacts("config \(key) malformed")
        }
        return boolValue
    }

    fileprivate static func optionalFloatValue(_ key: String, in dictionary: [String: Any]) throws -> Float? {
        guard let value = dictionary[key], !(value is NSNull) else {
            return nil
        }
        if let doubleValue = value as? Double {
            return Float(doubleValue)
        }
        if let intValue = value as? Int {
            return Float(intValue)
        }
        throw HibikiInferenceError.invalidArtifacts("config \(key) malformed")
    }

    fileprivate static func intFromNumber(_ key: String, in dictionary: [String: Any]) throws -> Int {
        guard let value = dictionary[key] else {
            throw HibikiInferenceError.invalidArtifacts("config missing \(key)")
        }
        if let intValue = value as? Int {
            return intValue
        }
        if let doubleValue = value as? Double {
            return Int(doubleValue)
        }
        throw HibikiInferenceError.invalidArtifacts("config \(key) malformed")
    }

    fileprivate static func stringValue(_ key: String, in dictionary: [String: Any]) throws -> String {
        guard let value = dictionary[key] as? String else {
            throw HibikiInferenceError.invalidArtifacts("config missing \(key)")
        }
        return value
    }

    fileprivate static func intArrayValue(_ key: String, in dictionary: [String: Any]) throws -> [Int] {
        guard let value = dictionary[key] as? [Int] else {
            throw HibikiInferenceError.invalidArtifacts("config missing \(key)")
        }
        return value
    }

    fileprivate static func optionalIntArrayValue(_ key: String, in dictionary: [String: Any]) throws -> [Int]? {
        guard let value = dictionary[key], !(value is NSNull) else {
            return nil
        }
        guard let intArray = value as? [Int] else {
            throw HibikiInferenceError.invalidArtifacts("config \(key) malformed")
        }
        return intArray
    }

    private static func positionalEmbedding(in dictionary: [String: Any]) throws -> String {
        if let value = dictionary["positional_embedding"] as? String {
            return value
        }
        if let transformer = dictionary["transformer"] as? [String: Any],
           let value = transformer["positional_embedding"] as? String {
            return value
        }
        throw HibikiInferenceError.invalidArtifacts("config missing positional_embedding")
    }
}

struct MLXHibikiModelTopology: Equatable {
    var mainTransformer: MLXMimiTransformerConfiguration
    var depformerTransformer: MLXMimiTransformerConfiguration
    var textInputVocabSize: Int
    var textOutputVocabSize: Int
    var audioVocabSize: Int
    var audioCodebookCount: Int
    var generatedCodebookCount: Int
    var audioDelays: [Int]
    var depformerWeightsPerStepSchedule: [Int]?

    var sourceCodebookCount: Int {
        audioCodebookCount - generatedCodebookCount
    }

    static func load(from dictionary: [String: Any]) throws -> MLXHibikiModelTopology {
        let hiddenScale = try MLXHibikiModelConfig.intValue("hidden_scale", in: dictionary)
        let mainDimension = try MLXHibikiModelConfig.intValue("dim", in: dictionary)
        let depformerDimension = try MLXHibikiModelConfig.intValue("depformer_dim", in: dictionary)
        let generatedCodebookCount = try MLXHibikiModelConfig.intValue("dep_q", in: dictionary)
        let audioCodebookCount = try MLXHibikiModelConfig.intValue("n_q", in: dictionary)
        let delays = try MLXHibikiModelConfig.intArrayValue("delays", in: dictionary)
        guard delays.count == audioCodebookCount + 1 else {
            throw HibikiInferenceError.invalidArtifacts(
                "delays expected \(audioCodebookCount + 1) values, got \(delays.count)"
            )
        }

        return MLXHibikiModelTopology(
            mainTransformer: MLXMimiTransformerConfiguration(
                modelDimension: mainDimension,
                headCount: try MLXHibikiModelConfig.intValue("num_heads", in: dictionary),
                layerCount: try MLXHibikiModelConfig.intValue("num_layers", in: dictionary),
                causal: try MLXHibikiModelConfig.boolValue("causal", in: dictionary),
                normFirst: true,
                feedForwardBias: false,
                attentionBias: false,
                layerScale: try MLXHibikiModelConfig.optionalFloatValue("layer_scale", in: dictionary),
                positionalEmbedding: try positionalEmbedding(
                    MLXHibikiModelConfig.stringValue("positional_embedding", in: dictionary)
                ),
                usesConvBias: true,
                gating: true,
                norm: try norm(MLXHibikiModelConfig.stringValue("norm", in: dictionary)),
                context: try MLXHibikiModelConfig.intValue("context", in: dictionary),
                maxPeriod: try MLXHibikiModelConfig.intFromNumber("max_period", in: dictionary),
                maxSequenceLength: 4_096,
                kvRepeat: try MLXHibikiModelConfig.intValue("kv_repeat", in: dictionary),
                feedForwardDimension: hiddenScale * mainDimension,
                convLayout: false,
                usesRotatingKVCache: true
            ),
            depformerTransformer: MLXMimiTransformerConfiguration(
                modelDimension: depformerDimension,
                headCount: try MLXHibikiModelConfig.intValue("depformer_num_heads", in: dictionary),
                layerCount: try MLXHibikiModelConfig.intValue("depformer_num_layers", in: dictionary),
                causal: try MLXHibikiModelConfig.optionalBoolValue("depformer_causal", in: dictionary) ?? true,
                normFirst: true,
                feedForwardBias: false,
                attentionBias: try MLXHibikiModelConfig.optionalBoolValue("depformer_layer_scale", in: dictionary) ?? false,
                layerScale: nil,
                positionalEmbedding: try positionalEmbedding(
                    MLXHibikiModelConfig.stringValue("depformer_pos_emb", in: dictionary)
                ),
                usesConvBias: true,
                gating: true,
                norm: try norm(MLXHibikiModelConfig.stringValue("depformer_norm", in: dictionary)),
                context: try MLXHibikiModelConfig.optionalIntValue("depformer_context", in: dictionary)
                    ?? generatedCodebookCount,
                maxPeriod: try MLXHibikiModelConfig.optionalIntValue("depformer_max_period", in: dictionary) ?? 8,
                maxSequenceLength: 4_096,
                kvRepeat: try MLXHibikiModelConfig.optionalIntValue("depformer_kv_repeat", in: dictionary) ?? 1,
                feedForwardDimension: try MLXHibikiModelConfig.optionalIntValue(
                    "depformer_dim_feedforward",
                    in: dictionary
                ) ?? hiddenScale * depformerDimension,
                convLayout: false,
                usesRotatingKVCache: false
            ),
            textInputVocabSize: try MLXHibikiModelConfig.intValue("text_card", in: dictionary) + 1,
            textOutputVocabSize: try MLXHibikiModelConfig.optionalIntValue("text_card_out", in: dictionary)
                ?? MLXHibikiModelConfig.intValue("text_card", in: dictionary),
            audioVocabSize: try MLXHibikiModelConfig.intValue("card", in: dictionary) + 1,
            audioCodebookCount: audioCodebookCount,
            generatedCodebookCount: generatedCodebookCount,
            audioDelays: Array(delays.dropFirst()),
            depformerWeightsPerStepSchedule: try MLXHibikiModelConfig.optionalIntArrayValue(
                "depformer_weights_per_step_schedule",
                in: dictionary
            )
        )
    }

    func validate(against expected: MLXHibikiRuntimeConfiguration) throws {
        guard mainTransformer.headCount % mainTransformer.kvRepeat == 0 else {
            throw HibikiInferenceError.invalidArtifacts(
                "main transformer heads \(mainTransformer.headCount) not divisible by kv_repeat \(mainTransformer.kvRepeat)"
            )
        }
        guard depformerTransformer.headCount % depformerTransformer.kvRepeat == 0 else {
            throw HibikiInferenceError.invalidArtifacts(
                "depformer heads \(depformerTransformer.headCount) not divisible by kv_repeat \(depformerTransformer.kvRepeat)"
            )
        }
        guard sourceCodebookCount == expected.codebookCount else {
            throw HibikiInferenceError.invalidArtifacts(
                "source codebook count expected \(expected.codebookCount), got \(sourceCodebookCount)"
            )
        }
        guard generatedCodebookCount == expected.codebookCount else {
            throw HibikiInferenceError.invalidArtifacts(
                "generated codebook count expected \(expected.codebookCount), got \(generatedCodebookCount)"
            )
        }
    }

    private static func positionalEmbedding(_ value: String) throws -> MLXMimiPositionalEmbedding {
        switch value {
        case "none":
            return .none
        case "rope":
            return .rope
        case "rope_concat":
            return .ropeConcat
        default:
            throw HibikiInferenceError.invalidArtifacts("unsupported positional_embedding \(value)")
        }
    }

    private static func norm(_ value: String) throws -> MLXMimiNorm {
        switch value {
        case "layer_norm":
            return .layerNorm
        case "rms_norm", "rms_norm_f32":
            return .rmsNorm
        default:
            throw HibikiInferenceError.invalidArtifacts("unsupported norm \(value)")
        }
    }
}

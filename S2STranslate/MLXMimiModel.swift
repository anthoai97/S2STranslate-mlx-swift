import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

public enum MLXMimiNorm: Equatable, Sendable {
    case layerNorm
    case rmsNorm
}

public enum MLXMimiPositionalEmbedding: Equatable, Sendable {
    case none
    case rope
}

public enum MLXMimiPadMode: Equatable, Sendable {
    case constant
    case edge

    var mlxPadMode: PadMode {
        switch self {
        case .constant:
            .constant
        case .edge:
            .edge
        }
    }
}

public struct MLXMimiSeanetConfiguration: Equatable, Sendable {
    public var dimension: Int
    public var channels: Int
    public var causal: Bool
    public var filterCount: Int
    public var residualLayerCount: Int
    public var ratios: [Int]
    public var kernelSize: Int
    public var residualKernelSize: Int
    public var lastKernelSize: Int
    public var dilationBase: Int
    public var padMode: MLXMimiPadMode
    public var trueSkip: Bool
    public var compress: Int

    nonisolated public static func v0_1() -> MLXMimiSeanetConfiguration {
        MLXMimiSeanetConfiguration(
            dimension: 512,
            channels: 1,
            causal: true,
            filterCount: 64,
            residualLayerCount: 1,
            ratios: [8, 6, 5, 4],
            kernelSize: 7,
            residualKernelSize: 3,
            lastKernelSize: 3,
            dilationBase: 2,
            padMode: .constant,
            trueSkip: true,
            compress: 2
        )
    }
}

public struct MLXMimiTransformerConfiguration: Equatable, Sendable {
    public var modelDimension: Int
    public var headCount: Int
    public var layerCount: Int
    public var causal: Bool
    public var normFirst: Bool
    public var feedForwardBias: Bool
    public var attentionBias: Bool
    public var layerScale: Float?
    public var positionalEmbedding: MLXMimiPositionalEmbedding
    public var usesConvBias: Bool
    public var gating: Bool
    public var norm: MLXMimiNorm
    public var context: Int
    public var maxPeriod: Int
    public var maxSequenceLength: Int
    public var kvRepeat: Int
    public var feedForwardDimension: Int
    public var convLayout: Bool
    public var usesRotatingKVCache: Bool

    nonisolated public var headDimension: Int {
        modelDimension / headCount
    }

    nonisolated public static func mimi202407(modelDimension: Int) -> MLXMimiTransformerConfiguration {
        MLXMimiTransformerConfiguration(
            modelDimension: modelDimension,
            headCount: 8,
            layerCount: 8,
            causal: true,
            normFirst: true,
            feedForwardBias: false,
            attentionBias: false,
            layerScale: 0.01,
            positionalEmbedding: .rope,
            usesConvBias: true,
            gating: false,
            norm: .layerNorm,
            context: 250,
            maxPeriod: 10_000,
            maxSequenceLength: 8_192,
            kvRepeat: 1,
            feedForwardDimension: 2_048,
            convLayout: true,
            usesRotatingKVCache: true
        )
    }
}

public struct MLXMimiConfiguration: Equatable, Sendable {
    public var channels: Int
    public var sampleRate: Int
    public var frameRate: Double
    public var renormalize: Bool
    public var seanet: MLXMimiSeanetConfiguration
    public var transformer: MLXMimiTransformerConfiguration
    public var quantizerCodebookCount: Int
    public var quantizerBins: Int
    public var quantizerDimension: Int

    nonisolated public var samplesPerFrame: Int {
        Int(Double(sampleRate) / frameRate)
    }

    nonisolated public var encoderFrameRate: Double {
        Double(sampleRate) / Double(seanet.ratios.reduce(1, *))
    }

    nonisolated public var downsampleStride: Int {
        Int(encoderFrameRate / frameRate)
    }

    nonisolated public static func mimi202407(codebookCount: Int = 16) -> MLXMimiConfiguration {
        let seanet = MLXMimiSeanetConfiguration.v0_1()
        return MLXMimiConfiguration(
            channels: 1,
            sampleRate: 24_000,
            frameRate: 12.5,
            renormalize: true,
            seanet: seanet,
            transformer: .mimi202407(modelDimension: seanet.dimension),
            quantizerCodebookCount: codebookCount,
            quantizerBins: 2_048,
            quantizerDimension: 256
        )
    }
}

public final class MLXMimiModel {
    public let configuration: MLXMimiConfiguration
    public let batchSize: Int
    public let encoder: MLXMimiSeanetEncoder
    public let decoder: MLXMimiSeanetDecoder
    public let encoderTransformer: MLXMimiProjectedTransformer
    public let decoderTransformer: MLXMimiProjectedTransformer
    public let encoderTransformerCache: [MLXMimiKVCache]
    public let decoderTransformerCache: [MLXMimiKVCache]
    public let downsample: MLXMimiConvDownsample1d
    public let upsample: MLXMimiConvTrUpsample1d
    public let quantizer: MLXMimiSplitResidualVectorQuantizer

    nonisolated public init(configuration: MLXMimiConfiguration = .mimi202407(), batchSize: Int = 1) {
        self.configuration = configuration
        self.batchSize = batchSize
        self.encoder = MLXMimiSeanetEncoder(configuration.seanet)
        self.decoder = MLXMimiSeanetDecoder(configuration.seanet)
        self.encoderTransformer = MLXMimiProjectedTransformer(
            configuration.transformer,
            inputDimension: configuration.seanet.dimension,
            outputDimensions: [configuration.seanet.dimension]
        )
        self.decoderTransformer = MLXMimiProjectedTransformer(
            configuration.transformer,
            inputDimension: configuration.seanet.dimension,
            outputDimensions: [configuration.seanet.dimension]
        )
        self.encoderTransformerCache = encoderTransformer.makeCache(batchSize: batchSize)
        self.decoderTransformerCache = decoderTransformer.makeCache(batchSize: batchSize)
        self.downsample = MLXMimiConvDownsample1d(
            stride: configuration.downsampleStride,
            dimension: configuration.seanet.dimension,
            causal: true
        )
        self.upsample = MLXMimiConvTrUpsample1d(
            stride: configuration.downsampleStride,
            dimension: configuration.seanet.dimension,
            causal: true
        )
        self.quantizer = MLXMimiSplitResidualVectorQuantizer(
            dimension: configuration.quantizerDimension,
            inputDimension: configuration.seanet.dimension,
            outputDimension: configuration.seanet.dimension,
            codebookCount: configuration.quantizerCodebookCount,
            bins: configuration.quantizerBins
        )
    }

    public func resetEncodeState() {
        encoder.resetState()
        MLXMimiProjectedTransformer.resetCache(encoderTransformerCache)
        downsample.resetState()
    }

    public func resetDecodeState() {
        MLXMimiProjectedTransformer.resetCache(decoderTransformerCache)
        upsample.resetState()
    }

    public func resetState() {
        resetEncodeState()
        resetDecodeState()
    }
}

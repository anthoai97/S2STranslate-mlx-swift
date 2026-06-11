import Foundation
import MLX
import Testing

@testable import S2STranslateCore

@Suite("MLX Mimi Graph Parameters")
struct MLXMimiGraphParameterTests {
    @Test("graph parameter applier exposes expected target shapes")
    func graphParameterApplierExposesExpectedTargetShapes() {
        let model = makeTinyMimiModel()
        let shapes = MLXMimiGraphParameterApplier.expectedShapes(for: model)

        #expect(shapes["encoder.init_conv1d.conv.weight"] == [2, 3, 1])
        #expect(shapes["encoder_transformer.layers.0.self_attn.in_proj.weight"] == [24, 8])
        #expect(shapes["encoder_transformer.layers.0.gating.linear1.weight"] == [16, 8])
        #expect(shapes["quantizer.rvq_first.input_proj.weight"] == [4, 1, 8])
        #expect(shapes["quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"] == [3, 4])
    }

    @Test("graph parameter applier validates injected tiny parameter shapes")
    func graphParameterApplierValidatesInjectedTinyParameterShapes() throws {
        let model = makeTinyMimiModel()
        let artifact = try makeGraphParameterArtifact()
        let weights = LoadedMLXMimiWeights(
            artifact: artifact,
            mappedTensors: [
                "encoder_transformer.layers.0.self_attn.in_proj.weight": MLXMimiWeightTensor(
                    shape: [24, 8],
                    array: nil
                ),
                "quantizer.rvq_first.input_proj.weight": MLXMimiWeightTensor(
                    shape: [4, 1, 8],
                    array: nil
                ),
            ]
        )
        let applier = MLXMimiGraphParameterApplier(
            requiredKeys: [
                "encoder_transformer.layers.0.self_attn.in_proj.weight",
                "quantizer.rvq_first.input_proj.weight",
            ],
            requiresArrayPayload: false
        )

        try applier.apply(weights, to: model)
    }

    @Test("graph parameter applier assigns injected tiny tensors")
    func graphParameterApplierAssignsInjectedTinyTensors() throws {
        let model = makeTinyMimiModel()
        let artifact = try makeGraphParameterArtifact()
        let weights = LoadedMLXMimiWeights(
            artifact: artifact,
            mappedTensors: [
                "encoder_transformer.layers.0.self_attn.in_proj.weight": MLXMimiWeightTensor(
                    shape: [24, 8],
                    array: tinyArray(shape: [24, 8])
                ),
                "quantizer.rvq_first.input_proj.weight": MLXMimiWeightTensor(
                    shape: [4, 1, 8],
                    array: tinyArray(shape: [4, 1, 8])
                ),
            ]
        )
        let applier = MLXMimiGraphParameterApplier(
            requiredKeys: [
                "encoder_transformer.layers.0.self_attn.in_proj.weight",
                "quantizer.rvq_first.input_proj.weight",
            ]
        )

        try applier.apply(weights, to: model)

        #expect(
            model.encoderTransformer.transformer.layers[0].selfAttention.qkvProjection.weight.shape
                == [24, 8]
        )
        #expect(model.quantizer.rvqFirst.inputProjection?.weight.shape == [4, 1, 8])
    }

    @Test("graph parameter applier reports missing incompatible and shape-only parameters")
    func graphParameterApplierReportsMissingIncompatibleAndShapeOnlyParameters() throws {
        let model = makeTinyMimiModel()
        let artifact = try makeGraphParameterArtifact()
        let requiredKey = "encoder_transformer.layers.0.self_attn.in_proj.weight"
        let applier = MLXMimiGraphParameterApplier(requiredKeys: [requiredKey])

        #expect(throws: MLXMimiWeightLoadError.missingKey(requiredKey)) {
            try applier.apply(
                LoadedMLXMimiWeights(artifact: artifact, mappedTensors: [:]),
                to: model
            )
        }

        #expect(
            throws: MLXMimiWeightLoadError.incompatibleShape(
                key: requiredKey,
                expected: [24, 8],
                actual: [1]
            )
        ) {
            try applier.apply(
                LoadedMLXMimiWeights(
                    artifact: artifact,
                    mappedTensors: [
                        requiredKey: MLXMimiWeightTensor(shape: [1], array: nil),
                    ]
                ),
                to: model
            )
        }

        #expect(
            throws: MLXMimiWeightLoadError.loadFailed(
                "Mapped tensor has no MLX array payload: \(requiredKey)"
            )
        ) {
            try applier.apply(
                LoadedMLXMimiWeights(
                    artifact: artifact,
                    mappedTensors: [
                        requiredKey: MLXMimiWeightTensor(shape: [24, 8], array: nil),
                    ]
                ),
                to: model
            )
        }
    }
}

private func makeTinyMimiModel() -> MLXMimiModel {
    let seanet = MLXMimiSeanetConfiguration(
        dimension: 8,
        channels: 1,
        causal: true,
        filterCount: 2,
        residualLayerCount: 1,
        ratios: [2],
        kernelSize: 3,
        residualKernelSize: 3,
        lastKernelSize: 3,
        dilationBase: 2,
        padMode: .constant,
        trueSkip: true,
        compress: 2
    )
    let transformer = MLXMimiTransformerConfiguration(
        modelDimension: 8,
        headCount: 2,
        layerCount: 1,
        causal: true,
        normFirst: true,
        feedForwardBias: false,
        attentionBias: false,
        layerScale: 0.01,
        positionalEmbedding: .rope,
        usesConvBias: true,
        gating: false,
        norm: .layerNorm,
        context: 4,
        maxPeriod: 10_000,
        maxSequenceLength: 16,
        kvRepeat: 1,
        feedForwardDimension: 16,
        convLayout: true,
        usesRotatingKVCache: true
    )
    let configuration = MLXMimiConfiguration(
        channels: 1,
        sampleRate: 24_000,
        frameRate: 12.5,
        renormalize: true,
        seanet: seanet,
        transformer: transformer,
        quantizerCodebookCount: 2,
        quantizerBins: 3,
        quantizerDimension: 4
    )
    return MLXMimiModel(configuration: configuration, batchSize: 1)
}

private func tinyArray(shape: [Int]) -> MLXArray {
    let count = shape.reduce(1, *)
    return MLXArray(Array(repeating: Float(0), count: count)).reshaped(shape)
}

private func makeGraphParameterArtifact() throws -> PreparedModelArtifact {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let weightsURL = directory.appendingPathComponent("mimi.safetensors")
    try Data().write(to: weightsURL)
    return PreparedModelArtifact(
        role: "mimiWeights",
        fileName: "mimi.safetensors",
        location: weightsURL.path,
        source: .prepared
    )
}

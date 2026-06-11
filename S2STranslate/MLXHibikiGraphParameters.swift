import MLX

struct MLXHibikiQuantizationSpec: Equatable {
    var bits: Int
    var groupSize: Int

    init(bits: Int = 4, groupSize: Int = 32) {
        self.bits = bits
        self.groupSize = groupSize
    }
}

private struct MLXHibikiDenseParameterTarget {
    var key: String
    var expectedShape: [Int]
    var assign: (MLXArray) -> Void
}

private struct MLXHibikiQuantizedParameterTarget {
    var prefix: String
    var weightShape: [Int]
    var scaleBiasShape: [Int]
    var assign: (MLXArray, MLXArray, MLXArray) -> Void

    var keyedShapes: [(String, [Int])] {
        [
            ("\(prefix).weight", weightShape),
            ("\(prefix).scales", scaleBiasShape),
            ("\(prefix).biases", scaleBiasShape),
        ]
    }
}

struct MLXHibikiGraphParameterApplier {
    var requiredKeys: Set<String>?
    var allowsUnexpectedKeys: Bool
    var requiresArrayPayload: Bool
    var quantization: MLXHibikiQuantizationSpec

    init(
        requiredKeys: Set<String>? = nil,
        allowsUnexpectedKeys: Bool = true,
        requiresArrayPayload: Bool = true,
        quantization: MLXHibikiQuantizationSpec = MLXHibikiQuantizationSpec()
    ) {
        self.requiredKeys = requiredKeys
        self.allowsUnexpectedKeys = allowsUnexpectedKeys
        self.requiresArrayPayload = requiresArrayPayload
        self.quantization = quantization
    }

    func apply(_ weights: [String: MLXMimiWeightTensor], to model: MLXHibikiLanguageModel) throws {
        let denseTargets = Self.denseTargets(for: model)
        let quantizedTargets = Self.quantizedTargets(for: model, quantization: quantization)
        let expectedShapes = Self.expectedShapes(
            denseTargets: denseTargets,
            quantizedTargets: quantizedTargets
        )
        let keysToRequire = requiredKeys ?? Set(expectedShapes.keys)

        for key in keysToRequire where weights[key] == nil {
            throw MLXMimiWeightLoadError.missingKey(key)
        }

        for (key, tensor) in weights {
            guard let expectedShape = expectedShapes[key] else {
                if allowsUnexpectedKeys {
                    continue
                }
                throw MLXMimiWeightLoadError.unexpectedKey(key)
            }

            guard tensor.shape == expectedShape else {
                throw MLXMimiWeightLoadError.incompatibleShape(
                    key: key,
                    expected: expectedShape,
                    actual: tensor.shape
                )
            }
        }

        guard requiresArrayPayload else { return }

        for target in denseTargets {
            guard let tensor = weights[target.key] else { continue }
            guard let array = tensor.array else {
                throw MLXMimiWeightLoadError.loadFailed(
                    "Mapped tensor has no MLX array payload: \(target.key)"
                )
            }
            target.assign(array)
        }

        for target in quantizedTargets {
            let weightKey = "\(target.prefix).weight"
            let scalesKey = "\(target.prefix).scales"
            let biasesKey = "\(target.prefix).biases"
            guard weights[weightKey] != nil || weights[scalesKey] != nil || weights[biasesKey] != nil else {
                continue
            }
            guard let weight = weights[weightKey]?.array,
                  let scales = weights[scalesKey]?.array,
                  let biases = weights[biasesKey]?.array else {
                throw MLXMimiWeightLoadError.loadFailed(
                    "Mapped quantized tensor group incomplete: \(target.prefix)"
                )
            }
            target.assign(weight, scales, biases)
        }
    }

    static func expectedShapes(
        for model: MLXHibikiLanguageModel,
        quantization: MLXHibikiQuantizationSpec = MLXHibikiQuantizationSpec()
    ) -> [String: [Int]] {
        expectedShapes(
            denseTargets: denseTargets(for: model),
            quantizedTargets: quantizedTargets(for: model, quantization: quantization)
        )
    }

    private static func expectedShapes(
        denseTargets: [MLXHibikiDenseParameterTarget],
        quantizedTargets: [MLXHibikiQuantizedParameterTarget]
    ) -> [String: [Int]] {
        var shapes = Dictionary(uniqueKeysWithValues: denseTargets.map { ($0.key, $0.expectedShape) })
        for target in quantizedTargets {
            for (key, shape) in target.keyedShapes {
                shapes[key] = shape
            }
        }
        return shapes
    }

    private static func denseTargets(for model: MLXHibikiLanguageModel) -> [MLXHibikiDenseParameterTarget] {
        var targets: [MLXHibikiDenseParameterTarget] = []
        appendTransformerNormTargets(prefix: "out_norm", norm: model.outNorm, includeLayerNormBias: false, to: &targets)
        appendTransformerDenseTargets(prefix: "transformer", transformer: model.transformer, to: &targets)
        for (sliceIndex, slice) in model.depformerSlices.enumerated() {
            let prefix = "depformer.slices.\(sliceIndex)"
            appendLayerNormTargets(prefix: "\(prefix).norm", norm: slice.norm, to: &targets)
            appendTransformerDenseTargets(
                prefix: "\(prefix).transformer",
                transformer: slice.transformer,
                to: &targets
            )
        }
        return targets
    }

    private static func quantizedTargets(
        for model: MLXHibikiLanguageModel,
        quantization: MLXHibikiQuantizationSpec
    ) -> [MLXHibikiQuantizedParameterTarget] {
        var targets: [MLXHibikiQuantizedParameterTarget] = []
        appendEmbeddingTarget(prefix: "text_emb", embedding: model.textEmbedding, quantization: quantization, to: &targets)
        appendLinearTarget(prefix: "text_linear", linear: model.textLinear, quantization: quantization, to: &targets)
        for (index, embedding) in model.audioEmbeddings.enumerated() {
            appendEmbeddingTarget(
                prefix: "audio_embs.\(index)",
                embedding: embedding,
                quantization: quantization,
                to: &targets
            )
        }
        appendTransformerQuantizedTargets(
            prefix: "transformer",
            transformer: model.transformer,
            quantization: quantization,
            to: &targets
        )
        for (sliceIndex, slice) in model.depformerSlices.enumerated() {
            let prefix = "depformer.slices.\(sliceIndex)"
            appendEmbeddingTarget(
                prefix: "\(prefix).emb",
                embedding: slice.embedding,
                quantization: quantization,
                to: &targets
            )
            appendLinearTarget(
                prefix: "\(prefix).linear_in",
                linear: slice.linearIn,
                quantization: quantization,
                to: &targets
            )
            appendLinearTarget(
                prefix: "\(prefix).linear_out",
                linear: slice.linearOut,
                quantization: quantization,
                to: &targets
            )
            appendTransformerQuantizedTargets(
                prefix: "\(prefix).transformer",
                transformer: slice.transformer,
                quantization: quantization,
                to: &targets
            )
        }
        return targets
    }

    private static func appendTransformerDenseTargets(
        prefix: String,
        transformer: MLXMimiTransformer,
        to targets: inout [MLXHibikiDenseParameterTarget]
    ) {
        for (layerIndex, layer) in transformer.layers.enumerated() {
            let layerPrefix = "\(prefix).layers.\(layerIndex)"
            appendTransformerNormTargets(
                prefix: "\(layerPrefix).norm1",
                norm: layer.norm1,
                includeLayerNormBias: false,
                to: &targets
            )
            appendTransformerNormTargets(
                prefix: "\(layerPrefix).norm2",
                norm: layer.norm2,
                includeLayerNormBias: false,
                to: &targets
            )
            appendLayerScaleTargets(prefix: "\(layerPrefix).layer_scale_1", scale: layer.layerScale1, to: &targets)
            appendLayerScaleTargets(prefix: "\(layerPrefix).layer_scale_2", scale: layer.layerScale2, to: &targets)
        }
    }

    private static func appendTransformerQuantizedTargets(
        prefix: String,
        transformer: MLXMimiTransformer,
        quantization: MLXHibikiQuantizationSpec,
        to targets: inout [MLXHibikiQuantizedParameterTarget]
    ) {
        for (layerIndex, layer) in transformer.layers.enumerated() {
            let layerPrefix = "\(prefix).layers.\(layerIndex)"
            appendLinearTarget(
                prefix: "\(layerPrefix).self_attn.in_proj",
                linear: layer.selfAttention.qkvProjection,
                quantization: quantization,
                to: &targets
            )
            appendLinearTarget(
                prefix: "\(layerPrefix).self_attn.out_proj",
                linear: layer.selfAttention.outputProjection,
                quantization: quantization,
                to: &targets
            )
            if let ungated = layer.feedForward.ungated {
                appendLinearTarget(
                    prefix: "\(layerPrefix).gating.linear1",
                    linear: ungated.linear1,
                    quantization: quantization,
                    to: &targets
                )
                appendLinearTarget(
                    prefix: "\(layerPrefix).gating.linear2",
                    linear: ungated.linear2,
                    quantization: quantization,
                    to: &targets
                )
            }
            if let gated = layer.feedForward.gated {
                appendLinearTarget(
                    prefix: "\(layerPrefix).gating.linear_in",
                    linear: gated.linearIn,
                    quantization: quantization,
                    to: &targets
                )
                appendLinearTarget(
                    prefix: "\(layerPrefix).gating.linear_out",
                    linear: gated.linearOut,
                    quantization: quantization,
                    to: &targets
                )
            }
        }
    }

    private static func appendEmbeddingTarget(
        prefix: String,
        embedding: MLXHibikiEmbedding,
        quantization: MLXHibikiQuantizationSpec,
        to targets: inout [MLXHibikiQuantizedParameterTarget]
    ) {
        targets.append(
            MLXHibikiQuantizedParameterTarget(
                prefix: prefix,
                weightShape: embedding.quantizedWeightShape(bits: quantization.bits),
                scaleBiasShape: embedding.quantizedScaleBiasShape(groupSize: quantization.groupSize),
                assign: { weight, scales, biases in
                    embedding.setQuantizedParameters(
                        weight: weight,
                        scales: scales,
                        biases: biases,
                        groupSize: quantization.groupSize,
                        bits: quantization.bits
                    )
                }
            )
        )
    }

    private static func appendLinearTarget(
        prefix: String,
        linear: MLXMimiLinear,
        quantization: MLXHibikiQuantizationSpec,
        to targets: inout [MLXHibikiQuantizedParameterTarget]
    ) {
        targets.append(
            MLXHibikiQuantizedParameterTarget(
                prefix: prefix,
                weightShape: linear.quantizedWeightShape(bits: quantization.bits),
                scaleBiasShape: linear.quantizedScaleBiasShape(groupSize: quantization.groupSize),
                assign: { weight, scales, biases in
                    linear.setQuantizedParameters(
                        weight: weight,
                        scales: scales,
                        biases: biases,
                        groupSize: quantization.groupSize,
                        bits: quantization.bits
                    )
                }
            )
        )
    }

    private static func appendTransformerNormTargets(
        prefix: String,
        norm: MLXMimiTransformerNorm,
        includeLayerNormBias: Bool,
        to targets: inout [MLXHibikiDenseParameterTarget]
    ) {
        if let layerNorm = norm.layerNorm {
            appendLayerNormTargets(
                prefix: prefix,
                norm: layerNorm,
                includeBias: includeLayerNormBias,
                to: &targets
            )
        }
        if let rmsNorm = norm.rmsNorm {
            targets.append(
                MLXHibikiDenseParameterTarget(
                    key: "\(prefix).weight",
                    expectedShape: rmsNorm.weightShape,
                    assign: { rmsNorm.weight = $0 }
                )
            )
        }
    }

    private static func appendLayerNormTargets(
        prefix: String,
        norm: MLXMimiLayerNorm,
        includeBias: Bool = true,
        to targets: inout [MLXHibikiDenseParameterTarget]
    ) {
        targets.append(
            MLXHibikiDenseParameterTarget(
                key: "\(prefix).weight",
                expectedShape: norm.weightShape,
                assign: { norm.weight = $0 }
            )
        )
        if includeBias {
            targets.append(
                MLXHibikiDenseParameterTarget(
                    key: "\(prefix).bias",
                    expectedShape: norm.biasShape!,
                    assign: { norm.bias = $0 }
                )
            )
        }
    }

    private static func appendLayerScaleTargets(
        prefix: String,
        scale: MLXMimiLayerScale?,
        to targets: inout [MLXHibikiDenseParameterTarget]
    ) {
        guard let scale else { return }
        targets.append(
            MLXHibikiDenseParameterTarget(
                key: "\(prefix).scale",
                expectedShape: scale.scaleShape,
                assign: { scale.scale = $0 }
            )
        )
    }
}

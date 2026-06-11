import MLX

private struct MLXMimiGraphParameterTarget {
    var key: String
    var expectedShape: [Int]
    var assign: (MLXArray) -> Void
}

public struct MLXMimiGraphParameterApplier {
    public var requiredKeys: Set<String>?
    public var allowsUnexpectedKeys: Bool
    public var requiresArrayPayload: Bool

    public init(
        requiredKeys: Set<String>? = nil,
        allowsUnexpectedKeys: Bool = true,
        requiresArrayPayload: Bool = true
    ) {
        self.requiredKeys = requiredKeys
        self.allowsUnexpectedKeys = allowsUnexpectedKeys
        self.requiresArrayPayload = requiresArrayPayload
    }

    public func apply(_ weights: LoadedMLXMimiWeights, to model: MLXMimiModel) throws {
        let targets = Self.targets(for: model)
        let keysToRequire = requiredKeys ?? Set(targets.keys)

        for key in keysToRequire where weights.mappedTensors[key] == nil {
            throw MLXMimiWeightLoadError.missingKey(key)
        }

        for (key, tensor) in weights.mappedTensors {
            guard let target = targets[key] else {
                if allowsUnexpectedKeys {
                    continue
                }
                throw MLXMimiWeightLoadError.unexpectedKey(key)
            }

            guard tensor.shape == target.expectedShape else {
                throw MLXMimiWeightLoadError.incompatibleShape(
                    key: key,
                    expected: target.expectedShape,
                    actual: tensor.shape
                )
            }
            guard let array = tensor.array else {
                if !requiresArrayPayload {
                    continue
                }
                throw MLXMimiWeightLoadError.loadFailed(
                    "Mapped tensor has no MLX array payload: \(key)"
                )
            }

            target.assign(array)
        }
    }

    public static func expectedShapes(for model: MLXMimiModel) -> [String: [Int]] {
        targets(for: model).mapValues(\.expectedShape)
    }

    private static func targets(for model: MLXMimiModel) -> [String: MLXMimiGraphParameterTarget] {
        var targets: [MLXMimiGraphParameterTarget] = []
        appendEncoderTargets(prefix: "encoder", encoder: model.encoder, to: &targets)
        appendDecoderTargets(prefix: "decoder", decoder: model.decoder, to: &targets)
        appendDownsampleTargets(prefix: "downsample", downsample: model.downsample, to: &targets)
        appendUpsampleTargets(prefix: "upsample", upsample: model.upsample, to: &targets)
        appendProjectedTransformerTargets(
            prefix: "encoder_transformer",
            projectedTransformer: model.encoderTransformer,
            to: &targets
        )
        appendProjectedTransformerTargets(
            prefix: "decoder_transformer",
            projectedTransformer: model.decoderTransformer,
            to: &targets
        )
        appendQuantizerTargets(prefix: "quantizer", quantizer: model.quantizer, to: &targets)

        return Dictionary(uniqueKeysWithValues: targets.map { ($0.key, $0) })
    }

    private static func appendEncoderTargets(
        prefix: String,
        encoder: MLXMimiSeanetEncoder,
        to targets: inout [MLXMimiGraphParameterTarget]
    ) {
        appendStreamableConvTargets(prefix: "\(prefix).init_conv1d", conv: encoder.initConv1d, to: &targets)
        for (layerIndex, layer) in encoder.layers.enumerated() {
            for (residualIndex, residual) in layer.residuals.enumerated() {
                appendResnetBlockTargets(
                    prefix: "\(prefix).layers.\(layerIndex).residuals.\(residualIndex)",
                    block: residual,
                    to: &targets
                )
            }
            appendStreamableConvTargets(
                prefix: "\(prefix).layers.\(layerIndex).downsample",
                conv: layer.downsample,
                to: &targets
            )
        }
        appendStreamableConvTargets(prefix: "\(prefix).final_conv1d", conv: encoder.finalConv1d, to: &targets)
    }

    private static func appendDecoderTargets(
        prefix: String,
        decoder: MLXMimiSeanetDecoder,
        to targets: inout [MLXMimiGraphParameterTarget]
    ) {
        appendStreamableConvTargets(prefix: "\(prefix).init_conv1d", conv: decoder.initConv1d, to: &targets)
        for (layerIndex, layer) in decoder.layers.enumerated() {
            appendStreamableConvTransposeTargets(
                prefix: "\(prefix).layers.\(layerIndex).upsample",
                conv: layer.upsample,
                to: &targets
            )
            for (residualIndex, residual) in layer.residuals.enumerated() {
                appendResnetBlockTargets(
                    prefix: "\(prefix).layers.\(layerIndex).residuals.\(residualIndex)",
                    block: residual,
                    to: &targets
                )
            }
        }
        appendStreamableConvTargets(prefix: "\(prefix).final_conv1d", conv: decoder.finalConv1d, to: &targets)
    }

    private static func appendResnetBlockTargets(
        prefix: String,
        block: MLXMimiSeanetResnetBlock,
        to targets: inout [MLXMimiGraphParameterTarget]
    ) {
        for (blockIndex, conv) in block.block.enumerated() {
            appendStreamableConvTargets(prefix: "\(prefix).block.\(blockIndex)", conv: conv, to: &targets)
        }
        if let shortcut = block.shortcut {
            appendStreamableConvTargets(prefix: "\(prefix).shortcut", conv: shortcut, to: &targets)
        }
    }

    private static func appendDownsampleTargets(
        prefix: String,
        downsample: MLXMimiConvDownsample1d,
        to targets: inout [MLXMimiGraphParameterTarget]
    ) {
        appendStreamableConvTargets(prefix: prefix, conv: downsample.conv, to: &targets)
    }

    private static func appendUpsampleTargets(
        prefix: String,
        upsample: MLXMimiConvTrUpsample1d,
        to targets: inout [MLXMimiGraphParameterTarget]
    ) {
        appendStreamableConvTransposeTargets(prefix: prefix, conv: upsample.convtr, to: &targets)
    }

    private static func appendStreamableConvTargets(
        prefix: String,
        conv: MLXMimiStreamableConv1d,
        to targets: inout [MLXMimiGraphParameterTarget]
    ) {
        appendConvTargets(prefix: "\(prefix).conv", conv: conv.conv.conv, to: &targets)
    }

    private static func appendConvTargets(
        prefix: String,
        conv: MLXMimiConv1d,
        to targets: inout [MLXMimiGraphParameterTarget]
    ) {
        targets.append(
            MLXMimiGraphParameterTarget(
                key: "\(prefix).weight",
                expectedShape: conv.weightShape,
                assign: { conv.weight = $0 }
            )
        )
        if let biasShape = conv.biasShape {
            targets.append(
                MLXMimiGraphParameterTarget(
                    key: "\(prefix).bias",
                    expectedShape: biasShape,
                    assign: { conv.bias = $0 }
                )
            )
        }
    }

    private static func appendStreamableConvTransposeTargets(
        prefix: String,
        conv: MLXMimiStreamableConvTranspose1d,
        to targets: inout [MLXMimiGraphParameterTarget]
    ) {
        appendConvTransposeTargets(prefix: "\(prefix).convtr", conv: conv.convtr.convtr, to: &targets)
    }

    private static func appendConvTransposeTargets(
        prefix: String,
        conv: MLXMimiConvTransposed1d,
        to targets: inout [MLXMimiGraphParameterTarget]
    ) {
        targets.append(
            MLXMimiGraphParameterTarget(
                key: "\(prefix).weight",
                expectedShape: conv.weightShape,
                assign: { conv.weight = $0 }
            )
        )
        if let biasShape = conv.biasShape {
            targets.append(
                MLXMimiGraphParameterTarget(
                    key: "\(prefix).bias",
                    expectedShape: biasShape,
                    assign: { conv.bias = $0 }
                )
            )
        }
    }

    private static func appendProjectedTransformerTargets(
        prefix: String,
        projectedTransformer: MLXMimiProjectedTransformer,
        to targets: inout [MLXMimiGraphParameterTarget]
    ) {
        if let inputProjection = projectedTransformer.inputProjection {
            appendLinearTargets(prefix: "\(prefix).input_proj", linear: inputProjection, to: &targets)
        }
        for (outputIndex, outputProjection) in projectedTransformer.outputProjections.enumerated() {
            if let outputProjection {
                appendLinearTargets(prefix: "\(prefix).output_projs.\(outputIndex)", linear: outputProjection, to: &targets)
            }
        }

        for (layerIndex, layer) in projectedTransformer.transformer.layers.enumerated() {
            let layerPrefix = "\(prefix).layers.\(layerIndex)"
            appendLinearTargets(
                prefix: "\(layerPrefix).self_attn.in_proj",
                linear: layer.selfAttention.qkvProjection,
                to: &targets
            )
            appendLinearTargets(
                prefix: "\(layerPrefix).self_attn.out_proj",
                linear: layer.selfAttention.outputProjection,
                to: &targets
            )
            appendTransformerNormTargets(prefix: "\(layerPrefix).norm1", norm: layer.norm1, to: &targets)
            appendTransformerNormTargets(prefix: "\(layerPrefix).norm2", norm: layer.norm2, to: &targets)
            appendLayerScaleTargets(prefix: "\(layerPrefix).layer_scale_1", scale: layer.layerScale1, to: &targets)
            appendLayerScaleTargets(prefix: "\(layerPrefix).layer_scale_2", scale: layer.layerScale2, to: &targets)
            if let ungated = layer.feedForward.ungated {
                appendLinearTargets(prefix: "\(layerPrefix).gating.linear1", linear: ungated.linear1, to: &targets)
                appendLinearTargets(prefix: "\(layerPrefix).gating.linear2", linear: ungated.linear2, to: &targets)
            }
            if let gated = layer.feedForward.gated {
                appendLinearTargets(prefix: "\(layerPrefix).gating.linear_in", linear: gated.linearIn, to: &targets)
                appendLinearTargets(prefix: "\(layerPrefix).gating.linear_out", linear: gated.linearOut, to: &targets)
            }
        }
    }

    private static func appendLinearTargets(
        prefix: String,
        linear: MLXMimiLinear,
        to targets: inout [MLXMimiGraphParameterTarget]
    ) {
        targets.append(
            MLXMimiGraphParameterTarget(
                key: "\(prefix).weight",
                expectedShape: linear.weightShape,
                assign: { linear.weight = $0 }
            )
        )
        if let biasShape = linear.biasShape {
            targets.append(
                MLXMimiGraphParameterTarget(
                    key: "\(prefix).bias",
                    expectedShape: biasShape,
                    assign: { linear.bias = $0 }
                )
            )
        }
    }

    private static func appendTransformerNormTargets(
        prefix: String,
        norm: MLXMimiTransformerNorm,
        to targets: inout [MLXMimiGraphParameterTarget]
    ) {
        if let layerNorm = norm.layerNorm {
            targets.append(
                MLXMimiGraphParameterTarget(
                    key: "\(prefix).weight",
                    expectedShape: layerNorm.weightShape,
                    assign: { layerNorm.weight = $0 }
                )
            )
            targets.append(
                MLXMimiGraphParameterTarget(
                    key: "\(prefix).bias",
                    expectedShape: layerNorm.biasShape!,
                    assign: { layerNorm.bias = $0 }
                )
            )
        }
        if let rmsNorm = norm.rmsNorm {
            targets.append(
                MLXMimiGraphParameterTarget(
                    key: "\(prefix).weight",
                    expectedShape: rmsNorm.weightShape,
                    assign: { rmsNorm.weight = $0 }
                )
            )
        }
    }

    private static func appendLayerScaleTargets(
        prefix: String,
        scale: MLXMimiLayerScale?,
        to targets: inout [MLXMimiGraphParameterTarget]
    ) {
        guard let scale else { return }

        targets.append(
            MLXMimiGraphParameterTarget(
                key: "\(prefix).scale",
                expectedShape: scale.scaleShape,
                assign: { scale.scale = $0 }
            )
        )
    }

    private static func appendQuantizerTargets(
        prefix: String,
        quantizer: MLXMimiSplitResidualVectorQuantizer,
        to targets: inout [MLXMimiGraphParameterTarget]
    ) {
        appendResidualVectorQuantizerTargets(prefix: "\(prefix).rvq_first", quantizer: quantizer.rvqFirst, to: &targets)
        appendResidualVectorQuantizerTargets(prefix: "\(prefix).rvq_rest", quantizer: quantizer.rvqRest, to: &targets)
    }

    private static func appendResidualVectorQuantizerTargets(
        prefix: String,
        quantizer: MLXMimiResidualVectorQuantizer,
        to targets: inout [MLXMimiGraphParameterTarget]
    ) {
        if let inputProjection = quantizer.inputProjection {
            targets.append(
                MLXMimiGraphParameterTarget(
                    key: "\(prefix).input_proj.weight",
                    expectedShape: inputProjection.weightShape,
                    assign: { inputProjection.weight = $0 }
                )
            )
        }
        if let outputProjection = quantizer.outputProjection {
            targets.append(
                MLXMimiGraphParameterTarget(
                    key: "\(prefix).output_proj.weight",
                    expectedShape: outputProjection.weightShape,
                    assign: { outputProjection.weight = $0 }
                )
            )
        }
        for (layerIndex, layer) in quantizer.vq.layers.enumerated() {
            let codebook = layer.codebook
            targets.append(
                MLXMimiGraphParameterTarget(
                    key: "\(prefix).vq.layers.\(layerIndex)._codebook.embedding_sum",
                    expectedShape: codebook.embeddingShape,
                    assign: { codebook.embeddingSum = $0 }
                )
            )
            targets.append(
                MLXMimiGraphParameterTarget(
                    key: "\(prefix).vq.layers.\(layerIndex)._codebook.cluster_usage",
                    expectedShape: codebook.clusterUsageShape,
                    assign: { codebook.clusterUsage = $0 }
                )
            )
        }
    }
}

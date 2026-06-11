import Testing

@testable import S2STranslateCore

@Suite("MLX Mimi Transformer")
struct MLXMimiTransformerTests {
    @Test("projected transformer owns Mimi encoder topology")
    func projectedTransformerOwnsMimiEncoderTopology() {
        let configuration = MLXMimiConfiguration.mimi202407()
        let transformer = MLXMimiProjectedTransformer(
            configuration.transformer,
            inputDimension: configuration.seanet.dimension,
            outputDimensions: [configuration.seanet.dimension]
        )

        #expect(transformer.configuration.modelDimension == 512)
        #expect(transformer.configuration.headCount == 8)
        #expect(transformer.configuration.layerCount == 8)
        #expect(transformer.configuration.positionalEmbedding == .rope)
        #expect(transformer.configuration.context == 250)
        #expect(transformer.configuration.usesRotatingKVCache)
        #expect(transformer.convLayout)
        #expect(transformer.inputProjection == nil)
        #expect(transformer.outputProjections.count == 1)
        #expect(transformer.outputProjections[0] == nil)
        #expect(transformer.transformer.layers.count == 8)
        #expect(transformer.transformer.layers[0].selfAttention.qkvProjection.weightShape == [1_536, 512])
        #expect(transformer.transformer.layers[0].selfAttention.outputProjection.weightShape == [512, 512])
        #expect(transformer.transformer.layers[0].feedForward.ungated?.linear1.weightShape == [2_048, 512])
        #expect(transformer.transformer.layers[0].feedForward.ungated?.linear2.weightShape == [512, 2_048])
        #expect(transformer.transformer.layers[0].layerScale1?.scaleShape == [512])
        #expect(transformer.transformer.layers[0].layerScale2?.scaleShape == [512])
    }

    @Test("transformer cache preserves chunk boundary offset and reset")
    func transformerCachePreservesChunkBoundaryOffsetAndReset() {
        let configuration = MLXMimiTransformerConfiguration.mimi202407(modelDimension: 8)
        let transformer = MLXMimiTransformer(configuration)
        let cache = transformer.makeCache(batchSize: 1)

        (cache[0] as? MLXMimiRotatingKVCache)?.advanceOffsetForTesting(by: 2)
        (cache[0] as? MLXMimiRotatingKVCache)?.advanceOffsetForTesting(by: 2)

        #expect(cache.count == 8)
        #expect(cache[0].offset == 4)
        #expect(cache[0].maxSize == 250)

        MLXMimiProjectedTransformer.resetCache(cache)

        #expect(cache.allSatisfy { $0.offset == 0 })
    }

    @Test("empty transformer stream remains empty and cache does not advance")
    func emptyTransformerStreamRemainsEmptyAndCacheDoesNotAdvance() {
        let configuration = MLXMimiTransformerConfiguration.mimi202407(modelDimension: 8)
        let transformer = MLXMimiProjectedTransformer(
            configuration,
            inputDimension: 8,
            outputDimensions: [8]
        )
        let cache = transformer.makeCache(batchSize: 1)

        let output = transformer.step(MLXMimiStreamArray(), cache: cache)

        #expect(output.isEmpty)
        #expect(cache.allSatisfy { $0.offset == 0 })
    }

    @Test("projected transformer reports conv layout output shape")
    func projectedTransformerReportsConvLayoutOutputShape() {
        let configuration = MLXMimiTransformerConfiguration.mimi202407(modelDimension: 8)
        let transformer = MLXMimiProjectedTransformer(
            configuration,
            inputDimension: 8,
            outputDimensions: [8]
        )

        #expect(transformer.expectedOutputShapes(inputShape: [1, 8, 3]) == [[1, 8, 3]])
    }

    @Test("model reset clears encoder transformer cache")
    func modelResetClearsEncoderTransformerCache() {
        let configuration = MLXMimiConfiguration.mimi202407()
        let model = MLXMimiModel(configuration: configuration, batchSize: 1)

        (model.encoderTransformerCache[0] as? MLXMimiRotatingKVCache)?.advanceOffsetForTesting(by: 1)
        #expect(model.encoderTransformerCache[0].offset == 1)

        model.resetEncodeState()

        #expect(model.encoderTransformerCache.allSatisfy { $0.offset == 0 })
    }
}

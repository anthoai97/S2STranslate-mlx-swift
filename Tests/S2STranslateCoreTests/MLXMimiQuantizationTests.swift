import MLX
import Testing

@testable import S2STranslateCore

@Suite("MLX Mimi Quantization")
struct MLXMimiQuantizationTests {
    @Test("Euclidean codebook owns embedding metadata")
    func euclideanCodebookOwnsEmbeddingMetadata() {
        let codebook = MLXMimiEuclideanCodebook(dimension: 2, codebookSize: 3)

        #expect(codebook.dimension == 2)
        #expect(codebook.codebookSize == 3)
        #expect(codebook.embeddingShape == [3, 2])
        #expect(codebook.clusterUsageShape == [3])
    }

    @Test("quantizer stream encode keeps empty streams empty")
    func quantizerStreamEncodeKeepsEmptyStreamsEmpty() {
        let quantizer = MLXMimiSplitResidualVectorQuantizer(
            dimension: 2,
            inputDimension: 2,
            outputDimension: 2,
            codebookCount: 2,
            bins: 3
        )

        let output = quantizer.encode(MLXMimiStreamArray())

        #expect(output.isEmpty)
    }

    @Test("vector quantization selects nearest deterministic codebook entries")
    func vectorQuantizationSelectsNearestDeterministicCodebookEntries() {
        let quantizer = MLXMimiVectorQuantization(dimension: 2, codebookSize: 3)
        quantizer.codebook.setEmbeddingForTesting([
            [0, 0],
            [1, 0],
            [0, 1],
        ])
        let input = MLXArray([
            Float(0.9), Float(0.1),
            Float(0.1), Float(0.8),
        ]).reshaped([1, 2, 2])

        let codes = quantizer.encode(input)

        #expect(codes.shape == [1, 2])
        #expect(codes.asArray(Int32.self).map(Int.init) == [1, 2])
    }

    @Test("split residual quantizer owns forced projection and codebook topology")
    func splitResidualQuantizerOwnsForcedProjectionAndCodebookTopology() {
        let quantizer = MLXMimiSplitResidualVectorQuantizer(
            dimension: 256,
            inputDimension: 512,
            outputDimension: 512,
            codebookCount: 16,
            bins: 2_048
        )

        #expect(quantizer.rvqFirst.inputProjection?.weightShape == [256, 1, 512])
        #expect(quantizer.rvqFirst.outputProjection?.weightShape == [512, 1, 256])
        #expect(quantizer.rvqRest.inputProjection?.weightShape == [256, 1, 512])
        #expect(quantizer.rvqRest.outputProjection?.weightShape == [512, 1, 256])
        #expect(quantizer.rvqFirst.vq.layers.count == 1)
        #expect(quantizer.rvqRest.vq.layers.count == 15)
        #expect(quantizer.rvqFirst.vq.layers[0].codebook.embeddingShape == [2_048, 256])
        #expect(quantizer.rvqRest.vq.layers[14].codebook.embeddingShape == [2_048, 256])
    }

    @Test("split residual quantizer decode returns conv layout")
    func splitResidualQuantizerDecodeReturnsConvLayout() {
        let quantizer = MLXMimiSplitResidualVectorQuantizer(
            dimension: 2,
            inputDimension: 4,
            outputDimension: 4,
            codebookCount: 2,
            bins: 3
        )
        let codes = MLXArray(Array(repeating: Int32(0), count: 6)).reshaped([1, 2, 3])

        let decoded = quantizer.decode(codes)

        #expect(decoded.shape == [1, 4, 3])
    }
}

import MLX

final class MLXMimiEuclideanCodebook {
    let epsilon: Float = 1e-5
    let dimension: Int
    let codebookSize: Int
    let embeddingShape: [Int]
    let clusterUsageShape: [Int]
    private var embeddingStorage: MLXArray?
    private var embeddingSumStorage: MLXArray?
    private var clusterUsageStorage: MLXArray?

    init(dimension: Int, codebookSize: Int) {
        self.dimension = dimension
        self.codebookSize = codebookSize
        self.embeddingShape = [codebookSize, dimension]
        self.clusterUsageShape = [codebookSize]
    }

    var embeddingSum: MLXArray {
        get {
            if let embeddingSumStorage { return embeddingSumStorage }
            let embeddingSum = MLXArray.zeros(embeddingShape, type: Float32.self)
            embeddingSumStorage = embeddingSum
            return embeddingSum
        }
        set {
            embeddingSumStorage = newValue
            embeddingStorage = nil
        }
    }

    var clusterUsage: MLXArray {
        get {
            if let clusterUsageStorage { return clusterUsageStorage }
            let clusterUsage = MLXArray.zeros(clusterUsageShape, type: Float32.self)
            clusterUsageStorage = clusterUsage
            return clusterUsage
        }
        set {
            clusterUsageStorage = newValue
            embeddingStorage = nil
        }
    }

    var embedding: MLXArray {
        get {
            if let embeddingStorage { return embeddingStorage }
            let usage = maximum(clusterUsage, epsilon)[0..., .newAxis]
            let embedding = embeddingSum / usage
            embeddingStorage = embedding
            return embedding
        }
        set {
            embeddingStorage = newValue
        }
    }

    func encode(_ x: MLXArray) -> MLXArray {
        let targetShape = Array(x.shape.dropLast())
        let x = x.flattened(end: -2)
        let embedding = self.embedding
        let dotProduct = x.matmul(embedding.swappedAxes(-1, -2))
        let codebookNorm = embedding.square().sum(axis: -1) / 2
        return (codebookNorm - dotProduct).argMin(axis: -1).reshaped(targetShape)
    }

    func decode(_ indexes: MLXArray) -> MLXArray {
        let finalShape = indexes.shape + [dimension]
        return embedding.take(indexes.flattened(), axis: 0).reshaped(finalShape)
    }

    func setEmbeddingForTesting(_ values: [[Float]]) {
        let flat = values.flatMap { $0 }
        embedding = MLXArray(flat).reshaped([values.count, dimension])
    }
}

final class MLXMimiVectorQuantization {
    let codebook: MLXMimiEuclideanCodebook

    init(dimension: Int, codebookSize: Int) {
        self.codebook = MLXMimiEuclideanCodebook(
            dimension: dimension,
            codebookSize: codebookSize
        )
    }

    func encode(_ x: MLXArray) -> MLXArray {
        codebook.encode(x.swappedAxes(-1, -2))
    }

    func decode(_ indexes: MLXArray) -> MLXArray {
        codebook.decode(indexes).swappedAxes(-1, -2)
    }
}

final class MLXMimiResidualVectorQuantization {
    let layers: [MLXMimiVectorQuantization]

    init(codebookCount: Int, dimension: Int, codebookSize: Int) {
        self.layers = (0..<codebookCount).map { _ in
            MLXMimiVectorQuantization(dimension: dimension, codebookSize: codebookSize)
        }
    }

    func encode(_ x: MLXArray) -> MLXArray {
        var codes: [MLXArray] = []
        var residual = x
        for layer in layers {
            let indices = layer.encode(residual)
            let quantized = layer.decode(indices)
            residual = residual - quantized
            codes.append(indices)
        }
        return stacked(codes, axis: 0)
    }

    func decode(_ indexes: MLXArray) -> MLXArray {
        let sequenceLength = indexes.dim(0)
        var quantized = layers[0].decode(indexes[0])
        for index in 1..<sequenceLength {
            quantized = quantized + layers[index].decode(indexes[index])
        }
        return quantized
    }
}

final class MLXMimiQuantizerProjection1d {
    let inputDimension: Int
    let outputDimension: Int
    let weightShape: [Int]
    private var weightStorage: MLXArray?

    init(inputDimension: Int, outputDimension: Int) {
        self.inputDimension = inputDimension
        self.outputDimension = outputDimension
        self.weightShape = [outputDimension, 1, inputDimension]
    }

    var weight: MLXArray {
        get {
            if let weightStorage { return weightStorage }
            let weight = MLXArray.zeros(weightShape, type: Float32.self)
            weightStorage = weight
            return weight
        }
        set {
            weightStorage = newValue
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let matrix = weight[0..., 0, 0...]
        return x.swappedAxes(-1, -2)
            .matmul(matrix.swappedAxes(-1, -2))
            .swappedAxes(-1, -2)
    }

    func setIdentityForTesting() {
        var values = Array(repeating: Float(0), count: outputDimension * inputDimension)
        for index in 0..<min(inputDimension, outputDimension) {
            values[index * inputDimension + index] = 1
        }
        weight = MLXArray(values).reshaped(weightShape)
    }
}

final class MLXMimiResidualVectorQuantizer {
    let dimension: Int
    let inputDimension: Int
    let outputDimension: Int
    let codebookCount: Int
    let bins: Int
    let inputProjection: MLXMimiQuantizerProjection1d?
    let outputProjection: MLXMimiQuantizerProjection1d?
    let vq: MLXMimiResidualVectorQuantization

    init(
        dimension: Int,
        inputDimension: Int?,
        outputDimension: Int?,
        codebookCount: Int,
        bins: Int,
        forceProjection: Bool
    ) {
        let inputDimension = inputDimension ?? dimension
        let outputDimension = outputDimension ?? dimension
        self.dimension = dimension
        self.inputDimension = inputDimension
        self.outputDimension = outputDimension
        self.codebookCount = codebookCount
        self.bins = bins
        self.inputProjection = inputDimension == dimension && !forceProjection
            ? nil
            : MLXMimiQuantizerProjection1d(
                inputDimension: inputDimension,
                outputDimension: dimension
            )
        self.outputProjection = outputDimension == dimension && !forceProjection
            ? nil
            : MLXMimiQuantizerProjection1d(
                inputDimension: dimension,
                outputDimension: outputDimension
            )
        self.vq = MLXMimiResidualVectorQuantization(
            codebookCount: codebookCount,
            dimension: dimension,
            codebookSize: bins
        )
    }

    func encode(_ x: MLXArray) -> MLXArray {
        var x = x
        if let inputProjection {
            x = inputProjection(x)
        }
        return vq.encode(x).swappedAxes(0, 1)
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        let codes = codes.swappedAxes(0, 1)
        var quantized = vq.decode(codes)
        if let outputProjection {
            quantized = outputProjection(quantized)
        }
        return quantized
    }
}

public final class MLXMimiSplitResidualVectorQuantizer {
    public let dimension: Int
    public let inputDimension: Int
    public let outputDimension: Int
    public let codebookCount: Int
    public let bins: Int
    let rvqFirst: MLXMimiResidualVectorQuantizer
    let rvqRest: MLXMimiResidualVectorQuantizer

    nonisolated public init(
        dimension: Int,
        inputDimension: Int,
        outputDimension: Int,
        codebookCount: Int,
        bins: Int
    ) {
        self.dimension = dimension
        self.inputDimension = inputDimension
        self.outputDimension = outputDimension
        self.codebookCount = codebookCount
        self.bins = bins
        self.rvqFirst = MLXMimiResidualVectorQuantizer(
            dimension: dimension,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            codebookCount: 1,
            bins: bins,
            forceProjection: true
        )
        self.rvqRest = MLXMimiResidualVectorQuantizer(
            dimension: dimension,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            codebookCount: codebookCount - 1,
            bins: bins,
            forceProjection: true
        )
    }

    public func encode(_ x: MLXArray) -> MLXArray {
        var codes = rvqFirst.encode(x)
        if codebookCount > 1 {
            let restCodes = rvqRest.encode(x)
            codes = concatenated([codes, restCodes], axis: 1)
        }
        return codes
    }

    public func encode(_ x: MLXMimiStreamArray) -> MLXMimiStreamArray {
        x.map(encode)
    }
}

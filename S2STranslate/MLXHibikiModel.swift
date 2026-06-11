import MLX

final class MLXHibikiEmbedding {
    let vocabSize: Int
    let dimensions: Int
    let weightShape: [Int]
    private var weightStorage: MLXArray?

    init(vocabSize: Int, dimensions: Int) {
        self.vocabSize = vocabSize
        self.dimensions = dimensions
        self.weightShape = [vocabSize, dimensions]
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
}

final class MLXHibikiDepformerSlice {
    let embedding: MLXHibikiEmbedding
    let linearIn: MLXMimiLinear
    let linearOut: MLXMimiLinear
    let norm: MLXMimiLayerNorm
    let transformer: MLXMimiTransformer

    init(
        inputVocabSize: Int,
        outputVocabSize: Int,
        mainTransformerDimension: Int,
        configuration: MLXMimiTransformerConfiguration
    ) {
        self.embedding = MLXHibikiEmbedding(
            vocabSize: inputVocabSize,
            dimensions: configuration.modelDimension
        )
        self.linearIn = MLXMimiLinear(
            inputDimensions: mainTransformerDimension,
            outputDimensions: configuration.modelDimension,
            bias: false
        )
        self.linearOut = MLXMimiLinear(
            inputDimensions: configuration.modelDimension,
            outputDimensions: outputVocabSize,
            bias: false
        )
        self.norm = MLXMimiLayerNorm(dimensions: configuration.modelDimension, epsilon: 1e-5)
        self.transformer = MLXMimiTransformer(configuration)
    }
}

final class MLXHibikiLanguageModel {
    let topology: MLXHibikiModelTopology
    let textEmbedding: MLXHibikiEmbedding
    let audioEmbeddings: [MLXHibikiEmbedding]
    let transformer: MLXMimiTransformer
    let outNorm: MLXMimiTransformerNorm
    let textLinear: MLXMimiLinear
    let depformerSlices: [MLXHibikiDepformerSlice]

    init(topology: MLXHibikiModelTopology) {
        self.topology = topology
        self.textEmbedding = MLXHibikiEmbedding(
            vocabSize: topology.textInputVocabSize,
            dimensions: topology.mainTransformer.modelDimension
        )
        self.audioEmbeddings = (0..<topology.audioCodebookCount).map { _ in
            MLXHibikiEmbedding(
                vocabSize: topology.audioVocabSize,
                dimensions: topology.mainTransformer.modelDimension
            )
        }
        self.transformer = MLXMimiTransformer(topology.mainTransformer)
        self.outNorm = MLXMimiTransformerNorm(
            topology.mainTransformer.norm,
            dimensions: topology.mainTransformer.modelDimension
        )
        self.textLinear = MLXMimiLinear(
            inputDimensions: topology.mainTransformer.modelDimension,
            outputDimensions: topology.textOutputVocabSize,
            bias: false
        )
        self.depformerSlices = (0..<topology.generatedCodebookCount).map { sliceIndex in
            MLXHibikiDepformerSlice(
                inputVocabSize: sliceIndex == 0
                    ? topology.textInputVocabSize
                    : topology.audioVocabSize,
                outputVocabSize: topology.audioVocabSize - 1,
                mainTransformerDimension: topology.mainTransformer.modelDimension,
                configuration: topology.depformerTransformer
            )
        }
    }
}

import Foundation
import MLX
import MLXFast
import MLXNN

public protocol MLXMimiKVCache: AnyObject {
    var offset: Int { get }
    var maxSize: Int? { get }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)
    func reset()
    func createAttentionMask(for x: MLXArray) -> MLXArray?
}

public final class MLXMimiLinear {
    public let inputDimensions: Int
    public let outputDimensions: Int
    public let weightShape: [Int]
    public let biasShape: [Int]?
    private var weightStorage: MLXArray?
    private var biasStorage: MLXArray?

    public init(inputDimensions: Int, outputDimensions: Int, bias: Bool = true) {
        self.inputDimensions = inputDimensions
        self.outputDimensions = outputDimensions
        self.weightShape = [outputDimensions, inputDimensions]
        self.biasShape = bias ? [outputDimensions] : nil
    }

    public var weight: MLXArray {
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

    public var bias: MLXArray? {
        get {
            guard let biasShape else { return nil }
            if let biasStorage { return biasStorage }
            let bias = MLXArray.zeros(biasShape, type: Float32.self)
            biasStorage = bias
            return bias
        }
        set {
            biasStorage = newValue
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var output = x.matmul(weight.swappedAxes(-1, -2))
        if let bias {
            output = output + bias
        }
        return output
    }
}

final class MLXMimiLayerNorm {
    let dimensions: Int
    let epsilon: Float
    let weightShape: [Int]
    let biasShape: [Int]?
    private var weightStorage: MLXArray?
    private var biasStorage: MLXArray?

    init(dimensions: Int, epsilon: Float = 1e-5) {
        self.dimensions = dimensions
        self.epsilon = epsilon
        self.weightShape = [dimensions]
        self.biasShape = [dimensions]
    }

    var weight: MLXArray {
        get {
            if let weightStorage { return weightStorage }
            let weight = MLXArray.ones(weightShape)
            weightStorage = weight
            return weight
        }
        set {
            weightStorage = newValue
        }
    }

    var bias: MLXArray {
        get {
            if let biasStorage { return biasStorage }
            let bias = MLXArray.zeros(biasShape!)
            biasStorage = bias
            return bias
        }
        set {
            biasStorage = newValue
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x, weight: weight, bias: bias, eps: epsilon)
    }
}

final class MLXMimiRMSNorm {
    let dimensions: Int
    let epsilon: Float
    let weightShape: [Int]
    private var weightStorage: MLXArray?

    init(dimensions: Int, epsilon: Float = 1e-8) {
        self.dimensions = dimensions
        self.epsilon = epsilon
        self.weightShape = [dimensions]
    }

    var weight: MLXArray {
        get {
            if let weightStorage { return weightStorage }
            let weight = MLXArray.ones(weightShape)
            weightStorage = weight
            return weight
        }
        set {
            weightStorage = newValue
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: epsilon)
    }
}

final class MLXMimiTransformerNorm {
    let kind: MLXMimiNorm
    let layerNorm: MLXMimiLayerNorm?
    let rmsNorm: MLXMimiRMSNorm?

    init(_ kind: MLXMimiNorm, dimensions: Int) {
        self.kind = kind
        switch kind {
        case .layerNorm:
            self.layerNorm = MLXMimiLayerNorm(dimensions: dimensions, epsilon: 1e-5)
            self.rmsNorm = nil
        case .rmsNorm:
            self.layerNorm = nil
            self.rmsNorm = MLXMimiRMSNorm(dimensions: dimensions, epsilon: 1e-8)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        switch kind {
        case .layerNorm:
            return layerNorm!(x)
        case .rmsNorm:
            return rmsNorm!(x)
        }
    }
}

final class MLXMimiMlpNoGating {
    let linear1: MLXMimiLinear
    let linear2: MLXMimiLinear

    init(_ configuration: MLXMimiTransformerConfiguration) {
        self.linear1 = MLXMimiLinear(
            inputDimensions: configuration.modelDimension,
            outputDimensions: configuration.feedForwardDimension,
            bias: configuration.feedForwardBias
        )
        self.linear2 = MLXMimiLinear(
            inputDimensions: configuration.feedForwardDimension,
            outputDimensions: configuration.modelDimension,
            bias: configuration.feedForwardBias
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(MLXNN.geluApproximate(linear1(x)))
    }
}

final class MLXMimiMlpGating {
    let linearIn: MLXMimiLinear
    let linearOut: MLXMimiLinear
    let hiddenDimension: Int

    init(_ configuration: MLXMimiTransformerConfiguration) {
        self.hiddenDimension = configuration.feedForwardDimension == 4 * configuration.modelDimension
            ? 11 * configuration.modelDimension / 4
            : 2 * configuration.feedForwardDimension / 3
        self.linearIn = MLXMimiLinear(
            inputDimensions: configuration.modelDimension,
            outputDimensions: 2 * hiddenDimension,
            bias: configuration.feedForwardBias
        )
        self.linearOut = MLXMimiLinear(
            inputDimensions: hiddenDimension,
            outputDimensions: configuration.modelDimension,
            bias: configuration.feedForwardBias
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x = linearIn(x)
        let batch = x.dim(0)
        let time = x.dim(1)
        let reshaped = x.reshaped(batch, time, 2, -1)
        return linearOut(MLXNN.silu(reshaped[0..., 0..., 0]) * reshaped[0..., 0..., 1])
    }
}

final class MLXMimiFeedForward {
    let gating: Bool
    let gated: MLXMimiMlpGating?
    let ungated: MLXMimiMlpNoGating?

    init(_ configuration: MLXMimiTransformerConfiguration) {
        self.gating = configuration.gating
        if configuration.gating {
            self.gated = MLXMimiMlpGating(configuration)
            self.ungated = nil
        } else {
            self.gated = nil
            self.ungated = MLXMimiMlpNoGating(configuration)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let gated {
            return gated(x)
        }
        return ungated!(x)
    }
}

final class MLXMimiLayerScale {
    let dimension: Int
    let initialValue: Float
    let scaleShape: [Int]
    private var scaleStorage: MLXArray?

    init(dimension: Int, initialValue: Float) {
        self.dimension = dimension
        self.initialValue = initialValue
        self.scaleShape = [dimension]
    }

    var scale: MLXArray {
        get {
            if let scaleStorage { return scaleStorage }
            let scale = MLXArray.ones(scaleShape) * initialValue
            scaleStorage = scale
            return scale
        }
        set {
            scaleStorage = newValue
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * scale
    }
}

public final class MLXMimiRotatingKVCache: MLXMimiKVCache {
    public let maxSize: Int?
    public private(set) var offset = 0
    private var keysStorage: MLXArray?
    private var valuesStorage: MLXArray?
    private let batchSize: Int
    private let headCount: Int
    private let headDimension: Int

    public init(batchSize: Int, headCount: Int, maxSize: Int, headDimension: Int) {
        self.batchSize = batchSize
        self.headCount = headCount
        self.maxSize = maxSize
        self.headDimension = headDimension
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let time = keys.dim(2)
        guard let maxSize, time <= maxSize else {
            fatalError("Mimi rotating KV cache update exceeds context size")
        }

        if keysStorage == nil {
            keysStorage = MLXArray.zeros(
                [batchSize, headCount, maxSize, headDimension],
                type: Float32.self
            )
            valuesStorage = MLXArray.zeros(
                [batchSize, headCount, maxSize, headDimension],
                type: Float32.self
            )
        }

        let currentOffset = offset % maxSize
        let firstEnd = min(maxSize, currentOffset + time)
        keysStorage![0..., 0..., currentOffset..<firstEnd] =
            keys[0..., 0..., 0..<(firstEnd - currentOffset)]
        valuesStorage![0..., 0..., currentOffset..<firstEnd] =
            values[0..., 0..., 0..<(firstEnd - currentOffset)]

        let remaining = time - firstEnd + currentOffset
        if remaining > 0 {
            keysStorage![0..., 0..., 0..<remaining] =
                keys[0..., 0..., (firstEnd - currentOffset)...]
            valuesStorage![0..., 0..., 0..<remaining] =
                values[0..., 0..., (firstEnd - currentOffset)...]
        }

        offset += time
        return (keysStorage!, valuesStorage!)
    }

    public func reset() {
        offset = 0
    }

    func advanceOffsetForTesting(by time: Int) {
        offset += time
    }

    public func createAttentionMask(for x: MLXArray) -> MLXArray? {
        guard let maxSize else { return nil }

        let time = x.dim(1)
        let finalOffset = offset + time
        let finalOffsetMod = finalOffset % maxSize
        var rinds = Array(repeating: Int32(finalOffset + 1), count: maxSize)
        for index in 0..<finalOffsetMod {
            rinds[index] = Int32(finalOffset + index - finalOffsetMod)
        }
        if finalOffsetMod != finalOffset {
            for index in finalOffsetMod..<rinds.count {
                rinds[index] = Int32(finalOffset + index - finalOffsetMod - rinds.count)
            }
        }

        let linds = MLXArray(Int32(offset)..<Int32(offset + time))
        let mask = linds[0..., .newAxis] .< MLXArray(rinds)[.newAxis]
        return (mask * Float32(-1e9)).asType(x.dtype)
    }
}

public final class MLXMimiKVCacheSimple: MLXMimiKVCache {
    public let headDimension: Int
    public let headCount: Int
    public let maxSize: Int? = nil
    public private(set) var offset = 0
    private var keysStorage: MLXArray?
    private var valuesStorage: MLXArray?

    public init(headDimension: Int, headCount: Int) {
        self.headDimension = headDimension
        self.headCount = headCount
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        offset += keys.dim(2)
        if let keysStorage, let valuesStorage {
            self.keysStorage = concatenated([keysStorage, keys], axis: 2)
            self.valuesStorage = concatenated([valuesStorage, values], axis: 2)
        } else {
            self.keysStorage = keys
            self.valuesStorage = values
        }
        return (keysStorage!, valuesStorage!)
    }

    public func reset() {
        offset = 0
        keysStorage = nil
        valuesStorage = nil
    }

    func advanceOffsetForTesting(by time: Int) {
        offset += time
    }

    public func createAttentionMask(for x: MLXArray) -> MLXArray? {
        let time = x.dim(1)
        guard time > 1 else { return nil }

        let rinds = MLXArray(Int32(0)..<Int32(offset + time))
        let linds = offset != 0 ? MLXArray(Int32(offset)..<Int32(offset + time)) : rinds
        let mask = linds[0..., .newAxis] .< rinds[.newAxis]
        return (mask * Float32(-1e9)).asType(x.dtype)
    }
}

final class MLXMimiAttention {
    let configuration: MLXMimiTransformerConfiguration
    let scale: Float
    let qkvProjection: MLXMimiLinear
    let outputProjection: MLXMimiLinear
    let rope: MLXNN.RoPE?
    let keyValueHeadCount: Int

    init(_ configuration: MLXMimiTransformerConfiguration) {
        self.configuration = configuration
        self.scale = 1.0 / sqrt(Float(configuration.headDimension))
        self.keyValueHeadCount = configuration.headCount / configuration.kvRepeat
        let outputDimension = configuration.modelDimension
            + 2 * keyValueHeadCount * configuration.headDimension
        self.qkvProjection = MLXMimiLinear(
            inputDimensions: configuration.modelDimension,
            outputDimensions: outputDimension,
            bias: configuration.attentionBias
        )
        self.outputProjection = MLXMimiLinear(
            inputDimensions: configuration.modelDimension,
            outputDimensions: configuration.modelDimension,
            bias: configuration.attentionBias
        )
        switch configuration.positionalEmbedding {
        case .none:
            self.rope = nil
        case .rope:
            self.rope = MLXNN.RoPE(
                dimensions: configuration.headDimension,
                traditional: true,
                base: Float(configuration.maxPeriod)
            )
        case .ropeConcat:
            self.rope = MLXNN.RoPE(
                dimensions: configuration.headDimension,
                traditional: false,
                base: Float(configuration.maxPeriod)
            )
        }
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: MLXMimiKVCache) -> MLXArray {
        let batch = x.dim(0)
        let time = x.dim(1)
        let hidden = x.dim(2)
        let headDimension = configuration.headDimension
        let queryDimension = configuration.headCount * headDimension
        let keyValueDimension = keyValueHeadCount * headDimension
        let qkv = qkvProjection(x)
        var query = qkv[0..., 0..., 0..<queryDimension]
            .reshaped(batch, time, configuration.headCount, headDimension)
            .transposed(0, 2, 1, 3)
        var key = qkv[0..., 0..., queryDimension..<(queryDimension + keyValueDimension)]
            .reshaped(batch, time, keyValueHeadCount, headDimension)
            .transposed(0, 2, 1, 3)
        var value = qkv[0..., 0..., (queryDimension + keyValueDimension)...]
            .reshaped(batch, time, keyValueHeadCount, headDimension)
            .transposed(0, 2, 1, 3)

        if let rope {
            let offset = cache.offset
            query = rope(query, offset: offset)
            key = rope(key, offset: offset)
        }

        (key, value) = cache.update(keys: key, values: value)

        let keyLength = key.dim(2)
        let targetLength = time + min(configuration.context, keyLength - time)
        if targetLength < keyLength {
            let offset = keyLength - targetLength
            key = key[0..., 0..., offset...]
            value = value[0..., 0..., offset...]
        }

        var mask = mask
        if let currentMask = mask {
            let maskLength = currentMask.dim(-1)
            if key.dim(2) < maskLength {
                let offset = maskLength - key.dim(2)
                mask = currentMask[0..., offset...]
            }
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: query,
            keys: key,
            values: value,
            scale: scale,
            mask: mask
        )
        return outputProjection(output.transposed(0, 2, 1, 3).reshaped(batch, time, hidden))
    }
}

final class MLXMimiTransformerLayer {
    let feedForward: MLXMimiFeedForward
    let norm1: MLXMimiTransformerNorm
    let norm2: MLXMimiTransformerNorm
    let layerScale1: MLXMimiLayerScale?
    let layerScale2: MLXMimiLayerScale?
    let selfAttention: MLXMimiAttention

    init(_ configuration: MLXMimiTransformerConfiguration) {
        self.feedForward = MLXMimiFeedForward(configuration)
        self.norm1 = MLXMimiTransformerNorm(
            configuration.norm,
            dimensions: configuration.modelDimension
        )
        self.norm2 = MLXMimiTransformerNorm(
            configuration.norm,
            dimensions: configuration.modelDimension
        )
        self.selfAttention = MLXMimiAttention(configuration)
        if let layerScale = configuration.layerScale {
            self.layerScale1 = MLXMimiLayerScale(
                dimension: configuration.modelDimension,
                initialValue: layerScale
            )
            self.layerScale2 = MLXMimiLayerScale(
                dimension: configuration.modelDimension,
                initialValue: layerScale
            )
        } else {
            self.layerScale1 = nil
            self.layerScale2 = nil
        }
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: MLXMimiKVCache) -> MLXArray {
        var residual = x
        var x = selfAttention(norm1(x), mask: mask, cache: cache)
        if let layerScale1 {
            x = layerScale1(x)
        }
        x = residual + x

        residual = x
        x = feedForward(norm2(x))
        if let layerScale2 {
            x = layerScale2(x)
        }
        return residual + x
    }
}

public final class MLXMimiTransformer {
    public let configuration: MLXMimiTransformerConfiguration
    let layers: [MLXMimiTransformerLayer]

    public init(_ configuration: MLXMimiTransformerConfiguration) {
        self.configuration = configuration
        self.layers = (0..<configuration.layerCount).map { _ in
            MLXMimiTransformerLayer(configuration)
        }
    }

    public func callAsFunction(_ x: MLXArray, cache: [MLXMimiKVCache]) -> MLXArray {
        var x = x
        let mask = cache.first?.createAttentionMask(for: x)
        for (layer, layerCache) in zip(layers, cache) {
            x = layer(x, mask: mask, cache: layerCache)
        }
        return x
    }

    public func makeCache(batchSize: Int) -> [MLXMimiKVCache] {
        let keyValueHeadCount = configuration.headCount / configuration.kvRepeat
        return (0..<configuration.layerCount).map { _ in
            if configuration.usesRotatingKVCache {
                return MLXMimiRotatingKVCache(
                    batchSize: batchSize,
                    headCount: keyValueHeadCount,
                    maxSize: configuration.context,
                    headDimension: configuration.headDimension
                )
            }
            return MLXMimiKVCacheSimple(
                headDimension: configuration.headDimension,
                headCount: keyValueHeadCount
            )
        }
    }
}

public final class MLXMimiProjectedTransformer {
    public let configuration: MLXMimiTransformerConfiguration
    public let inputDimension: Int
    public let outputDimensions: [Int]
    public let convLayout: Bool
    let transformer: MLXMimiTransformer
    let inputProjection: MLXMimiLinear?
    let outputProjections: [MLXMimiLinear?]

    nonisolated public init(
        _ configuration: MLXMimiTransformerConfiguration,
        inputDimension: Int,
        outputDimensions: [Int]
    ) {
        self.configuration = configuration
        self.inputDimension = inputDimension
        self.outputDimensions = outputDimensions
        self.convLayout = configuration.convLayout
        self.transformer = MLXMimiTransformer(configuration)
        self.inputProjection = inputDimension == configuration.modelDimension
            ? nil
            : MLXMimiLinear(
                inputDimensions: inputDimension,
                outputDimensions: configuration.modelDimension,
                bias: false
            )
        self.outputProjections = outputDimensions.map { outputDimension in
            outputDimension == configuration.modelDimension
                ? nil
                : MLXMimiLinear(
                    inputDimensions: configuration.modelDimension,
                    outputDimensions: outputDimension,
                    bias: false
                )
        }
    }

    public func callAsFunction(_ x: MLXArray, cache: [MLXMimiKVCache]) -> [MLXArray] {
        var x = x
        if convLayout {
            x = x.swappedAxes(1, 2)
        }
        if let inputProjection {
            x = inputProjection(x)
        }

        x = transformer(x, cache: cache)

        return outputProjections.map { outputProjection in
            var output = outputProjection?(x) ?? x
            if convLayout {
                output = output.swappedAxes(1, 2)
            }
            return output
        }
    }

    public func step(_ x: MLXMimiStreamArray, cache: [MLXMimiKVCache]) -> MLXMimiStreamArray {
        x.map { self($0, cache: cache)[0] }
    }

    public func makeCache(batchSize: Int) -> [MLXMimiKVCache] {
        transformer.makeCache(batchSize: batchSize)
    }

    public func expectedOutputShapes(inputShape: [Int]) -> [[Int]] {
        outputDimensions.map { outputDimension in
            if convLayout {
                return [inputShape[0], outputDimension, inputShape[2]]
            }
            return [inputShape[0], inputShape[1], outputDimension]
        }
    }

    public static func resetCache(_ cache: [MLXMimiKVCache]) {
        cache.forEach { $0.reset() }
    }
}

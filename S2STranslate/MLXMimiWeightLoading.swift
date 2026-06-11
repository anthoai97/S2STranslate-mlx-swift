import Foundation
import MLX
import MLXNN

public enum MLXMimiWeightLoadError: Error, Equatable {
    case missingArtifactRole(String)
    case missingArtifactFile(String)
    case loadFailed(String)
    case missingKey(String)
    case unexpectedKey(String)
    case incompatibleShape(key: String, expected: [Int], actual: [Int])

    public var userVisibleMessage: String {
        switch self {
        case let .missingArtifactRole(role):
            "Mimi weight artifact role missing: \(role)"
        case let .missingArtifactFile(fileName):
            "Mimi weight artifact file missing: \(fileName)"
        case let .loadFailed(message):
            "Mimi weight load failed: \(message)"
        case let .missingKey(key):
            "Mimi weight key missing: \(key)"
        case let .unexpectedKey(key):
            "Mimi weight key unexpected: \(key)"
        case let .incompatibleShape(key, expected, actual):
            "Mimi weight shape incompatible for \(key): expected \(expected), got \(actual)"
        }
    }
}

public struct LoadedMLXMimiWeights {
    public var artifact: PreparedModelArtifact
    public var mappedTensors: [String: MLXMimiWeightTensor]

    public func moduleParameters() throws -> ModuleParameters {
        var arrays: [String: MLXArray] = [:]
        for (key, tensor) in mappedTensors {
            guard let array = tensor.array else {
                throw MLXMimiWeightLoadError.loadFailed(
                    "Mapped tensor has no MLX array payload: \(key)"
                )
            }
            arrays[key] = array
        }
        return ModuleParameters.unflattened(arrays)
    }
}

public struct MLXMimiWeightTensor {
    public var shape: [Int]
    public var array: MLXArray?

    public init(shape: [Int], array: MLXArray? = nil) {
        self.shape = shape
        self.array = array
    }
}

public enum MLXMimiWeightKeyMapper {
    public static func map(_ sourceKey: String) -> String {
        var key = sourceKey

        if key.hasPrefix("encoder.model") {
            key = key.replacingOccurrences(of: "encoder.model.", with: "encoder.")
        }
        if key.hasPrefix("decoder.model") {
            key = key.replacingOccurrences(of: "decoder.model.", with: "decoder.")
        }
        if key.hasSuffix(".in_proj_weight") {
            key = key.replacingOccurrences(of: ".in_proj_weight", with: ".in_proj.weight")
        }
        if key.hasSuffix(".linear1.weight") {
            key = key.replacingOccurrences(of: ".linear1.weight", with: ".gating.linear1.weight")
        }
        if key.hasSuffix(".linear2.weight") {
            key = key.replacingOccurrences(of: ".linear2.weight", with: ".gating.linear2.weight")
        }
        while key.contains(".conv.conv.") {
            key = key.replacingOccurrences(of: ".conv.conv.", with: ".conv.")
        }
        while key.contains(".convtr.convtr.") {
            key = key.replacingOccurrences(of: ".convtr.convtr.", with: ".convtr.")
        }
        key = key.replacingOccurrences(of: "_transformer.transformer.", with: "_transformer.")

        for (layerIndex, decoderIndex) in [2, 5, 8, 11].enumerated() {
            key = key.replacingOccurrences(
                of: "decoder.\(decoderIndex).",
                with: "decoder.layers.\(layerIndex).upsample."
            )
            key = key.replacingOccurrences(
                of: "decoder.\(decoderIndex + 1).",
                with: "decoder.layers.\(layerIndex).residuals.0."
            )
        }

        for (layerIndex, encoderIndex) in [1, 4, 7, 10].enumerated() {
            key = key.replacingOccurrences(
                of: "encoder.\(encoderIndex).",
                with: "encoder.layers.\(layerIndex).residuals.0."
            )
            key = key.replacingOccurrences(
                of: "encoder.\(encoderIndex + 2).",
                with: "encoder.layers.\(layerIndex).downsample."
            )
        }

        key = key.replacingOccurrences(of: "decoder.0.", with: "decoder.init_conv1d.")
        key = key.replacingOccurrences(of: "decoder.14.", with: "decoder.final_conv1d.")
        key = key.replacingOccurrences(of: "encoder.0.", with: "encoder.init_conv1d.")
        key = key.replacingOccurrences(of: "encoder.14.", with: "encoder.final_conv1d.")
        key = key.replacingOccurrences(of: ".block.1.", with: ".block.0.")
        key = key.replacingOccurrences(of: ".block.3.", with: ".block.1.")

        return key
    }
}

public enum MLXMimiTensorLayoutMapper {
    public static func map(_ tensor: MLXMimiWeightTensor, mappedKey: String) -> MLXMimiWeightTensor {
        if mappedKey.hasSuffix(".conv.weight")
            || mappedKey.hasSuffix(".output_proj.weight")
            || mappedKey.hasSuffix(".input_proj.weight")
        {
            return MLXMimiWeightTensor(
                shape: swappedLastTwoAxes(tensor.shape),
                array: tensor.array?.swappedAxes(-1, -2)
            )
        }

        if mappedKey.hasSuffix(".convtr.weight") {
            return MLXMimiWeightTensor(
                shape: transposedShape(tensor.shape, axes: [1, 2, 0]),
                array: tensor.array?.transposed(axes: [1, 2, 0])
            )
        }

        return tensor
    }

    private static func swappedLastTwoAxes(_ shape: [Int]) -> [Int] {
        guard shape.count >= 2 else { return shape }

        var shape = shape
        shape.swapAt(shape.count - 1, shape.count - 2)
        return shape
    }

    private static func transposedShape(_ shape: [Int], axes: [Int]) -> [Int] {
        guard shape.count == axes.count else { return shape }
        return axes.map { shape[$0] }
    }
}

public struct MLXMimiWeightLoader {
    public typealias ArrayReader = (URL) throws -> [String: MLXMimiWeightTensor]

    private let expectedShapes: [String: [Int]]
    private let allowsUnexpectedKeys: Bool
    private let arrayReader: ArrayReader

    public init(
        expectedShapes: [String: [Int]] = [:],
        allowsUnexpectedKeys: Bool = true,
        arrayReader: @escaping ArrayReader = { url in
            try loadArrays(url: url).mapValues { MLXMimiWeightTensor(shape: $0.shape, array: $0) }
        }
    ) {
        self.expectedShapes = expectedShapes
        self.allowsUnexpectedKeys = allowsUnexpectedKeys
        self.arrayReader = arrayReader
    }

    public func load(from artifacts: PreparedModelArtifacts) throws -> LoadedMLXMimiWeights {
        guard let artifact = artifacts.files.first(where: { $0.role == "mimiWeights" }) else {
            throw MLXMimiWeightLoadError.missingArtifactRole("mimiWeights")
        }

        guard FileManager.default.fileExists(atPath: artifact.location) else {
            throw MLXMimiWeightLoadError.missingArtifactFile(artifact.fileName)
        }

        let sourceTensors: [String: MLXMimiWeightTensor]
        do {
            sourceTensors = try arrayReader(URL(fileURLWithPath: artifact.location))
        } catch {
            throw MLXMimiWeightLoadError.loadFailed(String(describing: error))
        }

        var mappedTensors: [String: MLXMimiWeightTensor] = [:]
        for (sourceKey, sourceTensor) in sourceTensors {
            let mappedKey = MLXMimiWeightKeyMapper.map(sourceKey)
            mappedTensors[mappedKey] = MLXMimiTensorLayoutMapper.map(sourceTensor, mappedKey: mappedKey)
        }

        try validate(mappedTensors)

        return LoadedMLXMimiWeights(
            artifact: artifact,
            mappedTensors: mappedTensors
        )
    }

    private func validate(_ mappedTensors: [String: MLXMimiWeightTensor]) throws {
        for (key, expectedShape) in expectedShapes {
            guard let tensor = mappedTensors[key] else {
                throw MLXMimiWeightLoadError.missingKey(key)
            }

            guard tensor.shape == expectedShape else {
                throw MLXMimiWeightLoadError.incompatibleShape(
                    key: key,
                    expected: expectedShape,
                    actual: tensor.shape
                )
            }
        }

        guard !allowsUnexpectedKeys else { return }

        for key in mappedTensors.keys where expectedShapes[key] == nil {
            throw MLXMimiWeightLoadError.unexpectedKey(key)
        }
    }
}

import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

public struct MLXMimiRuntimeConfiguration: Equatable, Sendable {
    public var sampleRate: Int
    public var frameRate: Double
    public var codebookCount: Int
    public var samplesPerFrame: Int

    nonisolated public init(
        sampleRate: Int,
        frameRate: Double,
        codebookCount: Int,
        samplesPerFrame: Int
    ) {
        self.sampleRate = sampleRate
        self.frameRate = frameRate
        self.codebookCount = codebookCount
        self.samplesPerFrame = samplesPerFrame
    }

    public static let mimi202407 = MLXMimiRuntimeConfiguration(
        sampleRate: 24_000,
        frameRate: 12.5,
        codebookCount: 16,
        samplesPerFrame: 1_920
    )
}

public enum MimiRuntimeError: Error, Equatable, Sendable {
    case missingArtifactRole(String)
    case missingArtifactFile(String)
    case incompatibleConfiguration(String)
    case loadFailed(String)

    public var userVisibleMessage: String {
        switch self {
        case let .missingArtifactRole(role):
            "Mimi runtime artifact role missing: \(role)"
        case let .missingArtifactFile(fileName):
            "Mimi runtime artifact file missing: \(fileName)"
        case let .incompatibleConfiguration(message):
            "Mimi runtime configuration incompatible: \(message)"
        case let .loadFailed(message):
            "Mimi runtime load failed: \(message)"
        }
    }
}

public final class MLXMimiRuntime: @unchecked Sendable {
    public let artifact: PreparedModelArtifact
    public let configuration: MLXMimiRuntimeConfiguration

    public init(
        artifact: PreparedModelArtifact,
        configuration: MLXMimiRuntimeConfiguration = .mimi202407
    ) {
        self.artifact = artifact
        self.configuration = configuration
    }

    public func resetEncodeState() {}

    public func resetDecodeState() {}

    public func resetState() {
        resetEncodeState()
        resetDecodeState()
    }
}

public struct MLXMimiRuntimeLoader: Sendable {
    private let configuration: MLXMimiRuntimeConfiguration

    nonisolated public init(configuration: MLXMimiRuntimeConfiguration = .mimi202407) {
        self.configuration = configuration
    }

    public func load(from artifacts: PreparedModelArtifacts) throws -> MLXMimiRuntime {
        guard let artifact = artifacts.files.first(where: { $0.role == "mimiWeights" }) else {
            throw MimiRuntimeError.missingArtifactRole("mimiWeights")
        }

        guard FileManager.default.fileExists(atPath: artifact.location) else {
            throw MimiRuntimeError.missingArtifactFile(artifact.fileName)
        }

        return MLXMimiRuntime(
            artifact: artifact,
            configuration: configuration
        )
    }
}

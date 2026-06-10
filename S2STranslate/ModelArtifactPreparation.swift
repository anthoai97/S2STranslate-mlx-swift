import Foundation

public struct ModelRuntimeManifest: Equatable, Sendable {
    public var modelRepo: String
    public var revision: String
    public var requiredFiles: [ModelArtifactRequirement]

    nonisolated public init(
        modelRepo: String,
        revision: String,
        requiredFiles: [ModelArtifactRequirement]
    ) {
        self.modelRepo = modelRepo
        self.revision = revision
        self.requiredFiles = requiredFiles
    }

    public static func decode(from data: Data) throws -> ModelRuntimeManifest {
        let decoded = try JSONDecoder().decode(ModelRuntimeManifestDTO.self, from: data)
        return ModelRuntimeManifest(
            modelRepo: decoded.modelRepo,
            revision: decoded.revision,
            requiredFiles: decoded.requiredFiles
                .map { ModelArtifactRequirement(role: $0.key, fileName: $0.value) }
                .sorted { $0.role < $1.role }
        )
    }

    public static let hibikiQ4Default = ModelRuntimeManifest(
        modelRepo: "anquachdev/hbk-zero-3b-mlx-q4",
        revision: "558daadd9272df9432642783b57b02756ff34d5b",
        requiredFiles: [
            ModelArtifactRequirement(role: "architectureConfig", fileName: "config.json"),
            ModelArtifactRequirement(role: "hibikiWeights", fileName: "hibiki.q4.safetensors"),
            ModelArtifactRequirement(role: "mimiWeights", fileName: "mimi-pytorch-e351c8d8@125.safetensors"),
            ModelArtifactRequirement(role: "tokenizer", fileName: "tokenizer_spm_48k_multi6_2.model"),
        ]
    )
}

private struct ModelRuntimeManifestDTO: Decodable {
    var modelRepo: String
    var revision: String
    var requiredFiles: [String: String]
}

public struct ModelArtifactRequirement: Equatable, Sendable {
    public var role: String
    public var fileName: String

    nonisolated public init(role: String, fileName: String) {
        self.role = role
        self.fileName = fileName
    }
}

public struct ModelArtifactHandle: Equatable, Sendable {
    public var fileName: String
    public var location: String
    public var byteCount: Int64
    public var integrity: ModelArtifactIntegrity

    nonisolated public init(
        fileName: String,
        location: String,
        byteCount: Int64,
        integrity: ModelArtifactIntegrity = .valid
    ) {
        self.fileName = fileName
        self.location = location
        self.byteCount = byteCount
        self.integrity = integrity
    }
}

public enum ModelArtifactIntegrity: Equatable, Sendable {
    case valid
    case corrupt
    case incompatible
    case tooLarge
}

public enum ModelArtifactSource: Equatable, Sendable {
    case cache
    case prepared
}

public struct PreparedModelArtifact: Equatable, Sendable {
    public var role: String
    public var fileName: String
    public var location: String
    public var source: ModelArtifactSource
}

public struct PreparedModelArtifacts: Equatable, Sendable {
    public var manifest: ModelRuntimeManifest
    public var files: [PreparedModelArtifact]
}

public enum ModelArtifactPreparationError: Error, Equatable, Sendable {
    case missing(String)
    case inaccessible(String)
    case corrupt(String)
    case incompatible(String)
    case tooLarge(String)

    public var userVisibleMessage: String {
        switch self {
        case let .missing(fileName):
            "Model artifact missing: \(fileName)"
        case let .inaccessible(fileName):
            "Model artifact inaccessible: \(fileName)"
        case let .corrupt(fileName):
            "Model artifact corrupt: \(fileName)"
        case let .incompatible(fileName):
            "Model artifact incompatible: \(fileName)"
        case let .tooLarge(fileName):
            "Model artifact too large for this device: \(fileName)"
        }
    }
}

public protocol ModelArtifactProviding: Sendable {
    func cachedArtifact(named fileName: String) async throws -> ModelArtifactHandle?
    func prepareArtifact(
        named fileName: String,
        from modelRepo: String,
        revision: String
    ) async throws -> ModelArtifactHandle
}

public struct ModelArtifactPreparationResult: Equatable, Sendable {
    public var progressEvents: [Double]
    public var artifacts: PreparedModelArtifacts?
    public var failure: ModelArtifactPreparationError?

    public var succeeded: Bool {
        artifacts != nil && failure == nil
    }
}

public struct ModelArtifactPreparer: Sendable {
    private let manifest: ModelRuntimeManifest
    private let provider: any ModelArtifactProviding

    nonisolated public init(manifest: ModelRuntimeManifest, provider: any ModelArtifactProviding) {
        self.manifest = manifest
        self.provider = provider
    }

    public func prepare() async -> ModelArtifactPreparationResult {
        var progressEvents: [Double] = [0]
        var preparedFiles: [PreparedModelArtifact] = []
        let requiredFiles = manifest.requiredFiles

        guard !requiredFiles.isEmpty else {
            return ModelArtifactPreparationResult(
                progressEvents: [1],
                artifacts: PreparedModelArtifacts(manifest: manifest, files: []),
                failure: nil
            )
        }

        for (index, requirement) in requiredFiles.enumerated() {
            do {
                let artifact = try await resolveArtifact(for: requirement)
                preparedFiles.append(artifact)
                progressEvents.append(Double(index + 1) / Double(requiredFiles.count))
            } catch let error as ModelArtifactPreparationError {
                return ModelArtifactPreparationResult(
                    progressEvents: progressEvents,
                    artifacts: nil,
                    failure: error
                )
            } catch {
                return ModelArtifactPreparationResult(
                    progressEvents: progressEvents,
                    artifacts: nil,
                    failure: .inaccessible(requirement.fileName)
                )
            }
        }

        return ModelArtifactPreparationResult(
            progressEvents: progressEvents,
            artifacts: PreparedModelArtifacts(manifest: manifest, files: preparedFiles),
            failure: nil
        )
    }

    private func resolveArtifact(
        for requirement: ModelArtifactRequirement
    ) async throws -> PreparedModelArtifact {
        if let cached = try await provider.cachedArtifact(named: requirement.fileName) {
            try validate(cached, for: requirement)
            return PreparedModelArtifact(
                role: requirement.role,
                fileName: cached.fileName,
                location: cached.location,
                source: .cache
            )
        }

        let prepared = try await provider.prepareArtifact(
            named: requirement.fileName,
            from: manifest.modelRepo,
            revision: manifest.revision
        )
        try validate(prepared, for: requirement)
        return PreparedModelArtifact(
            role: requirement.role,
            fileName: prepared.fileName,
            location: prepared.location,
            source: .prepared
        )
    }

    private func validate(
        _ handle: ModelArtifactHandle,
        for requirement: ModelArtifactRequirement
    ) throws {
        guard handle.fileName == requirement.fileName else {
            throw ModelArtifactPreparationError.incompatible(requirement.fileName)
        }

        switch handle.integrity {
        case .valid:
            guard handle.byteCount > 0 else {
                throw ModelArtifactPreparationError.corrupt(requirement.fileName)
            }
        case .corrupt:
            throw ModelArtifactPreparationError.corrupt(requirement.fileName)
        case .incompatible:
            throw ModelArtifactPreparationError.incompatible(requirement.fileName)
        case .tooLarge:
            throw ModelArtifactPreparationError.tooLarge(requirement.fileName)
        }
    }
}

public struct ModelArtifactExperimentBackend: ExperimentBackend, Sendable {
    private let preparer: ModelArtifactPreparer
    private let runEventsScript: [ExperimentEvent]

    public init(
        preparer: ModelArtifactPreparer,
        runEvents: [ExperimentEvent] = []
    ) {
        self.preparer = preparer
        self.runEventsScript = runEvents
    }

    public func prepareEvents() async -> [ExperimentEvent] {
        let result = await preparer.prepare()
        var events = result.progressEvents.map(ExperimentEvent.preparationProgress)

        if let failure = result.failure {
            events.append(.failure(failure.userVisibleMessage))
        } else if let artifacts = result.artifacts {
            let cachedCount = artifacts.files.filter { $0.source == .cache }.count
            let preparedCount = artifacts.files.filter { $0.source == .prepared }.count
            events.append(.observation(
                "Prepared \(artifacts.files.count) model artifacts (\(cachedCount) cached, \(preparedCount) prepared)."
            ))
            events.append(.ready)
        }

        return events
    }

    public func runEvents() async -> [ExperimentEvent] {
        runEventsScript
    }
}

public actor DemoModelArtifactProvider: ModelArtifactProviding {
    public init() {}

    public func cachedArtifact(named fileName: String) async throws -> ModelArtifactHandle? {
        nil
    }

    public func prepareArtifact(
        named fileName: String,
        from modelRepo: String,
        revision: String
    ) async throws -> ModelArtifactHandle {
        ModelArtifactHandle(
            fileName: fileName,
            location: "demo-cache://\(modelRepo)/\(revision)/\(fileName)",
            byteCount: 1
        )
    }
}

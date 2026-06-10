import Foundation
import Testing

@testable import S2STranslateCore

@Suite("Model Artifact Preparation")
struct ModelArtifactPreparationTests {
    @Test("manifest decoding produces sorted required artifact roles")
    func manifestDecodingProducesRequiredArtifacts() throws {
        let data = Data("""
        {
          "modelRepo": "anquachdev/hbk-zero-3b-mlx-q4",
          "revision": "abc123",
          "requiredFiles": {
            "tokenizer": "tokenizer.model",
            "architectureConfig": "config.json"
          }
        }
        """.utf8)

        let manifest = try ModelRuntimeManifest.decode(from: data)

        #expect(manifest.modelRepo == "anquachdev/hbk-zero-3b-mlx-q4")
        #expect(manifest.revision == "abc123")
        #expect(manifest.requiredFiles == [
            ModelArtifactRequirement(role: "architectureConfig", fileName: "config.json"),
            ModelArtifactRequirement(role: "tokenizer", fileName: "tokenizer.model"),
        ])
    }

    @Test("cached artifacts are used before first-run preparation")
    func cacheHitDoesNotPrepareArtifact() async {
        let provider = RecordingArtifactProvider(
            cached: [
                "config.json": ModelArtifactHandle(
                    fileName: "config.json",
                    location: "cache://config.json",
                    byteCount: 10
                ),
            ]
        )
        let preparer = ModelArtifactPreparer(
            manifest: ModelRuntimeManifest(
                modelRepo: "repo",
                revision: "rev",
                requiredFiles: [
                    ModelArtifactRequirement(role: "architectureConfig", fileName: "config.json"),
                ]
            ),
            provider: provider
        )

        let result = await preparer.prepare()
        let requests = await provider.requests()

        #expect(result.succeeded)
        #expect(result.artifacts?.files.first?.source == .cache)
        #expect(result.progressEvents == [0, 1])
        #expect(requests == [
            .cached("config.json"),
        ])
    }

    @Test("missing cached artifacts are prepared from the pinned model source")
    func cacheMissPreparesFromPinnedSource() async {
        let provider = RecordingArtifactProvider(
            prepared: [
                "hibiki.q4.safetensors": ModelArtifactHandle(
                    fileName: "hibiki.q4.safetensors",
                    location: "prepared://hibiki.q4.safetensors",
                    byteCount: 20
                ),
            ]
        )
        let preparer = ModelArtifactPreparer(
            manifest: ModelRuntimeManifest(
                modelRepo: "repo",
                revision: "rev",
                requiredFiles: [
                    ModelArtifactRequirement(role: "hibikiWeights", fileName: "hibiki.q4.safetensors"),
                ]
            ),
            provider: provider
        )

        let result = await preparer.prepare()
        let requests = await provider.requests()

        #expect(result.succeeded)
        #expect(result.artifacts?.files.first?.source == .prepared)
        #expect(requests == [
            .cached("hibiki.q4.safetensors"),
            .prepared(fileName: "hibiki.q4.safetensors", repo: "repo", revision: "rev"),
        ])
    }

    @Test(
        "preparation failures are distinct",
        arguments: [
            ModelArtifactPreparationError.missing("config.json"),
            ModelArtifactPreparationError.inaccessible("config.json"),
            ModelArtifactPreparationError.corrupt("config.json"),
            ModelArtifactPreparationError.incompatible("config.json"),
            ModelArtifactPreparationError.tooLarge("config.json"),
        ]
    )
    func preparationFailuresAreDistinct(error: ModelArtifactPreparationError) async {
        let provider = RecordingArtifactProvider(failure: error)
        let preparer = ModelArtifactPreparer(
            manifest: ModelRuntimeManifest(
                modelRepo: "repo",
                revision: "rev",
                requiredFiles: [
                    ModelArtifactRequirement(role: "architectureConfig", fileName: "config.json"),
                ]
            ),
            provider: provider
        )

        let result = await preparer.prepare()

        #expect(result.failure == error)
        #expect(!result.succeeded)
    }

    @Test("artifact backend reports progress and moves Experiment Session to ready")
    @MainActor
    func artifactBackendReportsProgressAndReady() async {
        let provider = RecordingArtifactProvider(
            prepared: [
                "config.json": ModelArtifactHandle(
                    fileName: "config.json",
                    location: "prepared://config.json",
                    byteCount: 10
                ),
            ]
        )
        let backend = ModelArtifactExperimentBackend(
            preparer: ModelArtifactPreparer(
                manifest: ModelRuntimeManifest(
                    modelRepo: "repo",
                    revision: "rev",
                    requiredFiles: [
                        ModelArtifactRequirement(role: "architectureConfig", fileName: "config.json"),
                    ]
                ),
                provider: provider
            )
        )
        let session = ExperimentSession(backend: backend)

        await session.prepare()

        #expect(session.state == .ready)
        #expect(session.observations.progress == 1)
        #expect(session.observations.output == "Prepared 1 model artifacts (0 cached, 1 prepared).")
    }

    @Test("artifact backend surfaces failure details in Experiment Session state")
    @MainActor
    func artifactBackendSurfacesFailureDetails() async {
        let backend = ModelArtifactExperimentBackend(
            preparer: ModelArtifactPreparer(
                manifest: ModelRuntimeManifest(
                    modelRepo: "repo",
                    revision: "rev",
                    requiredFiles: [
                        ModelArtifactRequirement(role: "architectureConfig", fileName: "config.json"),
                    ]
                ),
                provider: RecordingArtifactProvider(failure: .incompatible("config.json"))
            )
        )
        let session = ExperimentSession(backend: backend)

        await session.prepare()

        #expect(session.state == .failed("Model artifact incompatible: config.json"))
        #expect(session.observations.lastEventName == "failed")
    }
}

private enum ProviderRequest: Equatable, Sendable {
    case cached(String)
    case prepared(fileName: String, repo: String, revision: String)
}

private actor RecordingArtifactProvider: ModelArtifactProviding {
    private var cachedArtifacts: [String: ModelArtifactHandle]
    private var preparedArtifacts: [String: ModelArtifactHandle]
    private var failure: ModelArtifactPreparationError?
    private var recordedRequests: [ProviderRequest] = []

    init(
        cached: [String: ModelArtifactHandle] = [:],
        prepared: [String: ModelArtifactHandle] = [:],
        failure: ModelArtifactPreparationError? = nil
    ) {
        self.cachedArtifacts = cached
        self.preparedArtifacts = prepared
        self.failure = failure
    }

    func cachedArtifact(named fileName: String) async throws -> ModelArtifactHandle? {
        recordedRequests.append(.cached(fileName))
        return cachedArtifacts[fileName]
    }

    func prepareArtifact(
        named fileName: String,
        from modelRepo: String,
        revision: String
    ) async throws -> ModelArtifactHandle {
        recordedRequests.append(.prepared(fileName: fileName, repo: modelRepo, revision: revision))

        if let failure {
            throw failure
        }

        guard let prepared = preparedArtifacts[fileName] else {
            throw ModelArtifactPreparationError.missing(fileName)
        }

        return prepared
    }

    func requests() -> [ProviderRequest] {
        recordedRequests
    }
}

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

    @Test("Hugging Face provider downloads missing artifacts to repo revision store")
    func huggingFaceProviderDownloadsMissingArtifacts() async throws {
        let storeRootURL = try makeTemporaryDirectory()
        let downloader = FakeArtifactDownloader(
            behavior: .success(Data("artifact-bytes".utf8), expectedByteCount: 14)
        )
        let provider = HuggingFaceModelArtifactProvider(
            storeRootURL: storeRootURL,
            downloader: downloader
        )

        let handle = try await provider.prepareArtifact(
            named: "config.json",
            from: "owner/model",
            revision: "abc123"
        ) { _, _ in }
        let requestedURLs = await downloader.requestedURLs()
        let finalURL = storeRootURL
            .appendingPathComponent("owner__model", isDirectory: true)
            .appendingPathComponent("abc123", isDirectory: true)
            .appendingPathComponent("config.json", isDirectory: false)
        let temporaryURL = finalURL
            .deletingLastPathComponent()
            .appendingPathComponent(".config.json.download", isDirectory: false)

        #expect(handle.fileName == "config.json")
        #expect(handle.location == finalURL.path)
        #expect(handle.byteCount == 14)
        #expect(requestedURLs == [
            URL(string: "https://huggingface.co/owner/model/resolve/abc123/config.json")!,
        ])
        #expect(FileManager.default.fileExists(atPath: finalURL.path))
        #expect(!FileManager.default.fileExists(atPath: temporaryURL.path))
    }

    @Test("Hugging Face provider reuses completed cache before downloading")
    func huggingFaceProviderReusesCompletedCache() async throws {
        let storeRootURL = try makeTemporaryDirectory()
        let cachedURL = storeRootURL
            .appendingPathComponent("owner__model", isDirectory: true)
            .appendingPathComponent("abc123", isDirectory: true)
            .appendingPathComponent("config.json", isDirectory: false)
        try FileManager.default.createDirectory(
            at: cachedURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try Data("cached".utf8).write(to: cachedURL)
        let downloader = FakeArtifactDownloader(behavior: .failure(ModelArtifactDownloadError.httpStatus(500)))
        let provider = HuggingFaceModelArtifactProvider(
            storeRootURL: storeRootURL,
            downloader: downloader
        )

        let handle = try await provider.prepareArtifact(
            named: "config.json",
            from: "owner/model",
            revision: "abc123"
        )
        let requestedURLs = await downloader.requestedURLs()

        #expect(handle.location == cachedURL.path)
        #expect(handle.byteCount == 6)
        #expect(requestedURLs.isEmpty)
    }

    @Test("Hugging Face provider removes partial temp download on failure")
    func huggingFaceProviderRemovesPartialTempDownloadOnFailure() async throws {
        let storeRootURL = try makeTemporaryDirectory()
        let downloader = FakeArtifactDownloader(
            behavior: .failure(
                ModelArtifactDownloadError.httpStatus(404),
                partialData: Data("partial".utf8)
            )
        )
        let provider = HuggingFaceModelArtifactProvider(
            storeRootURL: storeRootURL,
            downloader: downloader
        )

        do {
            _ = try await provider.prepareArtifact(
                named: "missing.safetensors",
                from: "owner/model",
                revision: "abc123"
            )
            Issue.record("Expected provider to throw for failed download")
        } catch let error as ModelArtifactPreparationError {
            #expect(error == .inaccessible("missing.safetensors"))
        }

        let artifactDirectoryURL = storeRootURL
            .appendingPathComponent("owner__model", isDirectory: true)
            .appendingPathComponent("abc123", isDirectory: true)
        let finalURL = artifactDirectoryURL.appendingPathComponent("missing.safetensors", isDirectory: false)
        let temporaryURL = artifactDirectoryURL.appendingPathComponent(".missing.safetensors.download", isDirectory: false)

        #expect(!FileManager.default.fileExists(atPath: finalURL.path))
        #expect(!FileManager.default.fileExists(atPath: temporaryURL.path))
    }

    @Test("Hugging Face provider maps insufficient disk to too large")
    func huggingFaceProviderMapsInsufficientDiskToTooLarge() async throws {
        let provider = HuggingFaceModelArtifactProvider(
            storeRootURL: try makeTemporaryDirectory(),
            downloader: FakeArtifactDownloader(
                behavior: .failure(
                    ModelArtifactDownloadError.insufficientDiskSpace(
                        requiredBytes: 100,
                        availableBytes: 10
                    )
                )
            )
        )

        do {
            _ = try await provider.prepareArtifact(
                named: "hibiki.q4.safetensors",
                from: "owner/model",
                revision: "abc123"
            )
            Issue.record("Expected provider to throw for insufficient disk")
        } catch let error as ModelArtifactPreparationError {
            #expect(error == .tooLarge("hibiki.q4.safetensors"))
        }
    }

    @Test("preparer rejects provider artifact filename mismatch")
    func preparerRejectsProviderArtifactFilenameMismatch() async {
        let provider = RecordingArtifactProvider(
            prepared: [
                "config.json": ModelArtifactHandle(
                    fileName: "wrong-config.json",
                    location: "prepared://wrong-config.json",
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

        #expect(result.failure == .incompatible("config.json"))
        #expect(!result.succeeded)
    }

    @Test("artifact progress reaches Experiment Session observations")
    @MainActor
    func artifactProgressReachesExperimentSessionObservations() async {
        let provider = ProgressReportingArtifactProvider(
            handle: ModelArtifactHandle(
                fileName: "config.json",
                location: "prepared://config.json",
                byteCount: 10
            )
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
        #expect(session.observations.artifactFileName == "config.json")
        #expect(session.observations.artifactCompletedFileCount == 1)
        #expect(session.observations.artifactTotalFileCount == 1)
        #expect(session.observations.artifactFileProgress == 1)
        #expect(session.observations.progress == 1)
    }
}

private func makeTemporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
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

private enum FakeDownloadBehavior: Sendable {
    case success(Data, expectedByteCount: Int64?)
    case failure(any Error, partialData: Data? = nil)
}

private actor FakeArtifactDownloader: ModelArtifactDownloading {
    private let behavior: FakeDownloadBehavior
    private var recordedURLs: [URL] = []

    init(behavior: FakeDownloadBehavior) {
        self.behavior = behavior
    }

    func download(
        from url: URL,
        to destinationURL: URL,
        progress: @Sendable (Int64, Int64?) async -> Void
    ) async throws -> Int64 {
        recordedURLs.append(url)

        switch behavior {
        case let .success(data, expectedByteCount):
            try data.write(to: destinationURL)
            await progress(Int64(data.count), expectedByteCount)
            return Int64(data.count)
        case let .failure(error, partialData):
            if let partialData {
                try partialData.write(to: destinationURL)
                await progress(Int64(partialData.count), nil)
            }
            throw error
        }
    }

    func requestedURLs() -> [URL] {
        recordedURLs
    }
}

private actor ProgressReportingArtifactProvider: ModelArtifactProgressReportingProviding {
    private let handle: ModelArtifactHandle

    init(handle: ModelArtifactHandle) {
        self.handle = handle
    }

    func cachedArtifact(named fileName: String) async throws -> ModelArtifactHandle? {
        nil
    }

    func prepareArtifact(
        named fileName: String,
        from modelRepo: String,
        revision: String
    ) async throws -> ModelArtifactHandle {
        handle
    }

    func prepareArtifact(
        named fileName: String,
        from modelRepo: String,
        revision: String,
        progress: @Sendable (Int64, Int64?) async -> Void
    ) async throws -> ModelArtifactHandle {
        await progress(5, 10)
        await progress(10, 10)
        return handle
    }
}

import Foundation
import Testing

@testable import S2STranslateCore

@Suite("Real File French-English Smoke")
struct RealFileFrenchEnglishSmokeTests {
    @Test("French Europarl short 1 streams through the real MLX file backend")
    @MainActor
    func frenchEuroparlShortOneStreamsThroughRealMLXFileBackend() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["S2S_RUN_REAL_FILE_SMOKE_TESTS"] == "1" else {
            return
        }

        let weightsDirectory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(
                environment["S2S_REAL_FILE_SMOKE_WEIGHTS_DIR"] ?? "ref/hibiki-zero-mlx/weights",
                isDirectory: true
            )
        let artifacts = try localHibikiArtifacts(weightsDirectory: weightsDirectory)
        let session = ExperimentSession(
            backend: RealFileHibikiTranslationExperimentBackend(
                artifactPreparer: ModelArtifactPreparer(
                    manifest: .hibikiQ4Default,
                    provider: try LocalSmokeArtifactProvider(artifacts: artifacts)
                ),
                audioSource: smokeAudioSource(environment: environment),
                playbackSink: BufferedPlaybackSink(),
                generationConfiguration: HibikiGenerationConfiguration(
                    tailSilenceFrameCount: 8,
                    postInputPaddingStopFrameCount: 3
                )
            )
        )

        await session.prepare()
        #expect(session.state == .ready)
        guard session.state == .ready else { return }

        await session.start()

        #expect(session.state == .running)
        #expect(session.observations.audioChunkCount > 0)
        #expect(session.observations.mimiEncodedFrameCount > 0)
        #expect(session.observations.hibikiStepCount > 0)
        #expect(session.observations.hibikiTextTokenCount > 0)
        #expect(session.observations.hibikiVisibleTextCount > 0)
        #expect(!session.observations.output.isEmpty)
        #expect(session.observations.hibikiGeneratedAudioFrameCount > 0)
        #expect(session.observations.decodedAudioChunkCount > 0)
        #expect(session.observations.playbackChunkCount > 0)
    }
}

private func smokeAudioSource(environment: [String: String]) -> any AudioInputSource {
    if let audioPath = environment["S2S_REAL_FILE_SMOKE_AUDIO_PATH"], !audioPath.isEmpty {
        return FileAudioInputSource(fileURL: URL(fileURLWithPath: audioPath))
    }
    return RemoteAudioFileInputSource(fixture: FileAudioFixtureCatalog.frenchShortForm[0])
}

private func localHibikiArtifacts(weightsDirectory: URL) throws -> PreparedModelArtifacts {
    let fileManager = FileManager.default
    let files = try ModelRuntimeManifest.hibikiQ4Default.requiredFiles.map { requirement in
        let url = weightsDirectory.appendingPathComponent(requirement.fileName, isDirectory: false)
        guard fileManager.fileExists(atPath: url.path) else {
            throw ModelArtifactPreparationError.missing(url.path)
        }
        return PreparedModelArtifact(
            role: requirement.role,
            fileName: requirement.fileName,
            location: url.path,
            source: .cache
        )
    }
    return PreparedModelArtifacts(manifest: .hibikiQ4Default, files: files)
}

private actor LocalSmokeArtifactProvider: ModelArtifactProviding {
    private let handles: [String: ModelArtifactHandle]

    init(artifacts: PreparedModelArtifacts) throws {
        self.handles = try Dictionary(
            uniqueKeysWithValues: artifacts.files.map { artifact in
                let url = URL(fileURLWithPath: artifact.location)
                let byteCount = try Self.byteCount(at: url)
                return (
                    artifact.fileName,
                    ModelArtifactHandle(
                        fileName: artifact.fileName,
                        location: artifact.location,
                        byteCount: byteCount
                    )
                )
            }
        )
    }

    func cachedArtifact(named fileName: String) async throws -> ModelArtifactHandle? {
        handles[fileName]
    }

    func prepareArtifact(
        named fileName: String,
        from modelRepo: String,
        revision: String
    ) async throws -> ModelArtifactHandle {
        guard let handle = handles[fileName] else {
            throw ModelArtifactPreparationError.missing(fileName)
        }
        return handle
    }

    private static func byteCount(at url: URL) throws -> Int64 {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        return attributes[.size] as? Int64 ?? 1
    }
}

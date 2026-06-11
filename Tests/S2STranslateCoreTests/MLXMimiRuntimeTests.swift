import Foundation
import Testing

@testable import S2STranslateCore

@Suite("MLX Mimi Runtime")
struct MLXMimiRuntimeTests {
    @Test("runtime loader creates metadata runtime from prepared Mimi artifact")
    func runtimeLoaderCreatesMetadataRuntimeFromPreparedMimiArtifact() throws {
        let directory = try makeRuntimeTemporaryDirectory()
        let weightsURL = directory.appendingPathComponent("mimi.safetensors")
        try Data("fake mimi weights".utf8).write(to: weightsURL)
        let artifacts = PreparedModelArtifacts(
            manifest: ModelRuntimeManifest(
                modelRepo: "repo",
                revision: "rev",
                requiredFiles: [
                    ModelArtifactRequirement(role: "mimiWeights", fileName: "mimi.safetensors"),
                ]
            ),
            files: [
                PreparedModelArtifact(
                    role: "mimiWeights",
                    fileName: "mimi.safetensors",
                    location: weightsURL.path,
                    source: .prepared
                ),
            ]
        )

        let runtime = try MLXMimiRuntimeLoader().load(from: artifacts)

        #expect(runtime.artifact.fileName == "mimi.safetensors")
        #expect(runtime.artifact.location == weightsURL.path)
        #expect(runtime.configuration.sampleRate == 24_000)
        #expect(runtime.configuration.frameRate == 12.5)
        #expect(runtime.configuration.codebookCount == 16)
        #expect(runtime.configuration.samplesPerFrame == 1_920)
    }

    @Test("metadata runtime exposes safe reset entry points")
    func metadataRuntimeExposesSafeResetEntryPoints() throws {
        let directory = try makeRuntimeTemporaryDirectory()
        let weightsURL = directory.appendingPathComponent("mimi.safetensors")
        try Data("fake mimi weights".utf8).write(to: weightsURL)
        let runtime = MLXMimiRuntime(
            artifact: PreparedModelArtifact(
                role: "mimiWeights",
                fileName: "mimi.safetensors",
                location: weightsURL.path,
                source: .prepared
            )
        )

        runtime.resetEncodeState()
        runtime.resetDecodeState()
        runtime.resetState()
    }

    @Test("runtime loader fails when Mimi artifact role is missing")
    func runtimeLoaderFailsWhenMimiArtifactRoleIsMissing() throws {
        let artifacts = PreparedModelArtifacts(
            manifest: ModelRuntimeManifest(
                modelRepo: "repo",
                revision: "rev",
                requiredFiles: [
                    ModelArtifactRequirement(role: "hibikiWeights", fileName: "hibiki.safetensors"),
                ]
            ),
            files: [
                PreparedModelArtifact(
                    role: "hibikiWeights",
                    fileName: "hibiki.safetensors",
                    location: "/tmp/hibiki.safetensors",
                    source: .prepared
                ),
            ]
        )

        #expect(throws: MimiRuntimeError.missingArtifactRole("mimiWeights")) {
            try MLXMimiRuntimeLoader().load(from: artifacts)
        }
    }

    @Test("runtime loader fails when Mimi artifact file is missing")
    func runtimeLoaderFailsWhenMimiArtifactFileIsMissing() throws {
        let artifacts = PreparedModelArtifacts(
            manifest: ModelRuntimeManifest(
                modelRepo: "repo",
                revision: "rev",
                requiredFiles: [
                    ModelArtifactRequirement(role: "mimiWeights", fileName: "mimi.safetensors"),
                ]
            ),
            files: [
                PreparedModelArtifact(
                    role: "mimiWeights",
                    fileName: "mimi.safetensors",
                    location: "/tmp/missing-\(UUID().uuidString).safetensors",
                    source: .prepared
                ),
            ]
        )

        #expect(throws: MimiRuntimeError.missingArtifactFile("mimi.safetensors")) {
            try MLXMimiRuntimeLoader().load(from: artifacts)
        }
    }
}

private func makeRuntimeTemporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

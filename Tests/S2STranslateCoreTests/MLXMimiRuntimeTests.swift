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

        let runtime = try makeShapeOnlyRuntimeLoader().load(from: artifacts)

        #expect(runtime.artifact.fileName == "mimi.safetensors")
        #expect(runtime.artifact.location == weightsURL.path)
        #expect(runtime.configuration.sampleRate == 24_000)
        #expect(runtime.configuration.frameRate == 12.5)
        #expect(runtime.configuration.codebookCount == 16)
        #expect(runtime.configuration.quantizerBins == 2_048)
        #expect(runtime.configuration.samplesPerFrame == 1_920)
    }

    @Test("runtime validates Mimi 2024 07 metadata")
    func runtimeValidatesMimi202407Metadata() throws {
        let runtime = MLXMimiRuntime(
            artifact: try makePreparedMimiArtifact(),
            configuration: .mimi202407
        )

        try runtime.validateMetadata()
    }

    @Test("runtime reports incompatible metadata")
    func runtimeReportsIncompatibleMetadata() throws {
        let runtime = MLXMimiRuntime(
            artifact: try makePreparedMimiArtifact(),
            configuration: MLXMimiRuntimeConfiguration(
                sampleRate: 16_000,
                frameRate: 12.5,
                codebookCount: 16,
                samplesPerFrame: 1_920,
                quantizerBins: 2_048
            )
        )

        #expect(
            throws: MimiRuntimeError.incompatibleConfiguration(
                "sampleRate expected 24000, got 16000"
            )
        ) {
            try runtime.validateMetadata()
        }
    }

    @Test("runtime forwards reset entry points to engine")
    func runtimeForwardsResetEntryPointsToEngine() throws {
        let directory = try makeRuntimeTemporaryDirectory()
        let weightsURL = directory.appendingPathComponent("mimi.safetensors")
        try Data("fake mimi weights".utf8).write(to: weightsURL)
        let engine = FakeMimiRuntimeEngine()
        let runtime = MLXMimiRuntime(
            artifact: PreparedModelArtifact(
                role: "mimiWeights",
                fileName: "mimi.safetensors",
                location: weightsURL.path,
                source: .prepared
            ),
            engine: engine
        )

        runtime.resetEncodeState()
        runtime.resetDecodeState()
        runtime.resetState()

        #expect(engine.resetEncodeCount == 2)
        #expect(engine.resetDecodeCount == 2)
    }

    @Test("default runtime engine runs encode graph and preserves empty output")
    func defaultRuntimeEngineRunsEncodeGraphAndPreservesEmptyOutput() throws {
        var graphCallCount = 0
        var extractionCallCount = 0
        let engine = MLXMimiDefaultRuntimeEngine(
            inputBuilder: { input in
                #expect(input.pcmShape == [1, 1, 960])
                return MLXMimiStreamArray()
            },
            encodeGraphStep: { stream in
                graphCallCount += 1
                #expect(stream.isEmpty)
                return MLXMimiStreamArray()
            },
            tokenExtractor: { stream in
                extractionCallCount += 1
                #expect(stream.isEmpty)
                return []
            }
        )

        let frames = try engine.encode(
            MLXMimiPCMInput(
                samples: Array(repeating: 0, count: 960),
                sampleRate: 24_000,
                pcmShape: [1, 1, 960]
            )
        )

        #expect(frames.isEmpty)
        #expect(graphCallCount == 1)
        #expect(extractionCallCount == 1)
    }

    @Test("token frame extractor preserves codebook count and frame order")
    func tokenFrameExtractorPreservesCodebookCountAndFrameOrder() throws {
        let extractor = MLXMimiTokenFrameExtractor(codebookCount: 3)

        let frames = try extractor.frames(
            flattenedTokens: [
                10, 11,
                20, 21,
                30, 31,
            ],
            shape: [1, 3, 2]
        )

        #expect(frames == [
            MLXMimiEncodedFrame(tokens: [10, 20, 30]),
            MLXMimiEncodedFrame(tokens: [11, 21, 31]),
        ])
    }

    @Test("token frame extractor reports malformed output shape")
    func tokenFrameExtractorReportsMalformedOutputShape() throws {
        let extractor = MLXMimiTokenFrameExtractor(codebookCount: 16)

        #expect(
            throws: MimiRuntimeError.loadFailed(
                "Mimi token output codebooks malformed: expected 16, got 3"
            )
        ) {
            _ = try extractor.frames(
                flattenedTokens: [1, 2, 3],
                shape: [1, 3, 1]
            )
        }
    }

    @Test("warmup builds zero PCM request and exercises engine")
    func warmupBuildsZeroPCMRequestAndExercisesEngine() throws {
        let engine = FakeMimiRuntimeEngine()
        let runtime = MLXMimiRuntime(
            artifact: try makePreparedMimiArtifact(),
            engine: engine
        )

        try runtime.warmup()

        #expect(engine.warmupRequests == [
            MLXMimiWarmupRequest(
                pcmShape: [1, 1, 7_680],
                sampleCount: 7_680,
                frameCount: 4
            ),
        ])
    }

    @Test("warmup failures become user visible runtime errors")
    func warmupFailuresBecomeUserVisibleRuntimeErrors() throws {
        let engine = FakeMimiRuntimeEngine()
        engine.warmupError = ExampleMimiEngineError.failed
        let runtime = MLXMimiRuntime(
            artifact: try makePreparedMimiArtifact(),
            engine: engine
        )

        #expect(throws: MimiRuntimeError.warmupFailed("failed")) {
            try runtime.warmup()
        }
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

    @Test("runtime loader reports incompatible graph parameter shape")
    func runtimeLoaderReportsIncompatibleGraphParameterShape() throws {
        let artifacts = PreparedModelArtifacts(
            manifest: ModelRuntimeManifest(
                modelRepo: "repo",
                revision: "rev",
                requiredFiles: [
                    ModelArtifactRequirement(role: "mimiWeights", fileName: "mimi.safetensors"),
                ]
            ),
            files: [
                try makePreparedMimiArtifact(),
            ]
        )
        let key = "encoder_transformer.layers.0.self_attn.in_proj.weight"
        let loader = MLXMimiRuntimeLoader(
            weightLoader: MLXMimiWeightLoader(arrayReader: { _ in
                [
                    key: MLXMimiWeightTensor(shape: [1], array: nil),
                ]
            }),
            graphParameterApplier: MLXMimiGraphParameterApplier(requiredKeys: [key])
        )

        #expect(
            throws: MimiRuntimeError.loadFailed(
                "Mimi weight shape incompatible for \(key): expected [1536, 512], got [1]"
            )
        ) {
            try loader.load(from: artifacts)
        }
    }
}

private func makeShapeOnlyRuntimeLoader() -> MLXMimiRuntimeLoader {
    MLXMimiRuntimeLoader(
        weightLoader: MLXMimiWeightLoader(arrayReader: { _ in [:] }),
        graphParameterApplier: MLXMimiGraphParameterApplier(requiredKeys: [])
    )
}

private func makeRuntimeTemporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func makePreparedMimiArtifact() throws -> PreparedModelArtifact {
    let directory = try makeRuntimeTemporaryDirectory()
    let weightsURL = directory.appendingPathComponent("mimi.safetensors")
    try Data("fake mimi weights".utf8).write(to: weightsURL)
    return PreparedModelArtifact(
        role: "mimiWeights",
        fileName: "mimi.safetensors",
        location: weightsURL.path,
        source: .prepared
    )
}

private final class FakeMimiRuntimeEngine: MLXMimiRuntimeEngine {
    var resetEncodeCount = 0
    var resetDecodeCount = 0
    var warmupRequests: [MLXMimiWarmupRequest] = []
    var warmupError: Error?

    func resetEncodeState() {
        resetEncodeCount += 1
    }

    func resetDecodeState() {
        resetDecodeCount += 1
    }

    func warmup(request: MLXMimiWarmupRequest) throws {
        if let warmupError {
            throw warmupError
        }
        warmupRequests.append(request)
    }

    func encode(_ input: MLXMimiPCMInput) throws -> [MLXMimiEncodedFrame] {
        []
    }

    func decode(_ input: MLXMimiTokenInput) throws -> [MLXMimiDecodedChunk] {
        []
    }
}

private enum ExampleMimiEngineError: Error, CustomStringConvertible {
    case failed

    var description: String {
        "failed"
    }
}

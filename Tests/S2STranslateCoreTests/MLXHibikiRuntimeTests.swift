import Foundation
import Testing

@testable import S2STranslateCore

@Suite("MLX Hibiki Runtime")
struct MLXHibikiRuntimeTests {
    @Test("MLX Hibiki session initializes from prepared artifacts and steps through engine seam")
    func mlxHibikiSessionInitializesAndStepsThroughEngineSeam() async throws {
        let artifacts = try preparedHibikiArtifacts(configJSON: validConfigJSON())
        let engine = RecordingHibikiRuntimeEngine(
            output: MLXHibikiStepOutput(
                textToken: 42,
                textPiece: " hello",
                textCandidateTokens: [42, 43],
                audioTokens: Array(700..<716)
            )
        )
        let session = MLXHibikiInferenceSession(engine: engine)
        let generation = HibikiGenerationConfiguration(
            temperature: 0.7,
            textTemperature: 0.6,
            topK: 128,
            textTopK: 64
        )

        let description = try await session.initialize(
            artifacts: artifacts,
            configuration: generation
        )
        let step = try await session.step(sourceAudioTokens: sourceTokenFrame(frameIndex: 0))

        #expect(description.modelRevision == ModelRuntimeManifest.hibikiQ4Default.revision)
        #expect(description.artifactCount == 4)
        #expect(description.configuration == generation)
        #expect(engine.loadRequests.count == 1)
        #expect(engine.loadRequests[0].runtimeConfiguration.quantizationBits == 4)
        #expect(engine.loadRequests[0].runtimeConfiguration.quantizationGroupSize == 32)
        #expect(engine.stepRequests.count == 1)
        #expect(engine.stepRequests[0].configuration == generation)
        #expect(step.frameIndex == 0)
        #expect(step.text == HibikiTextOutput(frameIndex: 0, token: 42, piece: " hello", candidateTokens: [42, 43]))
        #expect(step.generatedAudioTokens.tokens == Array(700..<716))
    }

    @Test("MLX Hibiki default engine validates q4 architecture metadata")
    func mlxHibikiDefaultEngineValidatesQ4ArchitectureMetadata() throws {
        let artifacts = try preparedHibikiArtifacts(configJSON: validConfigJSON())
        let request = try makeLoadRequest(artifacts: artifacts)
        let engine = MLXHibikiDefaultRuntimeEngine { url in
            #expect(url.lastPathComponent == "hibiki.q4.safetensors")
            return MLXHibikiWeightsSummary(tensorCount: 3, hasDepformerOutputLayerNorms: true)
        }

        try engine.load(request: request)

        #expect(
            throws: HibikiInferenceError.unavailable("Hibiki MLX model step graph not implemented")
        ) {
            _ = try engine.step(
                sourceAudioTokens: sourceTokenFrame(frameIndex: 0),
                frameIndex: 0,
                configuration: HibikiGenerationConfiguration()
            )
        }
    }

    @Test("MLX Hibiki default engine rejects missing architecture deltas")
    func mlxHibikiDefaultEngineRejectsMissingArchitectureDeltas() throws {
        let artifacts = try preparedHibikiArtifacts(
            configJSON: validConfigJSON(kvRepeat: 1)
        )
        let request = try makeLoadRequest(artifacts: artifacts)
        let engine = MLXHibikiDefaultRuntimeEngine { _ in
            MLXHibikiWeightsSummary(tensorCount: 1, hasDepformerOutputLayerNorms: true)
        }

        #expect(
            throws: HibikiInferenceError.invalidArtifacts("kv_repeat expected 2, got 1")
        ) {
            try engine.load(request: request)
        }
    }

    @Test("MLX Hibiki default engine rejects weights without depformer output norms")
    func mlxHibikiDefaultEngineRejectsWeightsWithoutDepformerOutputNorms() throws {
        let artifacts = try preparedHibikiArtifacts(configJSON: validConfigJSON())
        let request = try makeLoadRequest(artifacts: artifacts)
        let engine = MLXHibikiDefaultRuntimeEngine { _ in
            MLXHibikiWeightsSummary(tensorCount: 1, hasDepformerOutputLayerNorms: false)
        }

        #expect(
            throws: HibikiInferenceError.invalidArtifacts("hibiki weights missing depformer output LayerNorm tensors")
        ) {
            try engine.load(request: request)
        }
    }

    @Test("MLX Hibiki session reports missing tokenizer artifacts")
    func mlxHibikiSessionReportsMissingTokenizerArtifacts() async throws {
        var artifacts = try preparedHibikiArtifacts(configJSON: validConfigJSON())
        artifacts.files.removeAll { $0.role == "tokenizer" }
        let session = MLXHibikiInferenceSession(engine: RecordingHibikiRuntimeEngine())

        await #expect(throws: HibikiInferenceError.invalidArtifacts("missing tokenizer")) {
            _ = try await session.initialize(
                artifacts: artifacts,
                configuration: HibikiGenerationConfiguration()
            )
        }
    }

    @Test("translation backend runs with MLX Hibiki session seam")
    @MainActor
    func translationBackendRunsWithMLXHibikiSessionSeam() async throws {
        let artifacts = try preparedHibikiArtifacts(configJSON: validConfigJSON())
        let engine = RecordingHibikiRuntimeEngine(
            output: MLXHibikiStepOutput(
                textToken: 801,
                textPiece: " real",
                textCandidateTokens: [801, 802],
                audioTokens: Array(900..<916)
            )
        )
        let session = ExperimentSession(
            backend: HibikiTranslationExperimentBackend(
                artifactPreparer: ModelArtifactPreparer(
                    manifest: .hibikiQ4Default,
                    provider: PreparedArtifactProvider(artifacts: artifacts)
                ),
                audioSource: FixtureAudioInputSource(sampleRate: 24_000, chunkSampleCount: 1_920, chunkCount: 2),
                mimiEncoder: DeterministicMimiStreamingEncoder(),
                inferenceSession: MLXHibikiInferenceSession(engine: engine),
                mimiDecoder: DeterministicMimiStreamingDecoder(),
                playbackSink: BufferedPlaybackSink()
            )
        )

        await session.prepare()
        await session.start()

        #expect(session.state == .running)
        #expect(session.observations.hibikiStepCount == 2)
        #expect(session.observations.hibikiVisibleTextCount == 2)
        #expect(session.observations.decodedAudioChunkCount == 2)
        #expect(session.observations.playbackChunkCount == 2)
        #expect(session.observations.output == " real real")
    }
}

private final class RecordingHibikiRuntimeEngine: MLXHibikiRuntimeEngine {
    struct StepRequest: Equatable {
        var sourceAudioTokens: MimiTokenFrame
        var frameIndex: Int
        var configuration: HibikiGenerationConfiguration
    }

    var loadRequests: [MLXHibikiLoadRequest] = []
    var stepRequests: [StepRequest] = []
    var output: MLXHibikiStepOutput

    init(
        output: MLXHibikiStepOutput = MLXHibikiStepOutput(
            textToken: 0,
            audioTokens: Array(repeating: 0, count: 16)
        )
    ) {
        self.output = output
    }

    func load(request: MLXHibikiLoadRequest) throws {
        loadRequests.append(request)
    }

    func step(
        sourceAudioTokens: MimiTokenFrame,
        frameIndex: Int,
        configuration: HibikiGenerationConfiguration
    ) throws -> MLXHibikiStepOutput {
        stepRequests.append(
            StepRequest(
                sourceAudioTokens: sourceAudioTokens,
                frameIndex: frameIndex,
                configuration: configuration
            )
        )
        return output
    }

    func reset() {
        loadRequests.removeAll()
        stepRequests.removeAll()
    }
}

private actor PreparedArtifactProvider: ModelArtifactProviding {
    private let handles: [String: ModelArtifactHandle]

    init(artifacts: PreparedModelArtifacts) {
        self.handles = Dictionary(
            uniqueKeysWithValues: artifacts.files.map {
                (
                    $0.fileName,
                    ModelArtifactHandle(
                        fileName: $0.fileName,
                        location: $0.location,
                        byteCount: 1
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
}

private func preparedHibikiArtifacts(configJSON: String) throws -> PreparedModelArtifacts {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

    let configURL = directory.appendingPathComponent("config.json")
    let weightsURL = directory.appendingPathComponent("hibiki.q4.safetensors")
    let mimiURL = directory.appendingPathComponent("mimi-pytorch-e351c8d8@125.safetensors")
    let tokenizerURL = directory.appendingPathComponent("tokenizer_spm_48k_multi6_2.model")
    try configJSON.data(using: .utf8)!.write(to: configURL)
    try Data("fake q4 weights".utf8).write(to: weightsURL)
    try Data("fake mimi weights".utf8).write(to: mimiURL)
    try Data("fake tokenizer".utf8).write(to: tokenizerURL)

    return PreparedModelArtifacts(
        manifest: .hibikiQ4Default,
        files: [
            PreparedModelArtifact(role: "architectureConfig", fileName: "config.json", location: configURL.path, source: .prepared),
            PreparedModelArtifact(role: "hibikiWeights", fileName: "hibiki.q4.safetensors", location: weightsURL.path, source: .prepared),
            PreparedModelArtifact(role: "mimiWeights", fileName: "mimi-pytorch-e351c8d8@125.safetensors", location: mimiURL.path, source: .prepared),
            PreparedModelArtifact(role: "tokenizer", fileName: "tokenizer_spm_48k_multi6_2.model", location: tokenizerURL.path, source: .prepared),
        ]
    )
}

private func makeLoadRequest(artifacts: PreparedModelArtifacts) throws -> MLXHibikiLoadRequest {
    let files = Dictionary(uniqueKeysWithValues: artifacts.files.map { ($0.role, $0) })
    return MLXHibikiLoadRequest(
        configURL: URL(fileURLWithPath: files["architectureConfig"]!.location),
        weightsURL: URL(fileURLWithPath: files["hibikiWeights"]!.location),
        tokenizerURL: URL(fileURLWithPath: files["tokenizer"]!.location),
        artifactCount: artifacts.files.count,
        modelRevision: artifacts.manifest.revision,
        runtimeConfiguration: MLXHibikiRuntimeConfiguration()
    )
}

private func validConfigJSON(
    hiddenScale: Int = 6,
    kvRepeat: Int = 2,
    positionalEmbedding: String = "rope_concat"
) -> String {
    """
    {
      "hidden_scale": \(hiddenScale),
      "kv_repeat": \(kvRepeat),
      "transformer": {
        "positional_embedding": "\(positionalEmbedding)"
      }
    }
    """
}

private func sourceTokenFrame(frameIndex: Int) -> MimiTokenFrame {
    MimiTokenFrame(
        frameIndex: frameIndex,
        timestampMilliseconds: Double(frameIndex) * 80,
        codebookCount: 16,
        tokens: Array((101 + frameIndex * 16)..<(117 + frameIndex * 16)),
        sourceAudioFrameIndex: frameIndex
    )
}

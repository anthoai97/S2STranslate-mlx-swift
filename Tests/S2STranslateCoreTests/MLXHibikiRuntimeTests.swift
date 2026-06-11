import Foundation
import MLX
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

    @Test("MLX Hibiki session decodes text tokens through tokenizer seam")
    func mlxHibikiSessionDecodesTextTokensThroughTokenizerSeam() async throws {
        let artifacts = try preparedHibikiArtifacts(configJSON: validConfigJSON())
        let engine = RecordingHibikiRuntimeEngine(
            outputs: [
                MLXHibikiStepOutput(
                    textToken: 42,
                    textCandidateTokens: [42, 0],
                    audioTokens: Array(700..<716)
                ),
                MLXHibikiStepOutput(
                    textToken: 3,
                    textCandidateTokens: [3],
                    audioTokens: Array(716..<732)
                ),
            ]
        )
        let session = MLXHibikiInferenceSession(
            engine: engine,
            textTokenDecoder: DictionaryHibikiTextTokenDecoder(piecesByToken: [42: "\u{2581}hello"])
        )

        _ = try await session.initialize(
            artifacts: artifacts,
            configuration: HibikiGenerationConfiguration()
        )
        let visible = try await session.step(sourceAudioTokens: sourceTokenFrame(frameIndex: 0))
        let skipped = try await session.step(sourceAudioTokens: sourceTokenFrame(frameIndex: 1))

        #expect(visible.text.piece == " hello")
        #expect(visible.text.isVisible)
        #expect(visible.text.candidateTokens == [42, 0])
        #expect(skipped.text.piece == nil)
        #expect(skipped.text.isBlankOrPadding)
        #expect(!skipped.text.isVisible)
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

    @Test("MLX Hibiki config maps real LM and Depformer topology")
    func mlxHibikiConfigMapsRealLMAndDepformerTopology() throws {
        let artifacts = try preparedHibikiArtifacts(configJSON: validConfigJSON())
        let request = try makeLoadRequest(artifacts: artifacts)

        let config = try MLXHibikiModelConfig.load(from: request.configURL)

        #expect(config.topology.textInputVocabSize == 48_001)
        #expect(config.topology.textOutputVocabSize == 48_000)
        #expect(config.topology.audioVocabSize == 2_049)
        #expect(config.topology.audioCodebookCount == 32)
        #expect(config.topology.generatedCodebookCount == 16)
        #expect(config.topology.sourceCodebookCount == 16)
        #expect(config.topology.audioDelays.count == 32)
        #expect(config.topology.mainTransformer.modelDimension == 2_048)
        #expect(config.topology.mainTransformer.headCount == 16)
        #expect(config.topology.mainTransformer.layerCount == 28)
        #expect(config.topology.mainTransformer.feedForwardDimension == 12_288)
        #expect(config.topology.mainTransformer.kvRepeat == 2)
        #expect(config.topology.mainTransformer.positionalEmbedding == .ropeConcat)
        #expect(config.topology.depformerTransformer.modelDimension == 1_024)
        #expect(config.topology.depformerTransformer.layerCount == 6)
        #expect(config.topology.depformerTransformer.feedForwardDimension == 6_144)
        #expect(config.topology.depformerTransformer.positionalEmbedding == .none)
        #expect(config.topology.depformerWeightsPerStepSchedule == [
            0, 1, 2, 3, 4, 5, 6, 7,
            8, 8, 8, 8, 8, 8, 8, 8,
        ])
    }

    @Test("MLX Hibiki language model owns real graph shell shapes")
    func mlxHibikiLanguageModelOwnsRealGraphShellShapes() throws {
        let artifacts = try preparedHibikiArtifacts(configJSON: validConfigJSON())
        let request = try makeLoadRequest(artifacts: artifacts)
        let config = try MLXHibikiModelConfig.load(from: request.configURL)

        let model = MLXHibikiLanguageModel(topology: config.topology)

        #expect(model.textEmbedding.weightShape == [48_001, 2_048])
        #expect(model.audioEmbeddings.count == 32)
        #expect(model.audioEmbeddings[0].weightShape == [2_049, 2_048])
        #expect(model.transformer.layers.count == 28)
        #expect(model.mainTransformerCache.count == 28)
        #expect(model.transformer.layers[0].selfAttention.keyValueHeadCount == 8)
        #expect(model.transformer.layers[0].selfAttention.qkvProjection.weightShape == [4_096, 2_048])
        #expect(model.textLinear.weightShape == [48_000, 2_048])
        #expect(model.depformerSlices.count == 16)
        #expect(model.depformerSlices[0].embedding.weightShape == [48_001, 1_024])
        #expect(model.depformerSlices[1].embedding.weightShape == [2_049, 1_024])
        #expect(model.depformerSlices[0].linearIn.weightShape == [1_024, 2_048])
        #expect(model.depformerSlices[0].linearOut.weightShape == [2_048, 1_024])
        #expect(model.depformerSlices[0].norm.weightShape == [1_024])
        #expect(model.depformerSlices[0].norm.biasShape == [1_024])
        #expect(model.depformerSlices[0].transformer.layers.count == 6)
        #expect(model.depformerSlices[0].transformer.layers[0].selfAttention.qkvProjection.weightShape == [3_072, 1_024])
    }

    @Test("MLX Hibiki language model executes cached main step")
    func mlxHibikiLanguageModelExecutesCachedMainStep() throws {
        let model = MLXHibikiLanguageModel(topology: tinyHibikiTopology())

        let first = try model.mainStep(textToken: model.textPaddingToken, audioTokens: [0, model.audioPaddingToken])
        let second = try model.mainStep(textToken: 3, audioTokens: [1, 2])

        #expect(first.transformerOutput.shape == [1, 1, 8])
        #expect(first.textLogits.shape == [1, 16])
        #expect(second.transformerOutput.shape == [1, 1, 8])
        #expect(model.mainTransformerCache.allSatisfy { $0.offset == 2 })

        model.resetMainCache()

        #expect(model.mainTransformerCache.allSatisfy { $0.offset == 0 })
    }

    @Test("MLX Hibiki language model validates main step audio token count")
    func mlxHibikiLanguageModelValidatesMainStepAudioTokenCount() throws {
        let model = MLXHibikiLanguageModel(topology: tinyHibikiTopology())

        #expect(
            throws: HibikiInferenceError.invalidArtifacts("main step expected 2 audio tokens, got 1")
        ) {
            _ = try model.mainStep(textToken: 0, audioTokens: [0])
        }
    }

    @Test("MLX Hibiki graph parameter applier matches q4 artifact target topology")
    func mlxHibikiGraphParameterApplierMatchesQ4ArtifactTargetTopology() throws {
        let artifacts = try preparedHibikiArtifacts(configJSON: validConfigJSON())
        let request = try makeLoadRequest(artifacts: artifacts)
        let config = try MLXHibikiModelConfig.load(from: request.configURL)
        let model = MLXHibikiLanguageModel(topology: config.topology)

        let shapes = MLXHibikiGraphParameterApplier.expectedShapes(for: model)

        #expect(shapes.count == 2_015)
        #expect(shapes["text_emb.weight"] == [48_001, 256])
        #expect(shapes["text_emb.scales"] == [48_001, 64])
        #expect(shapes["audio_embs.0.weight"] == [2_049, 256])
        #expect(shapes["out_norm.weight"] == [2_048])
        #expect(shapes["text_linear.weight"] == [48_000, 256])
        #expect(shapes["transformer.layers.0.self_attn.in_proj.weight"] == [4_096, 256])
        #expect(shapes["transformer.layers.0.self_attn.in_proj.scales"] == [4_096, 64])
        #expect(shapes["transformer.layers.0.gating.linear_in.weight"] == [16_384, 256])
        #expect(shapes["transformer.layers.0.gating.linear_out.weight"] == [2_048, 1_024])
        #expect(shapes["depformer.slices.0.emb.weight"] == [48_001, 128])
        #expect(shapes["depformer.slices.1.emb.weight"] == [2_049, 128])
        #expect(shapes["depformer.slices.0.norm.bias"] == [1_024])
        #expect(shapes["depformer.slices.0.transformer.layers.0.self_attn.in_proj.weight"] == [3_072, 128])
        #expect(shapes["depformer.slices.0.transformer.layers.0.gating.linear_out.weight"] == [1_024, 512])
    }

    @Test("MLX Hibiki graph parameter applier assigns q4 groups and dense norms")
    func mlxHibikiGraphParameterApplierAssignsQ4GroupsAndDenseNorms() throws {
        let model = MLXHibikiLanguageModel(topology: tinyHibikiTopology())
        let quantization = MLXHibikiQuantizationSpec(bits: 4, groupSize: 4)
        let requiredKeys: Set<String> = [
            "text_emb.weight",
            "text_emb.scales",
            "text_emb.biases",
            "out_norm.weight",
            "depformer.slices.0.norm.weight",
            "depformer.slices.0.norm.bias",
        ]
        let weights: [String: MLXMimiWeightTensor] = [
            "text_emb.weight": mlxTensor([16, 1], type: UInt32.self),
            "text_emb.scales": mlxTensor([16, 2]),
            "text_emb.biases": mlxTensor([16, 2]),
            "out_norm.weight": mlxTensor([8]),
            "depformer.slices.0.norm.weight": mlxTensor([8]),
            "depformer.slices.0.norm.bias": mlxTensor([8]),
        ]

        try MLXHibikiGraphParameterApplier(
            requiredKeys: requiredKeys,
            quantization: quantization
        )
        .apply(weights, to: model)

        #expect(model.textEmbedding.quantizedParameters?.weight.shape == [16, 1])
        #expect(model.textEmbedding.quantizedParameters?.scales.shape == [16, 2])
        #expect(model.textEmbedding.quantizedParameters?.groupSize == 4)
        #expect(model.outNorm.rmsNorm?.weight.shape == [8])
        #expect(model.depformerSlices[0].norm.weight.shape == [8])
        #expect(model.depformerSlices[0].norm.bias.shape == [8])
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

    @Test("MLX Hibiki default engine rejects incompatible codebook topology")
    func mlxHibikiDefaultEngineRejectsIncompatibleCodebookTopology() throws {
        let artifacts = try preparedHibikiArtifacts(
            configJSON: validConfigJSON(generatedCodebookCount: 15)
        )
        let request = try makeLoadRequest(artifacts: artifacts)
        let engine = MLXHibikiDefaultRuntimeEngine { _ in
            MLXHibikiWeightsSummary(tensorCount: 1, hasDepformerOutputLayerNorms: true)
        }

        #expect(
            throws: HibikiInferenceError.invalidArtifacts("source codebook count expected 16, got 17")
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

    @Test("real file backend builds MLX Mimi and Hibiki components after preparation")
    @MainActor
    func realFileBackendBuildsMLXMimiAndHibikiComponentsAfterPreparation() async throws {
        let artifacts = try preparedHibikiArtifacts(configJSON: validConfigJSON())
        let mimiEngine = FakeRealBackendMimiRuntimeEngine()
        let hibikiEngine = RecordingHibikiRuntimeEngine(
            output: MLXHibikiStepOutput(
                textToken: 901,
                textPiece: " mlx",
                textCandidateTokens: [901],
                audioTokens: Array(1_000..<1_016)
            )
        )
        let session = ExperimentSession(
            backend: RealFileHibikiTranslationExperimentBackend(
                artifactPreparer: ModelArtifactPreparer(
                    manifest: .hibikiQ4Default,
                    provider: PreparedArtifactProvider(artifacts: artifacts)
                ),
                audioSource: FixtureAudioInputSource(sampleRate: 24_000, chunkSampleCount: 1_920, chunkCount: 2),
                inferenceSession: MLXHibikiInferenceSession(engine: hibikiEngine),
                playbackSink: BufferedPlaybackSink(),
                mimiRuntimeLoader: { artifacts in
                    let mimiArtifact = artifacts.files.first { $0.role == "mimiWeights" }!
                    return MLXMimiRuntime(artifact: mimiArtifact, engine: mimiEngine)
                }
            )
        )

        await session.prepare()
        await session.start()

        #expect(session.state == .running)
        #expect(mimiEngine.encodeInputs.count == 2)
        #expect(mimiEngine.decodeInputs.count == 2)
        #expect(hibikiEngine.stepRequests.count == 2)
        #expect(session.observations.mimiEncodedFrameCount == 2)
        #expect(session.observations.hibikiStepCount == 2)
        #expect(session.observations.decodedAudioChunkCount == 2)
        #expect(session.observations.playbackChunkCount == 2)
        #expect(session.observations.output == " mlx mlx")
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
    var outputs: [MLXHibikiStepOutput]

    init(
        output: MLXHibikiStepOutput = MLXHibikiStepOutput(
            textToken: 0,
            audioTokens: Array(repeating: 0, count: 16)
        )
    ) {
        self.outputs = [output]
    }

    init(outputs: [MLXHibikiStepOutput]) {
        self.outputs = outputs
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
        if outputs.isEmpty {
            return MLXHibikiStepOutput(
                textToken: 0,
                audioTokens: Array(repeating: 0, count: 16)
            )
        }
        if outputs.count == 1 {
            return outputs[0]
        }
        return outputs.removeFirst()
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

private final class FakeRealBackendMimiRuntimeEngine: MLXMimiRuntimeEngine, @unchecked Sendable {
    var resetEncodeCount = 0
    var resetDecodeCount = 0
    var encodeInputs: [MLXMimiPCMInput] = []
    var decodeInputs: [MLXMimiTokenInput] = []

    func resetEncodeState() {
        resetEncodeCount += 1
    }

    func resetDecodeState() {
        resetDecodeCount += 1
    }

    func warmup(request: MLXMimiWarmupRequest) throws {}

    func encode(_ input: MLXMimiPCMInput) throws -> [MLXMimiEncodedFrame] {
        encodeInputs.append(input)
        return [
            MLXMimiEncodedFrame(tokens: Array(300..<316)),
        ]
    }

    func decode(_ input: MLXMimiTokenInput) throws -> [MLXMimiDecodedChunk] {
        decodeInputs.append(input)
        return [MLXMimiDecodedChunk(samples: [0.1, -0.1])]
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

private func tinyHibikiTopology() -> MLXHibikiModelTopology {
    MLXHibikiModelTopology(
        mainTransformer: MLXMimiTransformerConfiguration(
            modelDimension: 8,
            headCount: 2,
            layerCount: 1,
            causal: true,
            normFirst: true,
            feedForwardBias: false,
            attentionBias: false,
            layerScale: nil,
            positionalEmbedding: .ropeConcat,
            usesConvBias: true,
            gating: true,
            norm: .rmsNorm,
            context: 4,
            maxPeriod: 16,
            maxSequenceLength: 16,
            kvRepeat: 1,
            feedForwardDimension: 24,
            convLayout: false,
            usesRotatingKVCache: true
        ),
        depformerTransformer: MLXMimiTransformerConfiguration(
            modelDimension: 8,
            headCount: 2,
            layerCount: 1,
            causal: true,
            normFirst: true,
            feedForwardBias: false,
            attentionBias: false,
            layerScale: nil,
            positionalEmbedding: .none,
            usesConvBias: true,
            gating: true,
            norm: .layerNorm,
            context: 1,
            maxPeriod: 8,
            maxSequenceLength: 16,
            kvRepeat: 1,
            feedForwardDimension: 24,
            convLayout: false,
            usesRotatingKVCache: false
        ),
        textInputVocabSize: 16,
        textOutputVocabSize: 16,
        audioVocabSize: 16,
        audioCodebookCount: 2,
        generatedCodebookCount: 1,
        audioDelays: [0, 0],
        depformerWeightsPerStepSchedule: nil
    )
}

private func mlxTensor(_ shape: [Int], type: Float32.Type = Float32.self) -> MLXMimiWeightTensor {
    MLXMimiWeightTensor(shape: shape, array: MLXArray.zeros(shape, type: type))
}

private func mlxTensor(_ shape: [Int], type: UInt32.Type) -> MLXMimiWeightTensor {
    MLXMimiWeightTensor(shape: shape, array: MLXArray.zeros(shape, type: type))
}

private func validConfigJSON(
    hiddenScale: Int = 6,
    kvRepeat: Int = 2,
    positionalEmbedding: String = "rope_concat",
    generatedCodebookCount: Int = 16
) -> String {
    """
    {
      "card": 2048,
      "n_q": 32,
      "dep_q": \(generatedCodebookCount),
      "delays": [
        0,
        0, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2,
        0, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2
      ],
      "dim": 2048,
      "text_card": 48000,
      "num_heads": 16,
      "num_layers": 28,
      "hidden_scale": \(hiddenScale),
      "causal": true,
      "layer_scale": null,
      "context": 3000,
      "max_period": 20000.0,
      "gating": "silu",
      "norm": "rms_norm_f32",
      "positional_embedding": "\(positionalEmbedding)",
      "depformer_dim": 1024,
      "depformer_num_heads": 16,
      "depformer_num_layers": 6,
      "depformer_dim_feedforward": null,
      "depformer_norm": "layer_norm",
      "depformer_pos_emb": "none",
      "depformer_weights_per_step_schedule": [
        0, 1, 2, 3, 4, 5, 6, 7,
        8, 8, 8, 8, 8, 8, 8, 8
      ],
      "demux_second_stream": false,
      "kv_repeat": \(kvRepeat),
      "depformer_kv_repeat": 1,
      "text_card_out": null,
      "conditioners": {},
      "cross_attention": false
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

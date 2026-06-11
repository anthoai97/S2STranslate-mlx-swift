import Foundation
import Testing

@testable import S2STranslateCore

@Suite("Streaming Hibiki Inference")
struct StreamingHibikiInferenceTests {
    @Test("deterministic inference initializes from prepared artifacts with sampling config")
    func deterministicInferenceInitializesFromPreparedArtifacts() async throws {
        let inference = DeterministicHibikiInferenceSession()
        let configuration = HibikiGenerationConfiguration(
            temperature: 0.8,
            textTemperature: 0.8,
            topK: 250,
            textTopK: 250,
            voiceTransferEnabled: false
        )

        let description = try await inference.initialize(
            artifacts: preparedArtifacts(),
            configuration: configuration
        )

        #expect(description.modelRevision == ModelRuntimeManifest.hibikiQ4Default.revision)
        #expect(description.artifactCount == 4)
        #expect(description.configuration == configuration)
    }

    @Test("deterministic inference emits text and generated audio token frames")
    func deterministicInferenceEmitsTextAndGeneratedAudio() async throws {
        let inference = DeterministicHibikiInferenceSession(visibleTextPieces: [" bonjour"])
        _ = try await inference.initialize(
            artifacts: preparedArtifacts(),
            configuration: HibikiGenerationConfiguration()
        )

        let first = try await inference.step(sourceAudioTokens: sourceTokenFrame(frameIndex: 0))
        let second = try await inference.step(sourceAudioTokens: sourceTokenFrame(frameIndex: 1))

        #expect(first.text.token == 0)
        #expect(first.text.piece == nil)
        #expect(first.generatedAudioTokens.tokens.prefix(2) == [501, 502])
        #expect(second.text.token == 501)
        #expect(second.text.piece == " bonjour")
        #expect(second.generatedAudioTokens.frameIndex == 1)
    }

    @Test("top-k sampler reports candidates and supports greedy selection")
    func topKSamplerReportsCandidatesAndSupportsGreedySelection() throws {
        let sampler = HibikiTopKTokenSampler()

        let sample = try sampler.sample(
            logits: [0.5, 1.5, 1.0, -2.0],
            temperature: 0,
            topK: 2,
            randomUnit: 1
        )

        #expect(sample == HibikiSampledToken(token: 1, candidateTokens: [1, 2]))
    }

    @Test("top-k sampler samples only within configured candidate set")
    func topKSamplerSamplesOnlyWithinConfiguredCandidateSet() throws {
        let sampler = HibikiTopKTokenSampler()

        let first = try sampler.sample(
            logits: [0, 0, -100],
            temperature: 1,
            topK: 2,
            randomUnit: 0
        )
        let second = try sampler.sample(
            logits: [0, 0, -100],
            temperature: 1,
            topK: 2,
            randomUnit: 1
        )

        #expect(first == HibikiSampledToken(token: 0, candidateTokens: [0, 1]))
        #expect(second == HibikiSampledToken(token: 1, candidateTokens: [0, 1]))
    }

    @Test("top-k sampler rejects malformed sampling inputs")
    func topKSamplerRejectsMalformedSamplingInputs() throws {
        let sampler = HibikiTopKTokenSampler()

        #expect(throws: HibikiTokenSamplingError.emptyLogits) {
            _ = try sampler.sample(logits: [], temperature: 1, topK: 1, randomUnit: 0)
        }
        #expect(throws: HibikiTokenSamplingError.invalidTemperature(-0.1)) {
            _ = try sampler.sample(logits: [0], temperature: -0.1, topK: 1, randomUnit: 0)
        }
        #expect(throws: HibikiTokenSamplingError.invalidRandomValue(1.1)) {
            _ = try sampler.sample(logits: [0], temperature: 1, topK: 1, randomUnit: 1.1)
        }
    }

    @Test("translation backend streams source tokens through Hibiki, decode, and playback")
    @MainActor
    func translationBackendStreamsThroughHibikiDecodeAndPlayback() async {
        let session = ExperimentSession(
            backend: HibikiTranslationExperimentBackend(
                artifactPreparer: ModelArtifactPreparer(
                    manifest: .hibikiQ4Default,
                    provider: DemoModelArtifactProvider()
                ),
                audioSource: FixtureAudioInputSource(sampleRate: 24_000, chunkSampleCount: 1_920, chunkCount: 4),
                mimiEncoder: DeterministicMimiStreamingEncoder(),
                inferenceSession: DeterministicHibikiInferenceSession(),
                mimiDecoder: DeterministicMimiStreamingDecoder(),
                playbackSink: BufferedPlaybackSink()
            )
        )

        await session.prepare()
        await session.start()

        #expect(session.state == .running)
        #expect(session.observations.progress == 1)
        #expect(session.observations.hibikiInferenceStatus == "stopped")
        #expect(session.observations.hibikiStepCount == 4)
        #expect(session.observations.hibikiTextTokenCount == 4)
        #expect(session.observations.hibikiVisibleTextCount == 2)
        #expect(session.observations.hibikiGeneratedAudioFrameCount == 4)
        #expect(session.observations.hibikiSamplingSummary == "temp 0.8, top-k 250")
        #expect(session.observations.output == " hello world")
        #expect(session.observations.decodedAudioChunkCount == 4)
        #expect(session.observations.playbackChunkCount == 4)
        #expect(session.observations.lastEventName == "playback:streamStopped")
    }

    @Test("translation backend flushes delayed text after file input ends")
    @MainActor
    func translationBackendFlushesDelayedTextAfterFileInputEnds() async {
        let session = ExperimentSession(
            backend: HibikiTranslationExperimentBackend(
                artifactPreparer: ModelArtifactPreparer(
                    manifest: .hibikiQ4Default,
                    provider: DemoModelArtifactProvider()
                ),
                audioSource: FixtureAudioInputSource(sampleRate: 24_000, chunkSampleCount: 1_920, chunkCount: 1),
                mimiEncoder: DeterministicMimiStreamingEncoder(),
                inferenceSession: DeterministicHibikiInferenceSession(visibleTextPieces: [" tail"]),
                mimiDecoder: DeterministicMimiStreamingDecoder(),
                playbackSink: BufferedPlaybackSink(),
                generationConfiguration: HibikiGenerationConfiguration(
                    tailSilenceFrameCount: 8,
                    postInputPaddingStopFrameCount: 3
                )
            )
        )

        await session.prepare()
        await session.start()

        #expect(session.state == .running)
        #expect(session.observations.audioChunkCount == 1)
        #expect(session.observations.mimiEncodedFrameCount == 5)
        #expect(session.observations.hibikiStepCount == 5)
        #expect(session.observations.hibikiVisibleTextCount == 1)
        #expect(session.observations.output == " tail")
        #expect(session.observations.hibikiGeneratedAudioFrameCount == 5)
        #expect(session.observations.playbackChunkCount == 5)
    }

    @Test("inference failures reach session failure state")
    @MainActor
    func inferenceFailuresReachSessionFailureState() async {
        let session = ExperimentSession(
            backend: HibikiTranslationExperimentBackend(
                artifactPreparer: ModelArtifactPreparer(
                    manifest: .hibikiQ4Default,
                    provider: DemoModelArtifactProvider()
                ),
                audioSource: FixtureAudioInputSource(sampleRate: 24_000, chunkSampleCount: 1_920, chunkCount: 1),
                mimiEncoder: DeterministicMimiStreamingEncoder(),
                inferenceSession: FailingHibikiInferenceSession(error: .unavailable("hibiki weights missing")),
                mimiDecoder: DeterministicMimiStreamingDecoder(),
                playbackSink: BufferedPlaybackSink()
            )
        )

        await session.prepare()

        #expect(session.state == .failed("Hibiki inference unavailable: hibiki weights missing"))
        #expect(session.observations.hibikiInferenceStatus == "failed: Hibiki inference unavailable: hibiki weights missing")
    }

    @Test("stepping before initialization fails")
    func steppingBeforeInitializationFails() async {
        let inference = DeterministicHibikiInferenceSession()

        do {
            _ = try await inference.step(sourceAudioTokens: sourceTokenFrame(frameIndex: 0))
            Issue.record("Expected uninitialized inference session to fail")
        } catch let error as HibikiInferenceError {
            #expect(error == .notInitialized)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test("first Hibiki step matches reference trace model and text events")
    func firstHibikiStepMatchesReferenceTraceEvents() async throws {
        let expected = try loadFixtureTrace()
        let inference = DeterministicHibikiInferenceSession()
        _ = try await inference.initialize(
            artifacts: preparedArtifacts(),
            configuration: HibikiGenerationConfiguration()
        )

        let step = try await inference.step(sourceAudioTokens: sourceTokenFrame(frameIndex: 0))
        let expectedPrefix = ReferenceTrace(
            name: expected.name,
            source: expected.source,
            events: Array(expected.events[2...3])
        )
        let actual = ReferenceTrace(
            name: expected.name,
            source: expected.source,
            events: [
                step.referenceTraceEvent,
                step.text.referenceTraceEvent,
            ]
        )

        let mismatches = ReferenceTraceComparator.compare(expected: expectedPrefix, actual: actual)

        #expect(mismatches.isEmpty)
    }
}

private func preparedArtifacts() -> PreparedModelArtifacts {
    let files = ModelRuntimeManifest.hibikiQ4Default.requiredFiles.map { requirement in
        PreparedModelArtifact(
            role: requirement.role,
            fileName: requirement.fileName,
            location: "test://\(requirement.fileName)",
            source: .cache
        )
    }
    return PreparedModelArtifacts(manifest: .hibikiQ4Default, files: files)
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

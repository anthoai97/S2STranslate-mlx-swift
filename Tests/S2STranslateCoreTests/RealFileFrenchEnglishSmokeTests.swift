import AVFoundation
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
        let playbackSink = BufferedPlaybackSink()
        let session = ExperimentSession(
            backend: RealFileHibikiTranslationExperimentBackend(
                artifactPreparer: ModelArtifactPreparer(
                    manifest: .hibikiQ4Default,
                    provider: try LocalSmokeArtifactProvider(artifacts: artifacts)
                ),
                audioSource: smokeAudioSource(environment: environment),
                playbackSink: playbackSink,
                generationConfiguration: HibikiGenerationConfiguration(
                    textTemperature: 0.4,
                    textTopK: 25,
                    tailSilenceFrameCount: 100,
                    postInputPaddingStopFrameCount: 12
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

        let outputDirectory = smokeOutputDirectory(environment: environment)
        let textURL = outputDirectory.appendingPathComponent("translation.txt")
        let audioURL = outputDirectory.appendingPathComponent("translation.wav")
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        try session.observations.output.write(to: textURL, atomically: true, encoding: .utf8)
        try writeSmokeWAV(chunks: playbackSink.bufferedChunks(), to: audioURL)
        print("Real file smoke text: \(textURL.path)")
        print("Real file smoke audio: \(audioURL.path)")
    }

    @Test("French Europarl short 1 benchmarks the real MLX model flow")
    @MainActor
    func frenchEuroparlShortOneBenchmarksRealMLXModelFlow() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["S2S_RUN_REAL_FILE_BENCHMARKS"] == "1" else {
            return
        }

        let weightsDirectory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(
                environment["S2S_REAL_FILE_SMOKE_WEIGHTS_DIR"] ?? "ref/hibiki-zero-mlx/weights",
                isDirectory: true
        )
        let artifacts = try localHibikiArtifacts(weightsDirectory: weightsDirectory)
        let generationConfiguration = HibikiGenerationConfiguration(
            textTemperature: 0.4,
            textTopK: 25,
            tailSilenceFrameCount: 100,
            postInputPaddingStopFrameCount: 12
        )

        let loadStartedAt = Date()
        let runtime = try MLXMimiRuntimeLoader().load(from: artifacts)
        let runtimeLoadSeconds = Date().timeIntervalSince(loadStartedAt)
        let encoder = MLXMimiStreamingEncoder(runtime: runtime)
        let decoder = MLXMimiStreamingDecoder(runtime: runtime)
        let inference = MLXHibikiInferenceSession()

        let initializeStartedAt = Date()
        let inferenceDescription = try await inference.initialize(
            artifacts: artifacts,
            configuration: generationConfiguration
        )
        let initializeSeconds = Date().timeIntervalSince(initializeStartedAt)

        let source = smokeAudioSource(environment: environment)
        source.reset()
        encoder.reset()
        decoder.reset()

        let sourceLoadStartedAt = Date()
        let sourceChunks = try await source.chunks()
        let sourceLoadSeconds = Date().timeIntervalSince(sourceLoadStartedAt)
        let benchmarkedChunks = Array(sourceChunks.prefix(benchmarkSourceChunkLimit(environment: environment) ?? sourceChunks.count))
        let encoderDescription = await encoder.description()
        let decoderDescription = await decoder.description()

        var benchmark = RealFileModelFlowBenchmark(
            sourceAudioChunkCount: benchmarkedChunks.count,
            sourceAudioDurationSeconds: benchmarkedChunks.reduce(0) { $0 + $1.durationMilliseconds / 1000 },
            decoderSampleRate: decoderDescription.sampleRate,
            runtimeLoadSeconds: runtimeLoadSeconds,
            inferenceInitializeSeconds: initializeSeconds,
            sourceLoadSeconds: sourceLoadSeconds,
            modelRevision: inferenceDescription.modelRevision
        )
        var decodedChunks: [DecodedAudioChunk] = []
        var visibleText = ""

        let processingStartedAt = Date()
        for chunk in benchmarkedChunks {
            let encodeStartedAt = Date()
            let sourceTokenFrames = try await encoder.encode(chunk)
            benchmark.encodeMilliseconds.append(Date().timeIntervalSince(encodeStartedAt) * 1000)
            benchmark.encodedFrameCount += sourceTokenFrames.count

            for sourceTokens in sourceTokenFrames {
                let frameOutput = try await benchmarkGeneratedFrame(
                    sourceTokens: sourceTokens,
                    inference: inference,
                    decoder: decoder,
                    benchmark: &benchmark
                )
                visibleText += frameOutput.visibleText
                decodedChunks.append(contentsOf: frameOutput.decodedChunks)
            }
        }

        if benchmarkIncludesTail(environment: environment) {
            var stopDetector = BenchmarkPostInputStopDetector(
                requiredBlankOrPaddingFrameCount: generationConfiguration.postInputPaddingStopFrameCount
            )
            let startAudioFrameIndex = (benchmarkedChunks.last?.frameIndex ?? -1) + 1
            for tailFrameIndex in 0..<generationConfiguration.tailSilenceFrameCount {
                let chunkFrameIndex = startAudioFrameIndex + tailFrameIndex
                let chunk = PCMChunk(
                    frameIndex: chunkFrameIndex,
                    timestampMilliseconds: Double(chunkFrameIndex * encoderDescription.samplesPerFrame)
                        / Double(encoderDescription.sampleRate) * 1000,
                    sampleRate: encoderDescription.sampleRate,
                    samples: Array(repeating: 0, count: encoderDescription.samplesPerFrame)
                )
                let encodeStartedAt = Date()
                let sourceTokenFrames = try await encoder.encode(chunk)
                benchmark.tailEncodeMilliseconds.append(Date().timeIntervalSince(encodeStartedAt) * 1000)
                benchmark.tailEncodedFrameCount += sourceTokenFrames.count

                for sourceTokens in sourceTokenFrames {
                    let frameOutput = try await benchmarkGeneratedFrame(
                        sourceTokens: sourceTokens,
                        inference: inference,
                        decoder: decoder,
                        benchmark: &benchmark
                    )
                    visibleText += frameOutput.visibleText
                    decodedChunks.append(contentsOf: frameOutput.decodedChunks)
                    if stopDetector.shouldStop(after: frameOutput.text) {
                        break
                    }
                }
                if stopDetector.hasStopped {
                    break
                }
            }
        }

        benchmark.processingSeconds = Date().timeIntervalSince(processingStartedAt)
        benchmark.decodedAudioChunkCount = decodedChunks.count
        benchmark.decodedAudioDurationSeconds = decodedChunks.reduce(0) { $0 + $1.durationMilliseconds / 1000 }
        benchmark.visibleTextCharacterCount = visibleText.count

        let outputDirectory = benchmarkOutputDirectory(environment: environment)
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        let jsonURL = outputDirectory.appendingPathComponent("benchmark.json")
        let markdownURL = outputDirectory.appendingPathComponent("benchmark.md")
        let textURL = outputDirectory.appendingPathComponent("translation.txt")
        let audioURL = outputDirectory.appendingPathComponent("translation.wav")
        let jsonEncoder = JSONEncoder()
        jsonEncoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try jsonEncoder.encode(benchmark.report()).write(to: jsonURL)
        try benchmark.markdownReport().write(to: markdownURL, atomically: true, encoding: .utf8)
        try visibleText.write(to: textURL, atomically: true, encoding: .utf8)
        try writeSmokeWAV(chunks: decodedChunks, to: audioURL)

        print("Real file benchmark JSON: \(jsonURL.path)")
        print("Real file benchmark markdown: \(markdownURL.path)")
        print("Real file benchmark text: \(textURL.path)")
        print("Real file benchmark audio: \(audioURL.path)")
    }

    @Test("benchmark report includes Hibiki substage timing summaries")
    func benchmarkReportIncludesHibikiSubstageTimingSummaries() {
        var benchmark = RealFileModelFlowBenchmark(
            sourceAudioChunkCount: 40,
            sourceAudioDurationSeconds: 3.2,
            decoderSampleRate: 24_000,
            runtimeLoadSeconds: 1,
            inferenceInitializeSeconds: 2,
            sourceLoadSeconds: 0.5,
            modelRevision: "test"
        )
        benchmark.hibikiStepMilliseconds = [100, 140]
        benchmark.hibikiMainTransformerEvaluationMilliseconds = [40, 60]
        benchmark.hibikiTextLogitsExtractionMilliseconds = [4, 6]
        benchmark.hibikiTextSamplingMilliseconds = [2, 4]
        benchmark.hibikiDepformerEvaluationMilliseconds = [30, 50]
        benchmark.hibikiDepformerLogitsExtractionMilliseconds = [8, 12]
        benchmark.hibikiDepformerSamplingMilliseconds = [6, 10]
        benchmark.hibikiStateCacheUpdateMilliseconds = [1, 3]
        benchmark.hibikiGeneratedFrameConstructionMilliseconds = [0.5, 1.5]

        let report = benchmark.report()
        let markdown = benchmark.markdownReport()

        #expect(report.hibikiMainTransformerEvaluationMilliseconds.average == 50)
        #expect(report.hibikiTextLogitsExtractionMilliseconds.p50 == 6)
        #expect(report.hibikiDepformerSamplingMilliseconds.max == 10)
        #expect(markdown.contains("| Hibiki step | 2 | 120.000"))
        #expect(markdown.contains("| Hibiki main transformer evaluation | 2 | 50.000"))
        #expect(markdown.contains("| Hibiki Depformer logits extraction | 2 | 10.000"))
        #expect(markdown.contains("| Hibiki generated frame construction | 2 | 1.000"))
    }
}

private func smokeOutputDirectory(environment: [String: String]) -> URL {
    if let outputPath = environment["S2S_REAL_FILE_SMOKE_OUTPUT_DIR"], !outputPath.isEmpty {
        return URL(fileURLWithPath: outputPath, isDirectory: true)
    }
    return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        .appendingPathComponent(".scratch/real-file-smoke/latest", isDirectory: true)
}

private func benchmarkOutputDirectory(environment: [String: String]) -> URL {
    if let outputPath = environment["S2S_REAL_FILE_BENCHMARK_OUTPUT_DIR"], !outputPath.isEmpty {
        return URL(fileURLWithPath: outputPath, isDirectory: true)
    }
    return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        .appendingPathComponent(".scratch/real-file-benchmark/latest", isDirectory: true)
}

private func benchmarkSourceChunkLimit(environment: [String: String]) -> Int? {
    guard let value = environment["S2S_REAL_FILE_BENCHMARK_MAX_SOURCE_CHUNKS"],
          let limit = Int(value),
          limit > 0 else {
        return nil
    }
    return limit
}

private func benchmarkIncludesTail(environment: [String: String]) -> Bool {
    environment["S2S_REAL_FILE_BENCHMARK_INCLUDE_TAIL"] != "0"
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

private func writeSmokeWAV(chunks: [DecodedAudioChunk], to url: URL) throws {
    guard let sampleRate = chunks.first?.sampleRate else {
        throw PlaybackSinkError.unavailable("no decoded chunks to write")
    }
    let samples = chunks.flatMap(\.samples)
    guard !samples.isEmpty else {
        throw PlaybackSinkError.unavailable("no decoded samples to write")
    }
    guard let format = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: Double(sampleRate),
        channels: 1,
        interleaved: false
    ) else {
        throw PlaybackSinkError.unavailable("could not create WAV format")
    }
    guard let buffer = AVAudioPCMBuffer(
        pcmFormat: format,
        frameCapacity: AVAudioFrameCount(samples.count)
    ) else {
        throw PlaybackSinkError.unavailable("could not create WAV buffer")
    }
    buffer.frameLength = AVAudioFrameCount(samples.count)
    if let channel = buffer.floatChannelData?[0] {
        for (index, sample) in samples.enumerated() {
            channel[index] = sample
        }
    }
    let settings: [String: Any] = [
        AVFormatIDKey: kAudioFormatLinearPCM,
        AVSampleRateKey: Double(sampleRate),
        AVNumberOfChannelsKey: 1,
        AVLinearPCMBitDepthKey: 32,
        AVLinearPCMIsFloatKey: true,
        AVLinearPCMIsBigEndianKey: false,
    ]
    let file = try AVAudioFile(forWriting: url, settings: settings)
    try file.write(from: buffer)
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

private func benchmarkGeneratedFrame(
    sourceTokens: MimiTokenFrame,
    inference: MLXHibikiInferenceSession,
    decoder: MLXMimiStreamingDecoder,
    benchmark: inout RealFileModelFlowBenchmark
) async throws -> BenchmarkGeneratedFrameOutput {
    let stepStartedAt = Date()
    let step = try await inference.step(sourceAudioTokens: sourceTokens)
    benchmark.hibikiStepMilliseconds.append(Date().timeIntervalSince(stepStartedAt) * 1000)
    benchmark.hibikiStepCount += 1
    if let timings = step.timings {
        benchmark.hibikiMainTransformerEvaluationMilliseconds.append(
            timings.mainTransformerEvaluationMilliseconds
        )
        benchmark.hibikiTextLogitsExtractionMilliseconds.append(timings.textLogitsExtractionMilliseconds)
        benchmark.hibikiTextSamplingMilliseconds.append(timings.textSamplingMilliseconds)
        benchmark.hibikiDepformerEvaluationMilliseconds.append(timings.depformerEvaluationMilliseconds)
        benchmark.hibikiDepformerLogitsExtractionMilliseconds.append(
            timings.depformerLogitsExtractionMilliseconds
        )
        benchmark.hibikiDepformerSamplingMilliseconds.append(timings.depformerSamplingMilliseconds)
        benchmark.hibikiStateCacheUpdateMilliseconds.append(timings.stateCacheUpdateMilliseconds)
        benchmark.hibikiGeneratedFrameConstructionMilliseconds.append(
            timings.generatedFrameConstructionMilliseconds
        )
    }

    let decodeStartedAt = Date()
    let decodedChunks = try await decoder.decode(step.generatedAudioTokens)
    benchmark.decodeMilliseconds.append(Date().timeIntervalSince(decodeStartedAt) * 1000)
    benchmark.decodedSampleCount += decodedChunks.reduce(0) { $0 + $1.samples.count }

    return BenchmarkGeneratedFrameOutput(
        text: step.text,
        visibleText: step.text.piece ?? "",
        decodedChunks: decodedChunks
    )
}

private struct BenchmarkGeneratedFrameOutput {
    var text: HibikiTextOutput
    var visibleText: String
    var decodedChunks: [DecodedAudioChunk]
}

private struct BenchmarkPostInputStopDetector {
    private let requiredBlankOrPaddingFrameCount: Int
    private var blankOrPaddingRun = 0
    private(set) var hasStopped = false

    init(requiredBlankOrPaddingFrameCount: Int) {
        self.requiredBlankOrPaddingFrameCount = requiredBlankOrPaddingFrameCount
    }

    mutating func shouldStop(after text: HibikiTextOutput) -> Bool {
        guard requiredBlankOrPaddingFrameCount > 0 else { return false }
        if text.isBlankOrPadding {
            blankOrPaddingRun += 1
        } else {
            blankOrPaddingRun = 0
        }
        hasStopped = blankOrPaddingRun >= requiredBlankOrPaddingFrameCount
        return hasStopped
    }
}

private struct RealFileModelFlowBenchmark {
    var sourceAudioChunkCount: Int
    var sourceAudioDurationSeconds: Double
    var decoderSampleRate: Int
    var runtimeLoadSeconds: Double
    var inferenceInitializeSeconds: Double
    var sourceLoadSeconds: Double
    var modelRevision: String
    var processingSeconds: Double = 0
    var encodedFrameCount = 0
    var tailEncodedFrameCount = 0
    var hibikiStepCount = 0
    var decodedAudioChunkCount = 0
    var decodedSampleCount = 0
    var decodedAudioDurationSeconds: Double = 0
    var visibleTextCharacterCount = 0
    var encodeMilliseconds: [Double] = []
    var tailEncodeMilliseconds: [Double] = []
    var hibikiStepMilliseconds: [Double] = []
    var hibikiMainTransformerEvaluationMilliseconds: [Double] = []
    var hibikiTextLogitsExtractionMilliseconds: [Double] = []
    var hibikiTextSamplingMilliseconds: [Double] = []
    var hibikiDepformerEvaluationMilliseconds: [Double] = []
    var hibikiDepformerLogitsExtractionMilliseconds: [Double] = []
    var hibikiDepformerSamplingMilliseconds: [Double] = []
    var hibikiStateCacheUpdateMilliseconds: [Double] = []
    var hibikiGeneratedFrameConstructionMilliseconds: [Double] = []
    var decodeMilliseconds: [Double] = []

    func report() -> RealFileModelFlowBenchmarkReport {
        RealFileModelFlowBenchmarkReport(
            modelRevision: modelRevision,
            sourceAudioChunkCount: sourceAudioChunkCount,
            sourceAudioDurationSeconds: sourceAudioDurationSeconds,
            decoderSampleRate: decoderSampleRate,
            runtimeLoadSeconds: runtimeLoadSeconds,
            inferenceInitializeSeconds: inferenceInitializeSeconds,
            sourceLoadSeconds: sourceLoadSeconds,
            processingSeconds: processingSeconds,
            generatedAudioDurationSeconds: decodedAudioDurationSeconds,
            generatedRealtimeFactor: processingSeconds > 0 ? decodedAudioDurationSeconds / processingSeconds : 0,
            encodedFrameCount: encodedFrameCount,
            tailEncodedFrameCount: tailEncodedFrameCount,
            hibikiStepCount: hibikiStepCount,
            decodedAudioChunkCount: decodedAudioChunkCount,
            decodedSampleCount: decodedSampleCount,
            visibleTextCharacterCount: visibleTextCharacterCount,
            encodeMilliseconds: BenchmarkSummary(values: encodeMilliseconds),
            tailEncodeMilliseconds: BenchmarkSummary(values: tailEncodeMilliseconds),
            hibikiStepMilliseconds: BenchmarkSummary(values: hibikiStepMilliseconds),
            hibikiMainTransformerEvaluationMilliseconds: BenchmarkSummary(
                values: hibikiMainTransformerEvaluationMilliseconds
            ),
            hibikiTextLogitsExtractionMilliseconds: BenchmarkSummary(values: hibikiTextLogitsExtractionMilliseconds),
            hibikiTextSamplingMilliseconds: BenchmarkSummary(values: hibikiTextSamplingMilliseconds),
            hibikiDepformerEvaluationMilliseconds: BenchmarkSummary(values: hibikiDepformerEvaluationMilliseconds),
            hibikiDepformerLogitsExtractionMilliseconds: BenchmarkSummary(
                values: hibikiDepformerLogitsExtractionMilliseconds
            ),
            hibikiDepformerSamplingMilliseconds: BenchmarkSummary(values: hibikiDepformerSamplingMilliseconds),
            hibikiStateCacheUpdateMilliseconds: BenchmarkSummary(values: hibikiStateCacheUpdateMilliseconds),
            hibikiGeneratedFrameConstructionMilliseconds: BenchmarkSummary(
                values: hibikiGeneratedFrameConstructionMilliseconds
            ),
            decodeMilliseconds: BenchmarkSummary(values: decodeMilliseconds)
        )
    }

    func markdownReport() -> String {
        let report = report()
        return """
        # Real File Model Flow Benchmark

        - Model revision: `\(report.modelRevision)`
        - Source chunks: \(report.sourceAudioChunkCount)
        - Source duration: \(format(report.sourceAudioDurationSeconds))s
        - Generated audio duration: \(format(report.generatedAudioDurationSeconds))s
        - Processing time: \(format(report.processingSeconds))s
        - Generated realtime factor: \(format(report.generatedRealtimeFactor))x
        - Runtime load: \(format(report.runtimeLoadSeconds))s
        - Hibiki initialize: \(format(report.inferenceInitializeSeconds))s
        - Source load: \(format(report.sourceLoadSeconds))s
        - Encoded frames: \(report.encodedFrameCount)
        - Tail encoded frames: \(report.tailEncodedFrameCount)
        - Hibiki steps: \(report.hibikiStepCount)
        - Decoded chunks: \(report.decodedAudioChunkCount)
        - Visible text characters: \(report.visibleTextCharacterCount)

        | Stage | Count | Avg ms | P50 ms | P95 ms | Max ms |
        | --- | ---: | ---: | ---: | ---: | ---: |
        | Mimi encode | \(report.encodeMilliseconds.count) | \(format(report.encodeMilliseconds.average)) | \(format(report.encodeMilliseconds.p50)) | \(format(report.encodeMilliseconds.p95)) | \(format(report.encodeMilliseconds.max)) |
        | Tail Mimi encode | \(report.tailEncodeMilliseconds.count) | \(format(report.tailEncodeMilliseconds.average)) | \(format(report.tailEncodeMilliseconds.p50)) | \(format(report.tailEncodeMilliseconds.p95)) | \(format(report.tailEncodeMilliseconds.max)) |
        | Hibiki step | \(report.hibikiStepMilliseconds.count) | \(format(report.hibikiStepMilliseconds.average)) | \(format(report.hibikiStepMilliseconds.p50)) | \(format(report.hibikiStepMilliseconds.p95)) | \(format(report.hibikiStepMilliseconds.max)) |
        | Hibiki main transformer evaluation | \(report.hibikiMainTransformerEvaluationMilliseconds.count) | \(format(report.hibikiMainTransformerEvaluationMilliseconds.average)) | \(format(report.hibikiMainTransformerEvaluationMilliseconds.p50)) | \(format(report.hibikiMainTransformerEvaluationMilliseconds.p95)) | \(format(report.hibikiMainTransformerEvaluationMilliseconds.max)) |
        | Hibiki text logits extraction | \(report.hibikiTextLogitsExtractionMilliseconds.count) | \(format(report.hibikiTextLogitsExtractionMilliseconds.average)) | \(format(report.hibikiTextLogitsExtractionMilliseconds.p50)) | \(format(report.hibikiTextLogitsExtractionMilliseconds.p95)) | \(format(report.hibikiTextLogitsExtractionMilliseconds.max)) |
        | Hibiki text sampling | \(report.hibikiTextSamplingMilliseconds.count) | \(format(report.hibikiTextSamplingMilliseconds.average)) | \(format(report.hibikiTextSamplingMilliseconds.p50)) | \(format(report.hibikiTextSamplingMilliseconds.p95)) | \(format(report.hibikiTextSamplingMilliseconds.max)) |
        | Hibiki Depformer evaluation | \(report.hibikiDepformerEvaluationMilliseconds.count) | \(format(report.hibikiDepformerEvaluationMilliseconds.average)) | \(format(report.hibikiDepformerEvaluationMilliseconds.p50)) | \(format(report.hibikiDepformerEvaluationMilliseconds.p95)) | \(format(report.hibikiDepformerEvaluationMilliseconds.max)) |
        | Hibiki Depformer logits extraction | \(report.hibikiDepformerLogitsExtractionMilliseconds.count) | \(format(report.hibikiDepformerLogitsExtractionMilliseconds.average)) | \(format(report.hibikiDepformerLogitsExtractionMilliseconds.p50)) | \(format(report.hibikiDepformerLogitsExtractionMilliseconds.p95)) | \(format(report.hibikiDepformerLogitsExtractionMilliseconds.max)) |
        | Hibiki Depformer sampling | \(report.hibikiDepformerSamplingMilliseconds.count) | \(format(report.hibikiDepformerSamplingMilliseconds.average)) | \(format(report.hibikiDepformerSamplingMilliseconds.p50)) | \(format(report.hibikiDepformerSamplingMilliseconds.p95)) | \(format(report.hibikiDepformerSamplingMilliseconds.max)) |
        | Hibiki state/cache updates | \(report.hibikiStateCacheUpdateMilliseconds.count) | \(format(report.hibikiStateCacheUpdateMilliseconds.average)) | \(format(report.hibikiStateCacheUpdateMilliseconds.p50)) | \(format(report.hibikiStateCacheUpdateMilliseconds.p95)) | \(format(report.hibikiStateCacheUpdateMilliseconds.max)) |
        | Hibiki generated frame construction | \(report.hibikiGeneratedFrameConstructionMilliseconds.count) | \(format(report.hibikiGeneratedFrameConstructionMilliseconds.average)) | \(format(report.hibikiGeneratedFrameConstructionMilliseconds.p50)) | \(format(report.hibikiGeneratedFrameConstructionMilliseconds.p95)) | \(format(report.hibikiGeneratedFrameConstructionMilliseconds.max)) |
        | Mimi decode | \(report.decodeMilliseconds.count) | \(format(report.decodeMilliseconds.average)) | \(format(report.decodeMilliseconds.p50)) | \(format(report.decodeMilliseconds.p95)) | \(format(report.decodeMilliseconds.max)) |
        """
    }

    private func format(_ value: Double) -> String {
        String(format: "%.3f", value)
    }
}

private struct RealFileModelFlowBenchmarkReport: Codable {
    var modelRevision: String
    var sourceAudioChunkCount: Int
    var sourceAudioDurationSeconds: Double
    var decoderSampleRate: Int
    var runtimeLoadSeconds: Double
    var inferenceInitializeSeconds: Double
    var sourceLoadSeconds: Double
    var processingSeconds: Double
    var generatedAudioDurationSeconds: Double
    var generatedRealtimeFactor: Double
    var encodedFrameCount: Int
    var tailEncodedFrameCount: Int
    var hibikiStepCount: Int
    var decodedAudioChunkCount: Int
    var decodedSampleCount: Int
    var visibleTextCharacterCount: Int
    var encodeMilliseconds: BenchmarkSummary
    var tailEncodeMilliseconds: BenchmarkSummary
    var hibikiStepMilliseconds: BenchmarkSummary
    var hibikiMainTransformerEvaluationMilliseconds: BenchmarkSummary
    var hibikiTextLogitsExtractionMilliseconds: BenchmarkSummary
    var hibikiTextSamplingMilliseconds: BenchmarkSummary
    var hibikiDepformerEvaluationMilliseconds: BenchmarkSummary
    var hibikiDepformerLogitsExtractionMilliseconds: BenchmarkSummary
    var hibikiDepformerSamplingMilliseconds: BenchmarkSummary
    var hibikiStateCacheUpdateMilliseconds: BenchmarkSummary
    var hibikiGeneratedFrameConstructionMilliseconds: BenchmarkSummary
    var decodeMilliseconds: BenchmarkSummary
}

private struct BenchmarkSummary: Codable {
    var count: Int
    var average: Double
    var min: Double
    var p50: Double
    var p95: Double
    var max: Double

    init(values: [Double]) {
        let sorted = values.sorted()
        self.count = sorted.count
        self.average = sorted.isEmpty ? 0 : sorted.reduce(0, +) / Double(sorted.count)
        self.min = sorted.first ?? 0
        self.p50 = Self.percentile(0.50, sorted: sorted)
        self.p95 = Self.percentile(0.95, sorted: sorted)
        self.max = sorted.last ?? 0
    }

    private static func percentile(_ percentile: Double, sorted: [Double]) -> Double {
        guard !sorted.isEmpty else { return 0 }
        let index = Int((Double(sorted.count - 1) * percentile).rounded())
        return sorted[Swift.min(Swift.max(index, 0), sorted.count - 1)]
    }
}

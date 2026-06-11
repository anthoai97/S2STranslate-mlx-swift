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
}

private func smokeOutputDirectory(environment: [String: String]) -> URL {
    if let outputPath = environment["S2S_REAL_FILE_SMOKE_OUTPUT_DIR"], !outputPath.isEmpty {
        return URL(fileURLWithPath: outputPath, isDirectory: true)
    }
    return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        .appendingPathComponent(".scratch/real-file-smoke/latest", isDirectory: true)
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

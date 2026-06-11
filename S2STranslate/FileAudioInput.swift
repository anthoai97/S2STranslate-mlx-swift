@preconcurrency import AVFoundation
import Foundation

public struct FileAudioFixture: Equatable, Identifiable, Sendable {
    public var id: String
    public var title: String
    public var languageCode: String
    public var sourceURL: URL
    public var referenceOutputURL: URL?

    nonisolated public init(
        id: String,
        title: String,
        languageCode: String,
        sourceURL: URL,
        referenceOutputURL: URL? = nil
    ) {
        self.id = id
        self.title = title
        self.languageCode = languageCode
        self.sourceURL = sourceURL
        self.referenceOutputURL = referenceOutputURL
    }
}

public enum FileAudioFixtureCatalog {
    public static let frenchShortForm = [
        FileAudioFixture(
            id: "fr-europarl-30ef344",
            title: "French Europarl short 1",
            languageCode: "fr",
            sourceURL: URL(string: "https://huggingface.co/spaces/kyutai/hibiki-zero-samples/resolve/main/data/europarl_st/fr/source/30ef344ae8687926.mp3")!,
            referenceOutputURL: URL(string: "https://huggingface.co/spaces/kyutai/hibiki-zero-samples/resolve/main/data/europarl_st/fr/hibiki-zero/30ef344ae8687926.mp3")
        ),
        FileAudioFixture(
            id: "fr-europarl-4539f03",
            title: "French Europarl short 2",
            languageCode: "fr",
            sourceURL: URL(string: "https://huggingface.co/spaces/kyutai/hibiki-zero-samples/resolve/main/data/europarl_st/fr/source/4539f03d07ce7fbf.mp3")!,
            referenceOutputURL: URL(string: "https://huggingface.co/spaces/kyutai/hibiki-zero-samples/resolve/main/data/europarl_st/fr/hibiki-zero/4539f03d07ce7fbf.mp3")
        ),
    ]

    public static let frenchLongForm = [
        FileAudioFixture(
            id: "fr-ntrex-ee67adf",
            title: "French NTREX long 1",
            languageCode: "fr",
            sourceURL: URL(string: "https://huggingface.co/spaces/kyutai/hibiki-zero-samples/resolve/main/data/audio_ntrex_4L/fr/source/ee67adf3f3768b1d_11labs.mp3")!,
            referenceOutputURL: URL(string: "https://huggingface.co/spaces/kyutai/hibiki-zero-samples/resolve/main/data/audio_ntrex_4L/fr/hibiki-zero/ee67adf3f3768b1d_11labs.mp3")
        ),
        FileAudioFixture(
            id: "fr-ntrex-f9fcfb4",
            title: "French NTREX long 2",
            languageCode: "fr",
            sourceURL: URL(string: "https://huggingface.co/spaces/kyutai/hibiki-zero-samples/resolve/main/data/audio_ntrex_4L/fr/source/f9fcfb48c566cfad_11labs.mp3")!,
            referenceOutputURL: URL(string: "https://huggingface.co/spaces/kyutai/hibiki-zero-samples/resolve/main/data/audio_ntrex_4L/fr/hibiki-zero/f9fcfb48c566cfad_11labs.mp3")
        ),
    ]

    public static var frenchFixtures: [FileAudioFixture] {
        frenchShortForm + frenchLongForm
    }
}

public final class FileAudioInputSource: AudioInputSource, @unchecked Sendable {
    private let fileURL: URL
    private let sourceDescription: AudioInputDescription
    private let lock = NSLock()
    private var stopped = false

    public init(fileURL: URL, targetSampleRate: Int = 24_000, chunkSampleCount: Int = 1_920) {
        self.fileURL = fileURL
        self.sourceDescription = AudioInputDescription(
            sampleRate: targetSampleRate,
            chunkSampleCount: chunkSampleCount
        )
    }

    public func description() async -> AudioInputDescription {
        sourceDescription
    }

    public func chunks() async throws -> [PCMChunk] {
        guard !isStopped else { return [] }

        let samples = try decodeMonoSamples()
        guard !samples.isEmpty else {
            throw AudioInputError.malformedChunk("file decoded to no samples")
        }

        return stride(from: 0, to: samples.count, by: sourceDescription.chunkSampleCount).map { start in
            let end = min(start + sourceDescription.chunkSampleCount, samples.count)
            let frameIndex = start / sourceDescription.chunkSampleCount
            return PCMChunk(
                frameIndex: frameIndex,
                timestampMilliseconds: Double(start) / Double(sourceDescription.sampleRate) * 1000,
                sampleRate: sourceDescription.sampleRate,
                samples: Array(samples[start..<end])
            )
        }
    }

    public func stop() {
        lock.lock()
        defer { lock.unlock() }
        stopped = true
    }

    private var isStopped: Bool {
        lock.lock()
        defer { lock.unlock() }
        return stopped
    }

    private func decodeMonoSamples() throws -> [Float] {
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            throw AudioInputError.unavailable("audio file missing: \(fileURL.lastPathComponent)")
        }

        do {
            let file = try AVAudioFile(forReading: fileURL)
            guard file.length > 0 else {
                throw AudioInputError.malformedChunk("empty audio file: \(fileURL.lastPathComponent)")
            }

            guard let sourceBuffer = AVAudioPCMBuffer(
                pcmFormat: file.processingFormat,
                frameCapacity: AVAudioFrameCount(file.length)
            ) else {
                throw AudioInputError.unavailable("could not allocate audio decode buffer")
            }
            try file.read(into: sourceBuffer)

            guard let targetFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: Double(sourceDescription.sampleRate),
                channels: 1,
                interleaved: false
            ) else {
                throw AudioInputError.unsupportedSampleRate(sourceDescription.sampleRate)
            }

            let convertedBuffer = try convert(sourceBuffer, to: targetFormat)
            return try monoSamples(from: convertedBuffer)
        } catch let error as AudioInputError {
            throw error
        } catch {
            throw AudioInputError.unavailable("could not decode \(fileURL.lastPathComponent): \(error.localizedDescription)")
        }
    }

    private func convert(
        _ sourceBuffer: AVAudioPCMBuffer,
        to targetFormat: AVAudioFormat
    ) throws -> AVAudioPCMBuffer {
        guard let converter = AVAudioConverter(from: sourceBuffer.format, to: targetFormat) else {
            throw AudioInputError.unavailable("could not create audio converter")
        }

        let frameRatio = targetFormat.sampleRate / sourceBuffer.format.sampleRate
        let outputCapacity = max(
            1,
            AVAudioFrameCount(Double(sourceBuffer.frameLength) * frameRatio.rounded(.up)) + 1_024
        )
        guard let outputBuffer = AVAudioPCMBuffer(
            pcmFormat: targetFormat,
            frameCapacity: outputCapacity
        ) else {
            throw AudioInputError.unavailable("could not allocate converted audio buffer")
        }

        let inputState = AudioConverterInputState(sourceBuffer: sourceBuffer)
        var conversionError: NSError?
        let status = converter.convert(to: outputBuffer, error: &conversionError) { _, outStatus in
            if inputState.didProvideInput {
                outStatus.pointee = .noDataNow
                return nil
            }

            inputState.didProvideInput = true
            outStatus.pointee = .haveData
            return inputState.sourceBuffer
        }

        if let conversionError {
            throw AudioInputError.unavailable("audio conversion failed: \(conversionError.localizedDescription)")
        }

        guard status != .error else {
            throw AudioInputError.unavailable("audio conversion failed")
        }

        return outputBuffer
    }

    private func monoSamples(from buffer: AVAudioPCMBuffer) throws -> [Float] {
        guard let channelData = buffer.floatChannelData else {
            throw AudioInputError.unavailable("decoded audio is not float PCM")
        }

        let frameCount = Int(buffer.frameLength)
        guard frameCount > 0 else { return [] }

        let samples = UnsafeBufferPointer(start: channelData[0], count: frameCount)
        return Array(samples)
    }
}

private final class AudioConverterInputState: @unchecked Sendable {
    let sourceBuffer: AVAudioPCMBuffer
    var didProvideInput = false

    init(sourceBuffer: AVAudioPCMBuffer) {
        self.sourceBuffer = sourceBuffer
    }
}

public final class RemoteAudioFileInputSource: AudioInputSource, @unchecked Sendable {
    private let fixture: FileAudioFixture
    private let targetSampleRate: Int
    private let chunkSampleCount: Int
    private let lock = NSLock()
    private var stopped = false

    public init(
        fixture: FileAudioFixture,
        targetSampleRate: Int = 24_000,
        chunkSampleCount: Int = 1_920
    ) {
        self.fixture = fixture
        self.targetSampleRate = targetSampleRate
        self.chunkSampleCount = chunkSampleCount
    }

    public func description() async -> AudioInputDescription {
        AudioInputDescription(sampleRate: targetSampleRate, chunkSampleCount: chunkSampleCount)
    }

    public func chunks() async throws -> [PCMChunk] {
        guard !isStopped else { return [] }

        let localURL = try await cachedFileURL()
        let source = FileAudioInputSource(
            fileURL: localURL,
            targetSampleRate: targetSampleRate,
            chunkSampleCount: chunkSampleCount
        )
        return try await source.chunks()
    }

    public func stop() {
        lock.lock()
        defer { lock.unlock() }
        stopped = true
    }

    private var isStopped: Bool {
        lock.lock()
        defer { lock.unlock() }
        return stopped
    }

    private func cachedFileURL() async throws -> URL {
        let fileManager = FileManager.default
        let cacheDirectory = try fileManager.url(
            for: .cachesDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        .appendingPathComponent("S2STranslateAudioSamples", isDirectory: true)

        try fileManager.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)

        let cachedURL = cacheDirectory
            .appendingPathComponent(fixture.id)
            .appendingPathExtension(fixture.sourceURL.pathExtension.isEmpty ? "mp3" : fixture.sourceURL.pathExtension)

        if fileManager.fileExists(atPath: cachedURL.path) {
            return cachedURL
        }

        let (downloadedURL, _) = try await URLSession.shared.download(from: fixture.sourceURL)
        if fileManager.fileExists(atPath: cachedURL.path) {
            try fileManager.removeItem(at: cachedURL)
        }
        try fileManager.moveItem(at: downloadedURL, to: cachedURL)
        return cachedURL
    }
}

public final class ConfigurableAudioInputSource: AudioInputSource, @unchecked Sendable {
    private let lock = NSLock()
    private var currentSource: any AudioInputSource

    public init(source: any AudioInputSource) {
        self.currentSource = source
    }

    public func update(source: any AudioInputSource) {
        lock.lock()
        defer { lock.unlock() }
        currentSource = source
    }

    public func description() async -> AudioInputDescription {
        await source.description()
    }

    public func chunks() async throws -> [PCMChunk] {
        try await source.chunks()
    }

    public func stop() {
        source.stop()
    }

    private var source: any AudioInputSource {
        lock.lock()
        defer { lock.unlock() }
        return currentSource
    }
}

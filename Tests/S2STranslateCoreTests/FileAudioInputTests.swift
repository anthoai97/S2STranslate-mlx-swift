import AVFoundation
import Foundation
import Testing

@testable import S2STranslateCore

@Suite("File Audio Input")
struct FileAudioInputTests {
    @Test("file source decodes and chunks audio as mono 24 kHz PCM")
    func fileSourceDecodesAndChunksAudio() async throws {
        let url = try makeTemporaryWAV(sampleRate: 48_000, frameCount: 4_800)
        defer { try? FileManager.default.removeItem(at: url) }

        let source = FileAudioInputSource(fileURL: url, targetSampleRate: 24_000, chunkSampleCount: 1_920)
        let description = await source.description()
        let chunks = try await source.chunks()

        #expect(description.sampleRate == 24_000)
        #expect(description.chunkSampleCount == 1_920)
        #expect(chunks.count == 2)
        #expect(chunks[0].frameIndex == 0)
        #expect(chunks[0].sampleRate == 24_000)
        #expect(chunks[0].samples.count == 1_920)
        #expect(chunks[1].samples.count > 0)
        #expect(chunks[1].timestampMilliseconds == 80)
    }

    @Test("file-based translation backend streams through deterministic pipeline")
    @MainActor
    func fileBasedTranslationBackendStreamsThroughPipeline() async throws {
        let url = try makeTemporaryWAV(sampleRate: 24_000, frameCount: 3_840)
        defer { try? FileManager.default.removeItem(at: url) }

        let session = ExperimentSession(
            backend: HibikiTranslationExperimentBackend(
                artifactPreparer: ModelArtifactPreparer(
                    manifest: .hibikiQ4Default,
                    provider: DemoModelArtifactProvider()
                ),
                audioSource: FileAudioInputSource(fileURL: url, targetSampleRate: 24_000, chunkSampleCount: 1_920),
                mimiEncoder: DeterministicMimiStreamingEncoder(),
                inferenceSession: DeterministicHibikiInferenceSession(),
                mimiDecoder: DeterministicMimiStreamingDecoder(),
                playbackSink: BufferedPlaybackSink()
            )
        )

        await session.prepare()
        await session.start()

        #expect(session.state == .running)
        #expect(session.observations.audioInputStatus == "stopped")
        #expect(session.observations.audioChunkCount == 2)
        #expect(session.observations.mimiEncodedFrameCount == 2)
        #expect(session.observations.hibikiStepCount == 2)
        #expect(session.observations.decodedAudioChunkCount == 2)
        #expect(session.observations.playbackChunkCount == 2)
        #expect(session.observations.output == " hello")
    }

    @Test("missing audio file reaches unavailable error")
    func missingAudioFileFailsClearly() async throws {
        let source = FileAudioInputSource(
            fileURL: FileManager.default.temporaryDirectory.appendingPathComponent("missing-\(UUID().uuidString).wav")
        )

        do {
            _ = try await source.chunks()
            Issue.record("Expected missing file to fail")
        } catch let error as AudioInputError {
            #expect(error.userVisibleMessage.contains("audio file missing"))
        }
    }

    private func makeTemporaryWAV(sampleRate: Double, frameCount: Int) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("s2s-audio-\(UUID().uuidString)")
            .appendingPathExtension("caf")

        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw AudioInputError.unavailable("could not create test audio format")
        }

        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(frameCount)
        ) else {
            throw AudioInputError.unavailable("could not create test audio buffer")
        }
        buffer.frameLength = AVAudioFrameCount(frameCount)

        if let samples = buffer.floatChannelData?[0] {
            for frameIndex in 0..<frameCount {
                samples[frameIndex] = Float(frameIndex % 128) / 128
            }
        }

        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false,
        ]
        let file = try AVAudioFile(forWriting: url, settings: settings)
        try file.write(from: buffer)
        return url
    }
}

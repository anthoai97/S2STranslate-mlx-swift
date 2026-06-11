import Foundation
import Synchronization
import Testing

@testable import S2STranslateCore

@Suite("AV Audio Playback Sink")
struct AVAudioPlaybackSinkTests {
    @Test("AV playback sink starts schedules chunks and stops through engine seam")
    func avPlaybackSinkStartsSchedulesChunksAndStopsThroughEngineSeam() async throws {
        let engine = RecordingAudioPlaybackEngine()
        let sink = AVAudioPlaybackSink(engine: engine)

        try await sink.start(sampleRate: 24_000)
        try await sink.receive(
            DecodedAudioChunk(
                frameIndex: 0,
                timestampMilliseconds: 0,
                sampleRate: 24_000,
                samples: [0.1, 0.2, 0.3],
                sourceTokenFrameIndex: 0
            )
        )
        try await sink.receive(
            DecodedAudioChunk(
                frameIndex: 1,
                timestampMilliseconds: 80,
                sampleRate: 24_000,
                samples: [0.4, 0.5],
                sourceTokenFrameIndex: 1
            )
        )
        try await sink.finish()
        sink.stop()

        #expect(engine.startedSampleRates == [24_000])
        #expect(engine.scheduledSampleCounts == [3, 2])
        #expect(engine.finishCount == 1)
        #expect(engine.stopCount == 1)
        await #expect(throws: PlaybackSinkError.stopped) {
            try await sink.receive(
                DecodedAudioChunk(
                    frameIndex: 2,
                    timestampMilliseconds: 160,
                    sampleRate: 24_000,
                    samples: [0.6],
                    sourceTokenFrameIndex: 2
                )
            )
        }
    }

    @Test("AV playback sink converts engine failures to playback errors")
    func avPlaybackSinkConvertsEngineFailuresToPlaybackErrors() async throws {
        let engine = RecordingAudioPlaybackEngine()
        engine.startError = ExamplePlaybackEngineError.failed
        let sink = AVAudioPlaybackSink(engine: engine)

        await #expect(throws: PlaybackSinkError.unavailable("failed")) {
            try await sink.start(sampleRate: 24_000)
        }

        engine.startError = nil
        engine.scheduleError = ExamplePlaybackEngineError.failed
        try await sink.start(sampleRate: 24_000)

        await #expect(throws: PlaybackSinkError.unavailable("failed")) {
            try await sink.receive(
                DecodedAudioChunk(
                    frameIndex: 0,
                    timestampMilliseconds: 0,
                    sampleRate: 24_000,
                    samples: [0.1],
                    sourceTokenFrameIndex: 0
                )
            )
        }
    }

    @Test("deferred playback sink starts wrapped audio only on finish")
    func deferredPlaybackSinkStartsWrappedAudioOnlyOnFinish() async throws {
        let engine = RecordingAudioPlaybackEngine()
        let sink = DeferredAudioPlaybackSink(wrapped: AVAudioPlaybackSink(engine: engine))

        try await sink.start(sampleRate: 24_000)
        try await sink.receive(
            DecodedAudioChunk(
                frameIndex: 0,
                timestampMilliseconds: 0,
                sampleRate: 24_000,
                samples: [0.1, 0.2, 0.3],
                sourceTokenFrameIndex: 0
            )
        )
        try await sink.receive(
            DecodedAudioChunk(
                frameIndex: 1,
                timestampMilliseconds: 80,
                sampleRate: 24_000,
                samples: [0.4, 0.5],
                sourceTokenFrameIndex: 1
            )
        )

        #expect(engine.startedSampleRates.isEmpty)
        #expect(engine.scheduledSampleCounts.isEmpty)

        try await sink.finish()

        #expect(engine.startedSampleRates == [24_000])
        #expect(engine.scheduledSampleCounts == [3, 2])
        #expect(engine.finishCount == 1)
    }
}

private final class RecordingAudioPlaybackEngine: AudioPlaybackEngine, @unchecked Sendable {
    private let state = Mutex(RecordingAudioPlaybackEngineState())

    var startedSampleRates: [Int] {
        state.withLock { $0.startedSampleRates }
    }

    var scheduledSampleCounts: [Int] {
        state.withLock { $0.scheduledSampleCounts }
    }

    var stopCount: Int {
        state.withLock { $0.stopCount }
    }

    var finishCount: Int {
        state.withLock { $0.finishCount }
    }

    var startError: ExamplePlaybackEngineError? {
        get { state.withLock { $0.startError } }
        set { state.withLock { $0.startError = newValue } }
    }

    var scheduleError: ExamplePlaybackEngineError? {
        get { state.withLock { $0.scheduleError } }
        set { state.withLock { $0.scheduleError = newValue } }
    }

    func start(sampleRate: Int) throws {
        try state.withLock { state in
            if let startError = state.startError {
                throw startError
            }
            state.startedSampleRates.append(sampleRate)
        }
    }

    func schedule(samples: [Float], sampleRate: Int) throws {
        try state.withLock { state in
            if let scheduleError = state.scheduleError {
                throw scheduleError
            }
            state.scheduledSampleCounts.append(samples.count)
        }
    }

    func finish() async throws {
        state.withLock { $0.finishCount += 1 }
    }

    func stop() {
        state.withLock { $0.stopCount += 1 }
    }

    func reset() {
        state.withLock { $0.resetCount += 1 }
    }
}

private struct RecordingAudioPlaybackEngineState {
    var startedSampleRates: [Int] = []
    var scheduledSampleCounts: [Int] = []
    var finishCount = 0
    var stopCount = 0
    var resetCount = 0
    var startError: ExamplePlaybackEngineError?
    var scheduleError: ExamplePlaybackEngineError?
}

private enum ExamplePlaybackEngineError: Error, CustomStringConvertible {
    case failed

    var description: String {
        "failed"
    }
}

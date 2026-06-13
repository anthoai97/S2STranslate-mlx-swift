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

    @Test("AV playback sink exposes playback diagnostics from engine")
    func avPlaybackSinkExposesPlaybackDiagnosticsFromEngine() async throws {
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

        let snapshot = try #require(sink.diagnosticsSnapshot())
        #expect(snapshot.playbackStarted)
        #expect(snapshot.sampleRate == 24_000)
        #expect(snapshot.scheduledBufferCount == 1)
        #expect(snapshot.scheduledSampleCount == 3)
        #expect(snapshot.pendingSampleCount == 3)
        #expect(snapshot.pendingDurationMilliseconds == 0.125)
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

    @Test("buffered streaming playback starts after prebuffer threshold")
    func bufferedStreamingPlaybackStartsAfterPrebufferThreshold() async throws {
        let engine = RecordingAudioPlaybackEngine()
        let sink = BufferedStreamingAudioPlaybackSink(
            wrapped: AVAudioPlaybackSink(engine: engine),
            prebufferDurationSeconds: 0.2
        )

        try await sink.start(sampleRate: 10)
        try await sink.receive(
            DecodedAudioChunk(
                frameIndex: 0,
                timestampMilliseconds: 0,
                sampleRate: 10,
                samples: [0.1],
                sourceTokenFrameIndex: 0
            )
        )
        #expect(engine.startedSampleRates.isEmpty)
        #expect(engine.scheduledSampleCounts.isEmpty)
        let prebufferSnapshot = try #require(sink.diagnosticsSnapshot())
        #expect(!prebufferSnapshot.playbackStarted)
        #expect(prebufferSnapshot.pendingSampleCount == 1)
        #expect(prebufferSnapshot.pendingDurationMilliseconds == 100)

        try await sink.receive(
            DecodedAudioChunk(
                frameIndex: 1,
                timestampMilliseconds: 100,
                sampleRate: 10,
                samples: [0.2],
                sourceTokenFrameIndex: 1
            )
        )
        #expect(engine.startedSampleRates == [10])
        #expect(engine.scheduledSampleCounts == [1, 1])
        let playbackSnapshot = try #require(sink.diagnosticsSnapshot())
        #expect(playbackSnapshot.playbackStarted)
        #expect(playbackSnapshot.scheduledSampleCount == 2)
        #expect(playbackSnapshot.pendingSampleCount == 2)

        try await sink.receive(
            DecodedAudioChunk(
                frameIndex: 2,
                timestampMilliseconds: 200,
                sampleRate: 10,
                samples: [0.3],
                sourceTokenFrameIndex: 2
            )
        )
        try await sink.finish()

        #expect(engine.scheduledSampleCounts == [1, 1, 1])
        #expect(engine.finishCount == 1)
    }
}

private final class RecordingAudioPlaybackEngine: AudioPlaybackEngine, PlaybackDiagnosticsReporting, @unchecked Sendable {
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
            state.sampleRate = sampleRate
            state.scheduledBufferCount = 0
            state.completedBufferCount = 0
            state.scheduledSampleCount = 0
            state.completedSampleCount = 0
            state.pendingSampleCount = 0
        }
    }

    func schedule(samples: [Float], sampleRate: Int) throws {
        try state.withLock { state in
            if let scheduleError = state.scheduleError {
                throw scheduleError
            }
            state.scheduledSampleCounts.append(samples.count)
            state.scheduledBufferCount += 1
            state.scheduledSampleCount += samples.count
            state.pendingSampleCount += samples.count
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

    func diagnosticsSnapshot() -> PlaybackDiagnosticsSnapshot? {
        state.withLock { state in
            guard let sampleRate = state.sampleRate else { return nil }
            return PlaybackDiagnosticsSnapshot(
                sampleRate: sampleRate,
                playbackStarted: true,
                scheduledBufferCount: state.scheduledBufferCount,
                completedBufferCount: state.completedBufferCount,
                pendingBufferCount: state.scheduledBufferCount - state.completedBufferCount,
                scheduledSampleCount: state.scheduledSampleCount,
                completedSampleCount: state.completedSampleCount,
                pendingSampleCount: state.pendingSampleCount,
                pendingDurationMilliseconds: Double(state.pendingSampleCount) / Double(sampleRate) * 1000,
                underrunCount: 0,
                elapsedMilliseconds: 0
            )
        }
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
    var sampleRate: Int?
    var scheduledBufferCount = 0
    var completedBufferCount = 0
    var scheduledSampleCount = 0
    var completedSampleCount = 0
    var pendingSampleCount = 0
}

private enum ExamplePlaybackEngineError: Error, CustomStringConvertible {
    case failed

    var description: String {
        "failed"
    }
}

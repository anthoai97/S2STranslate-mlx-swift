import Synchronization
import Testing

@testable import S2STranslateCore

@Suite("Playback-Only Diagnostic")
struct PlaybackOnlyDiagnosticTests {
    @Test("diagnostic plays decoded chunks through playback sink and reports healthy accounting")
    func diagnosticPlaysDecodedChunksThroughPlaybackSinkAndReportsHealthyAccounting() async throws {
        let sink = DiagnosticRecordingPlaybackSink()
        let chunks = [
            diagnosticChunk(frameIndex: 0, sampleCount: 2_400),
            diagnosticChunk(frameIndex: 1, sampleCount: 2_400),
        ]

        let result = try await PlaybackOnlyDiagnostic().run(
            chunks: chunks,
            playbackSink: sink
        )

        #expect(sink.startedSampleRates == [24_000])
        #expect(sink.receivedFrameIndexes == [0, 1])
        #expect(sink.finishCount == 1)
        #expect(result.classification == .healthy)
        #expect(result.scheduledDurationMilliseconds == 200)
        #expect(result.completedDurationMilliseconds == 200)
        #expect(result.pendingDurationMilliseconds == 0)
        #expect(result.scheduleGapMilliseconds == 100)
        #expect(result.underrunCount == 0)
    }

    @Test("diagnostic plays synthetic steady PCM through playback sink")
    func diagnosticPlaysSyntheticSteadyPCMThroughPlaybackSink() async throws {
        let sink = DiagnosticRecordingPlaybackSink()

        let result = try await PlaybackOnlyDiagnostic().runSyntheticPCM(
            durationMilliseconds: 200,
            sampleRate: 24_000,
            chunkDurationMilliseconds: 100,
            playbackSink: sink
        )

        #expect(sink.startedSampleRates == [24_000])
        #expect(sink.receivedFrameIndexes == [0, 1])
        #expect(result.classification == .healthy)
        #expect(result.scheduledDurationMilliseconds == 200)
        #expect(result.completedDurationMilliseconds == 200)
        #expect(result.pendingDurationMilliseconds == 0)
    }

    @Test("diagnostic classifies audio route failures as environment interruptions")
    func diagnosticClassifiesAudioRouteFailuresAsEnvironmentInterruptions() async throws {
        let sink = FailingDiagnosticPlaybackSink(error: .unavailable("audio route unavailable"))

        let result = try await PlaybackOnlyDiagnostic().run(
            chunks: [diagnosticChunk(frameIndex: 0, sampleCount: 2_400)],
            playbackSink: sink
        )

        #expect(result.classification == .interruptedByEnvironment)
        #expect(result.message == "Playback sink unavailable: audio route unavailable")
    }
}

private func diagnosticChunk(frameIndex: Int, sampleCount: Int) -> DecodedAudioChunk {
    DecodedAudioChunk(
        frameIndex: frameIndex,
        timestampMilliseconds: Double(frameIndex) * 100,
        sampleRate: 24_000,
        samples: Array(repeating: 0.1, count: sampleCount),
        sourceTokenFrameIndex: frameIndex
    )
}

private final class DiagnosticRecordingPlaybackSink: PlaybackSink, PlaybackDiagnosticsReporting, @unchecked Sendable {
    private let state = Mutex(DiagnosticRecordingPlaybackSinkState())

    var startedSampleRates: [Int] {
        state.withLock { $0.startedSampleRates }
    }

    var receivedFrameIndexes: [Int] {
        state.withLock { $0.receivedFrameIndexes }
    }

    var finishCount: Int {
        state.withLock { $0.finishCount }
    }

    func start(sampleRate: Int) async throws {
        state.withLock { state in
            state.sampleRate = sampleRate
            state.startedSampleRates.append(sampleRate)
        }
    }

    func receive(_ chunk: DecodedAudioChunk) async throws {
        state.withLock { state in
            state.receivedFrameIndexes.append(chunk.frameIndex)
            state.scheduledSampleCount += chunk.samples.count
            state.pendingSampleCount += chunk.samples.count
            state.scheduledBufferCount += 1
            state.pendingBufferCount += 1
            state.lastScheduleGapMilliseconds = chunk.frameIndex == 0 ? nil : chunk.durationMilliseconds
        }
    }

    func finish() async throws {
        state.withLock { state in
            state.finishCount += 1
            state.completedSampleCount = state.scheduledSampleCount
            state.pendingSampleCount = 0
            state.completedBufferCount = state.scheduledBufferCount
            state.pendingBufferCount = 0
        }
    }

    func stop() {}

    func reset() {}

    func diagnosticsSnapshot() -> PlaybackDiagnosticsSnapshot? {
        state.withLock { state in
            guard let sampleRate = state.sampleRate else { return nil }
            return PlaybackDiagnosticsSnapshot(
                sampleRate: sampleRate,
                playbackStarted: true,
                scheduledBufferCount: state.scheduledBufferCount,
                completedBufferCount: state.completedBufferCount,
                pendingBufferCount: state.pendingBufferCount,
                scheduledSampleCount: state.scheduledSampleCount,
                completedSampleCount: state.completedSampleCount,
                pendingSampleCount: state.pendingSampleCount,
                pendingDurationMilliseconds: Double(state.pendingSampleCount) / Double(sampleRate) * 1000,
                lastScheduleGapMilliseconds: state.lastScheduleGapMilliseconds,
                underrunCount: state.underrunCount,
                elapsedMilliseconds: 0
            )
        }
    }
}

private struct DiagnosticRecordingPlaybackSinkState {
    var sampleRate: Int?
    var startedSampleRates: [Int] = []
    var receivedFrameIndexes: [Int] = []
    var finishCount = 0
    var scheduledBufferCount = 0
    var completedBufferCount = 0
    var pendingBufferCount = 0
    var scheduledSampleCount = 0
    var completedSampleCount = 0
    var pendingSampleCount = 0
    var lastScheduleGapMilliseconds: Double?
    var underrunCount = 0
}

private struct FailingDiagnosticPlaybackSink: PlaybackSink {
    var error: PlaybackSinkError

    func start(sampleRate: Int) async throws {
        throw error
    }

    func receive(_ chunk: DecodedAudioChunk) async throws {}

    func finish() async throws {}

    func stop() {}

    func reset() {}
}

import Testing
import Synchronization

@testable import S2STranslateCore

@Suite("Realtime Output Policy")
struct RealtimeOutputPolicyTests {
    @Test("policy selects output strategy from generated realtime factor")
    func policySelectsOutputStrategyFromGeneratedRealtimeFactor() {
        let policy = RealtimeOutputPolicy()

        let subRealtime = policy.selectStrategy(generatedRealtimeFactor: 0.216)
        let bareRealtime = policy.selectStrategy(generatedRealtimeFactor: 1.0)
        let almostPractical = policy.selectStrategy(generatedRealtimeFactor: 1.24)
        let practicalRealtime = policy.selectStrategy(generatedRealtimeFactor: 1.25)

        #expect(subRealtime.capability == .subRealtime)
        #expect(subRealtime.strategy == .deferredPlayback)
        #expect(bareRealtime.capability == .bareRealtime)
        #expect(bareRealtime.strategy == .diagnosticLivePlaybackAttempt)
        #expect(almostPractical.capability == .bareRealtime)
        #expect(almostPractical.strategy == .diagnosticLivePlaybackAttempt)
        #expect(practicalRealtime.capability == .practicalRealtime)
        #expect(practicalRealtime.strategy == .defaultLivePlayback)
    }

    @Test("policy interprets generated realtime factor as generated duration over processing time")
    func policyInterpretsGeneratedRealtimeFactorFromDurations() {
        let policy = RealtimeOutputPolicy()

        let decision = policy.selectStrategy(
            generatedAudioDurationSeconds: 2.16,
            processingWallTimeSeconds: 10
        )

        #expect(abs((decision.generatedRealtimeFactor ?? 0) - 0.216) < 0.000001)
        #expect(decision.capability == .subRealtime)
        #expect(decision.strategy == .deferredPlayback)
    }

    @Test("policy routes sub-realtime playback through deferred output while keeping practical realtime live")
    func policyRoutesSubRealtimePlaybackThroughDeferredOutput() async throws {
        let policy = RealtimeOutputPolicy()
        let subRealtimeSink = RecordingPlaybackSink()
        let subRealtimeRoute = policy.routePlayback(
            generatedRealtimeFactor: 0.216,
            livePlaybackSink: subRealtimeSink
        )

        try await subRealtimeRoute.playbackSink.start(sampleRate: 24_000)
        try await subRealtimeRoute.playbackSink.receive(decodedChunk(frameIndex: 0))

        #expect(subRealtimeRoute.decision.strategy == .deferredPlayback)
        #expect(subRealtimeSink.startedSampleRates.isEmpty)
        #expect(subRealtimeSink.receivedFrameIndexes.isEmpty)

        try await subRealtimeRoute.playbackSink.finish()

        #expect(subRealtimeSink.startedSampleRates == [24_000])
        #expect(subRealtimeSink.receivedFrameIndexes == [0])
        #expect(subRealtimeSink.finishCount == 1)

        let practicalSink = RecordingPlaybackSink()
        let practicalRoute = policy.routePlayback(
            generatedRealtimeFactor: 1.25,
            livePlaybackSink: practicalSink
        )

        try await practicalRoute.playbackSink.start(sampleRate: 24_000)
        try await practicalRoute.playbackSink.receive(decodedChunk(frameIndex: 1))

        #expect(practicalRoute.decision.strategy == .defaultLivePlayback)
        #expect(practicalSink.startedSampleRates == [24_000])
        #expect(practicalSink.receivedFrameIndexes == [1])
    }

    @Test("policy can force diagnostic live playback for sub-realtime debugging")
    func policyCanForceDiagnosticLivePlaybackForSubRealtimeDebugging() async throws {
        let policy = RealtimeOutputPolicy()
        let sink = RecordingPlaybackSink()

        let route = policy.routePlayback(
            generatedRealtimeFactor: 0.23,
            livePlaybackSink: sink,
            forceDiagnosticLivePlayback: true
        )

        try await route.playbackSink.start(sampleRate: 24_000)
        try await route.playbackSink.receive(decodedChunk(frameIndex: 0))

        #expect(route.decision.capability == .subRealtime)
        #expect(route.decision.strategy == .diagnosticLivePlaybackAttempt)
        #expect(route.decision.reason == "live playback was explicitly enabled for diagnostics")
        #expect(sink.startedSampleRates == [24_000])
        #expect(sink.receivedFrameIndexes == [0])
    }
}

private func decodedChunk(frameIndex: Int) -> DecodedAudioChunk {
    DecodedAudioChunk(
        frameIndex: frameIndex,
        timestampMilliseconds: Double(frameIndex) * 80,
        sampleRate: 24_000,
        samples: [0.1, 0.2, 0.3],
        sourceTokenFrameIndex: frameIndex
    )
}

private final class RecordingPlaybackSink: PlaybackSink, @unchecked Sendable {
    private let state = Mutex(RecordingPlaybackSinkState())

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
        state.withLock { $0.startedSampleRates.append(sampleRate) }
    }

    func receive(_ chunk: DecodedAudioChunk) async throws {
        state.withLock { $0.receivedFrameIndexes.append(chunk.frameIndex) }
    }

    func finish() async throws {
        state.withLock { $0.finishCount += 1 }
    }

    func stop() {}

    func reset() {}
}

private struct RecordingPlaybackSinkState {
    var startedSampleRates: [Int] = []
    var receivedFrameIndexes: [Int] = []
    var finishCount = 0
}

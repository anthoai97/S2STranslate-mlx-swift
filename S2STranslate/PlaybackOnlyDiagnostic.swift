import Foundation

public enum PlaybackHealthClassification: String, Equatable, Sendable {
    case healthy
    case interruptedByEnvironment
    case unhealthy
}

public struct PlaybackOnlyDiagnosticResult: Equatable, Sendable {
    public var classification: PlaybackHealthClassification
    public var scheduledDurationMilliseconds: Double
    public var completedDurationMilliseconds: Double
    public var pendingDurationMilliseconds: Double
    public var scheduleGapMilliseconds: Double?
    public var underrunCount: Int
    public var message: String

    nonisolated public init(
        classification: PlaybackHealthClassification,
        scheduledDurationMilliseconds: Double,
        completedDurationMilliseconds: Double,
        pendingDurationMilliseconds: Double,
        scheduleGapMilliseconds: Double? = nil,
        underrunCount: Int,
        message: String
    ) {
        self.classification = classification
        self.scheduledDurationMilliseconds = scheduledDurationMilliseconds
        self.completedDurationMilliseconds = completedDurationMilliseconds
        self.pendingDurationMilliseconds = pendingDurationMilliseconds
        self.scheduleGapMilliseconds = scheduleGapMilliseconds
        self.underrunCount = underrunCount
        self.message = message
    }
}

public struct PlaybackOnlyDiagnostic: Sendable {
    nonisolated public init() {}

    public func run(
        chunks: [DecodedAudioChunk],
        playbackSink: any PlaybackSink
    ) async throws -> PlaybackOnlyDiagnosticResult {
        guard let sampleRate = chunks.first?.sampleRate else {
            throw PlaybackSinkError.unavailable("no decoded chunks to diagnose")
        }

        playbackSink.reset()
        do {
            try await playbackSink.start(sampleRate: sampleRate)
            for chunk in chunks {
                try await playbackSink.receive(chunk)
            }
            try await playbackSink.finish()
        } catch let error as PlaybackSinkError {
            return diagnosticResult(for: error)
        }

        guard let snapshot = (playbackSink as? any PlaybackDiagnosticsReporting)?.diagnosticsSnapshot() else {
            return PlaybackOnlyDiagnosticResult(
                classification: .unhealthy,
                scheduledDurationMilliseconds: 0,
                completedDurationMilliseconds: 0,
                pendingDurationMilliseconds: 0,
                underrunCount: 0,
                message: "playback sink did not provide diagnostics"
            )
        }

        return PlaybackOnlyDiagnosticResult(
            classification: classify(snapshot),
            scheduledDurationMilliseconds: snapshot.scheduledDurationMilliseconds,
            completedDurationMilliseconds: snapshot.completedDurationMilliseconds,
            pendingDurationMilliseconds: snapshot.pendingDurationMilliseconds,
            scheduleGapMilliseconds: snapshot.lastScheduleGapMilliseconds,
            underrunCount: snapshot.underrunCount,
            message: message(for: snapshot)
        )
    }

    public func runSyntheticPCM(
        durationMilliseconds: Double,
        sampleRate: Int = 24_000,
        chunkDurationMilliseconds: Double = 100,
        playbackSink: any PlaybackSink
    ) async throws -> PlaybackOnlyDiagnosticResult {
        guard durationMilliseconds > 0, sampleRate > 0, chunkDurationMilliseconds > 0 else {
            throw PlaybackSinkError.unavailable("synthetic PCM diagnostic requires positive duration and sample rate")
        }

        let totalSampleCount = Int((durationMilliseconds / 1000 * Double(sampleRate)).rounded())
        let chunkSampleCount = max(1, Int((chunkDurationMilliseconds / 1000 * Double(sampleRate)).rounded()))
        var chunks: [DecodedAudioChunk] = []
        var sampleOffset = 0
        var frameIndex = 0
        while sampleOffset < totalSampleCount {
            let sampleCount = min(chunkSampleCount, totalSampleCount - sampleOffset)
            chunks.append(
                DecodedAudioChunk(
                    frameIndex: frameIndex,
                    timestampMilliseconds: Double(sampleOffset) / Double(sampleRate) * 1000,
                    sampleRate: sampleRate,
                    samples: Array(repeating: 0.1, count: sampleCount),
                    sourceTokenFrameIndex: frameIndex
                )
            )
            sampleOffset += sampleCount
            frameIndex += 1
        }

        return try await run(chunks: chunks, playbackSink: playbackSink)
    }

    private func classify(_ snapshot: PlaybackDiagnosticsSnapshot) -> PlaybackHealthClassification {
        if snapshot.underrunCount == 0,
           snapshot.pendingSampleCount == 0,
           snapshot.completedSampleCount >= snapshot.scheduledSampleCount {
            return .healthy
        }
        return .unhealthy
    }

    private func message(for snapshot: PlaybackDiagnosticsSnapshot) -> String {
        switch classify(snapshot) {
        case .healthy:
            "playback completed without pending audio or underruns"
        case .interruptedByEnvironment:
            "playback was interrupted by the audio environment"
        case .unhealthy:
            "playback reported pending audio, underruns, or incomplete device-played output"
        }
    }

    private func diagnosticResult(for error: PlaybackSinkError) -> PlaybackOnlyDiagnosticResult {
        let message = error.userVisibleMessage
        let lowercased = message.lowercased()
        let classification: PlaybackHealthClassification =
            lowercased.contains("route") || lowercased.contains("interrupt")
            ? .interruptedByEnvironment
            : .unhealthy
        return PlaybackOnlyDiagnosticResult(
            classification: classification,
            scheduledDurationMilliseconds: 0,
            completedDurationMilliseconds: 0,
            pendingDurationMilliseconds: 0,
            underrunCount: 0,
            message: message
        )
    }
}

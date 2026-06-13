import Foundation

public enum RealtimeCapabilityClass: String, Equatable, Sendable {
    case subRealtime
    case bareRealtime
    case practicalRealtime

    public var displayName: String {
        switch self {
        case .subRealtime:
            "sub-realtime"
        case .bareRealtime:
            "bare-realtime"
        case .practicalRealtime:
            "practical-realtime"
        }
    }
}

public enum RealtimeOutputStrategy: String, Equatable, Sendable {
    case diagnosticLivePlaybackAttempt
    case defaultLivePlayback
    case deferredPlayback
    case outputOnly

    public var displayName: String {
        switch self {
        case .diagnosticLivePlaybackAttempt:
            "diagnostic live playback attempt"
        case .defaultLivePlayback:
            "default live playback"
        case .deferredPlayback:
            "deferred playback"
        case .outputOnly:
            "output-only"
        }
    }
}

public struct RealtimeOutputStrategyDecision: Equatable, Sendable {
    public var strategy: RealtimeOutputStrategy
    public var capability: RealtimeCapabilityClass?
    public var generatedRealtimeFactor: Double?
    public var reason: String

    nonisolated public init(
        strategy: RealtimeOutputStrategy,
        capability: RealtimeCapabilityClass? = nil,
        generatedRealtimeFactor: Double? = nil,
        reason: String
    ) {
        self.strategy = strategy
        self.capability = capability
        self.generatedRealtimeFactor = generatedRealtimeFactor
        self.reason = reason
    }

    public var summary: String {
        let capabilityText = capability.map { ", \($0.displayName)" } ?? ""
        let factorText = generatedRealtimeFactor.map { ", \(String(format: "%.3f", $0))x" } ?? ""
        return "\(strategy.displayName)\(capabilityText)\(factorText): \(reason)"
    }
}

public struct RealtimePlaybackRoute: Sendable {
    public var decision: RealtimeOutputStrategyDecision
    public var playbackSink: any PlaybackSink

    nonisolated public init(
        decision: RealtimeOutputStrategyDecision,
        playbackSink: any PlaybackSink
    ) {
        self.decision = decision
        self.playbackSink = playbackSink
    }
}

public struct RealtimeOutputPolicy: Equatable, Sendable {
    public var liveRealtimeThreshold: Double
    public var practicalRealtimeThreshold: Double

    nonisolated public init(
        liveRealtimeThreshold: Double = 1.0,
        practicalRealtimeThreshold: Double = 1.25
    ) {
        self.liveRealtimeThreshold = liveRealtimeThreshold
        self.practicalRealtimeThreshold = practicalRealtimeThreshold
    }

    public func classify(generatedRealtimeFactor: Double) -> RealtimeCapabilityClass {
        if generatedRealtimeFactor < liveRealtimeThreshold {
            return .subRealtime
        }
        if generatedRealtimeFactor < practicalRealtimeThreshold {
            return .bareRealtime
        }
        return .practicalRealtime
    }

    public func selectStrategy(
        generatedRealtimeFactor: Double,
        forceDiagnosticLivePlayback: Bool = false
    ) -> RealtimeOutputStrategyDecision {
        let capability = classify(generatedRealtimeFactor: generatedRealtimeFactor)
        if forceDiagnosticLivePlayback {
            return RealtimeOutputStrategyDecision(
                strategy: .diagnosticLivePlaybackAttempt,
                capability: capability,
                generatedRealtimeFactor: generatedRealtimeFactor,
                reason: "live playback was explicitly enabled for diagnostics"
            )
        }
        switch capability {
        case .subRealtime:
            return RealtimeOutputStrategyDecision(
                strategy: .deferredPlayback,
                capability: capability,
                generatedRealtimeFactor: generatedRealtimeFactor,
                reason: "generated audio is below the live playback threshold"
            )
        case .bareRealtime:
            return RealtimeOutputStrategyDecision(
                strategy: .diagnosticLivePlaybackAttempt,
                capability: capability,
                generatedRealtimeFactor: generatedRealtimeFactor,
                reason: "generated audio meets the minimum live threshold without practical headroom"
            )
        case .practicalRealtime:
            return RealtimeOutputStrategyDecision(
                strategy: .defaultLivePlayback,
                capability: capability,
                generatedRealtimeFactor: generatedRealtimeFactor,
                reason: "generated audio has practical live playback headroom"
            )
        }
    }

    public func selectStrategy(
        generatedAudioDurationSeconds: Double,
        processingWallTimeSeconds: Double,
        forceDiagnosticLivePlayback: Bool = false
    ) -> RealtimeOutputStrategyDecision {
        selectStrategy(
            generatedRealtimeFactor: processingWallTimeSeconds > 0
                ? generatedAudioDurationSeconds / processingWallTimeSeconds
                : 0,
            forceDiagnosticLivePlayback: forceDiagnosticLivePlayback
        )
    }

    public func routePlayback(
        generatedRealtimeFactor: Double,
        livePlaybackSink: any PlaybackSink,
        forceDiagnosticLivePlayback: Bool = false
    ) -> RealtimePlaybackRoute {
        let decision = selectStrategy(
            generatedRealtimeFactor: generatedRealtimeFactor,
            forceDiagnosticLivePlayback: forceDiagnosticLivePlayback
        )
        let playbackSink: any PlaybackSink
        switch decision.strategy {
        case .deferredPlayback:
            playbackSink = DeferredAudioPlaybackSink(wrapped: livePlaybackSink)
        case .diagnosticLivePlaybackAttempt, .defaultLivePlayback, .outputOnly:
            playbackSink = livePlaybackSink
        }
        return RealtimePlaybackRoute(decision: decision, playbackSink: playbackSink)
    }
}

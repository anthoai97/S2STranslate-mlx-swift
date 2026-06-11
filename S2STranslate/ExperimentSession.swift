import Combine
import Foundation
import OSLog

@MainActor
public final class ExperimentSession: ObservableObject {
    @Published public private(set) var state: ExperimentSessionState
    @Published public private(set) var observations: ExperimentObservations

    private let backend: ExperimentBackend
    private let logger = Logger(subsystem: "andyq.S2STranslate", category: "ExperimentSession")

    public init(
        backend: ExperimentBackend,
        state: ExperimentSessionState = .unloaded,
        observations: ExperimentObservations = ExperimentObservations()
    ) {
        self.backend = backend
        self.state = state
        self.observations = observations
    }

    public func prepare() async {
        logger.info("prepare begin state=\(String(describing: self.state), privacy: .public)")
        state = .preparing
        await backend.prepareEvents { [weak self] event in
            self?.apply(event)
        }
        logger.info("prepare end state=\(String(describing: self.state), privacy: .public) events=\(self.observations.eventCount, privacy: .public)")
    }

    public func start() async {
        guard state == .ready else {
            logger.warning("start ignored state=\(String(describing: self.state), privacy: .public)")
            return
        }

        logger.info("start begin")
        state = .running
        for event in await backend.runEvents() {
            apply(event)
        }
        logger.info("start end state=\(String(describing: self.state), privacy: .public) events=\(self.observations.eventCount, privacy: .public)")
    }

    public func stop() {
        guard state == .running else {
            logger.warning("stop ignored state=\(String(describing: self.state), privacy: .public)")
            return
        }

        logger.info("stop")
        backend.stop()
        apply(.stopped)
        state = .stopped
    }

    public func newSession() {
        guard state.isTerminal else {
            logger.warning("new session ignored state=\(String(describing: self.state), privacy: .public)")
            return
        }

        logger.info("new session from state=\(String(describing: self.state), privacy: .public)")
        state = .unloaded
        observations = ExperimentObservations()
    }

    public func triggerFailureDemo() {
        logger.info("trigger failure demo")
        apply(.failure("Fake backend failure"))
    }

    private func apply(_ event: ExperimentEvent) {
        let stateBefore = state
        observations.record(event)

        switch event {
        case let .preparationProgress(progress):
            observations.progress = progress
        case let .artifactPreparationProgress(progress):
            observations.recordArtifactPreparation(progress)
        case .observation:
            break
        case let .audioInput(event):
            observations.recordAudioInput(event)
        case let .mimiEncode(event):
            observations.recordMimiEncode(event)
        case let .mimiDecode(event):
            observations.recordMimiDecode(event)
        case let .playback(event):
            observations.recordPlayback(event)
        case let .hibikiInference(event):
            observations.recordHibikiInference(event)
        case .ready:
            state = .ready
        case .stopped:
            break
        case let .failure(message):
            state = .failed(message)
        }
        if event.shouldLog {
            logger.info("event=\(event.name, privacy: .public) state=\(String(describing: stateBefore), privacy: .public)->\(String(describing: self.state), privacy: .public) count=\(self.observations.eventCount, privacy: .public)")
        }
    }
}

public enum ExperimentSessionState: Equatable {
    case unloaded
    case preparing
    case ready
    case running
    case stopped
    case failed(String)

    var isTerminal: Bool {
        switch self {
        case .stopped, .failed:
            true
        case .unloaded, .preparing, .ready, .running:
            false
        }
    }
}

public struct ExperimentObservations: Equatable {
    public var progress: Double
    public var artifactPreparationSummary: String
    public var artifactFileName: String
    public var artifactCompletedFileCount: Int
    public var artifactTotalFileCount: Int
    public var artifactFileProgress: Double?
    public var eventCount: Int
    public var lastEventName: String
    public var output: String
    public var audioInputStatus: String
    public var audioChunkCount: Int
    public var audioSampleRate: Int
    public var audioDurationMilliseconds: Double
    public var lastAudioFrameIndex: Int?
    public var mimiEncodeStatus: String
    public var mimiEncodedFrameCount: Int
    public var mimiCodebookCount: Int
    public var mimiTokenCount: Int
    public var mimiFrameDurationMilliseconds: Double
    public var lastMimiFrameIndex: Int?
    public var mimiDecodeStatus: String
    public var decodedAudioChunkCount: Int
    public var decodedAudioSampleRate: Int
    public var decodedAudioDurationMilliseconds: Double
    public var lastDecodedAudioFrameIndex: Int?
    public var playbackStatus: String
    public var playbackChunkCount: Int
    public var playbackDurationMilliseconds: Double
    public var lastPlaybackFrameIndex: Int?
    public var hibikiInferenceStatus: String
    public var hibikiStepCount: Int
    public var hibikiTextTokenCount: Int
    public var hibikiVisibleTextCount: Int
    public var hibikiGeneratedAudioFrameCount: Int
    public var hibikiSamplingSummary: String

    nonisolated public init(
        progress: Double = 0,
        artifactPreparationSummary: String = "n/a",
        artifactFileName: String = "n/a",
        artifactCompletedFileCount: Int = 0,
        artifactTotalFileCount: Int = 0,
        artifactFileProgress: Double? = nil,
        eventCount: Int = 0,
        lastEventName: String = "none",
        output: String = "",
        audioInputStatus: String = "idle",
        audioChunkCount: Int = 0,
        audioSampleRate: Int = 0,
        audioDurationMilliseconds: Double = 0,
        lastAudioFrameIndex: Int? = nil,
        mimiEncodeStatus: String = "idle",
        mimiEncodedFrameCount: Int = 0,
        mimiCodebookCount: Int = 0,
        mimiTokenCount: Int = 0,
        mimiFrameDurationMilliseconds: Double = 0,
        lastMimiFrameIndex: Int? = nil,
        mimiDecodeStatus: String = "idle",
        decodedAudioChunkCount: Int = 0,
        decodedAudioSampleRate: Int = 0,
        decodedAudioDurationMilliseconds: Double = 0,
        lastDecodedAudioFrameIndex: Int? = nil,
        playbackStatus: String = "idle",
        playbackChunkCount: Int = 0,
        playbackDurationMilliseconds: Double = 0,
        lastPlaybackFrameIndex: Int? = nil,
        hibikiInferenceStatus: String = "idle",
        hibikiStepCount: Int = 0,
        hibikiTextTokenCount: Int = 0,
        hibikiVisibleTextCount: Int = 0,
        hibikiGeneratedAudioFrameCount: Int = 0,
        hibikiSamplingSummary: String = "n/a"
    ) {
        self.progress = progress
        self.artifactPreparationSummary = artifactPreparationSummary
        self.artifactFileName = artifactFileName
        self.artifactCompletedFileCount = artifactCompletedFileCount
        self.artifactTotalFileCount = artifactTotalFileCount
        self.artifactFileProgress = artifactFileProgress
        self.eventCount = eventCount
        self.lastEventName = lastEventName
        self.output = output
        self.audioInputStatus = audioInputStatus
        self.audioChunkCount = audioChunkCount
        self.audioSampleRate = audioSampleRate
        self.audioDurationMilliseconds = audioDurationMilliseconds
        self.lastAudioFrameIndex = lastAudioFrameIndex
        self.mimiEncodeStatus = mimiEncodeStatus
        self.mimiEncodedFrameCount = mimiEncodedFrameCount
        self.mimiCodebookCount = mimiCodebookCount
        self.mimiTokenCount = mimiTokenCount
        self.mimiFrameDurationMilliseconds = mimiFrameDurationMilliseconds
        self.lastMimiFrameIndex = lastMimiFrameIndex
        self.mimiDecodeStatus = mimiDecodeStatus
        self.decodedAudioChunkCount = decodedAudioChunkCount
        self.decodedAudioSampleRate = decodedAudioSampleRate
        self.decodedAudioDurationMilliseconds = decodedAudioDurationMilliseconds
        self.lastDecodedAudioFrameIndex = lastDecodedAudioFrameIndex
        self.playbackStatus = playbackStatus
        self.playbackChunkCount = playbackChunkCount
        self.playbackDurationMilliseconds = playbackDurationMilliseconds
        self.lastPlaybackFrameIndex = lastPlaybackFrameIndex
        self.hibikiInferenceStatus = hibikiInferenceStatus
        self.hibikiStepCount = hibikiStepCount
        self.hibikiTextTokenCount = hibikiTextTokenCount
        self.hibikiVisibleTextCount = hibikiVisibleTextCount
        self.hibikiGeneratedAudioFrameCount = hibikiGeneratedAudioFrameCount
        self.hibikiSamplingSummary = hibikiSamplingSummary
    }

    mutating func record(_ event: ExperimentEvent) {
        eventCount += 1
        lastEventName = event.name
        if case let .observation(line) = event {
            if output.isEmpty {
                output = line
            } else {
                output += "\n\(line)"
            }
        }
    }

    mutating func recordArtifactPreparation(_ progress: ArtifactPreparationProgress) {
        artifactPreparationSummary = progress.summary
        artifactFileName = progress.fileName
        artifactTotalFileCount = progress.totalFileCount
        if progress.phase == .completed {
            artifactCompletedFileCount = min(progress.completedFileCount + 1, progress.totalFileCount)
        } else {
            artifactCompletedFileCount = progress.completedFileCount
        }
        artifactFileProgress = progress.fileFractionCompleted
        self.progress = progress.overallFractionCompleted
    }

    mutating func recordAudioInput(_ event: AudioInputEvent) {
        switch event {
        case let .streamStarted(sampleRate):
            audioInputStatus = "streaming"
            audioSampleRate = sampleRate
        case let .chunk(chunk):
            audioInputStatus = "streaming"
            audioChunkCount += 1
            audioSampleRate = chunk.sampleRate
            audioDurationMilliseconds += chunk.durationMilliseconds
            lastAudioFrameIndex = chunk.frameIndex
        case .streamStopped:
            audioInputStatus = "stopped"
        case let .streamFailed(message):
            audioInputStatus = "failed: \(message)"
        }
    }

    mutating func recordMimiEncode(_ event: MimiEncodeEvent) {
        switch event {
        case let .streamStarted(description):
            mimiEncodeStatus = "encoding"
            mimiCodebookCount = description.codebookCount
            mimiFrameDurationMilliseconds = description.frameDurationMilliseconds
        case let .frame(frame):
            mimiEncodeStatus = "encoding"
            mimiEncodedFrameCount += 1
            mimiCodebookCount = frame.codebookCount
            mimiTokenCount += frame.tokens.count
            lastMimiFrameIndex = frame.frameIndex
        case .streamStopped:
            mimiEncodeStatus = "stopped"
        case let .streamFailed(message):
            mimiEncodeStatus = "failed: \(message)"
        }
    }

    mutating func recordMimiDecode(_ event: MimiDecodeEvent) {
        switch event {
        case let .streamStarted(description):
            mimiDecodeStatus = "decoding"
            decodedAudioSampleRate = description.sampleRate
        case let .chunk(chunk):
            mimiDecodeStatus = "decoding"
            decodedAudioChunkCount += 1
            decodedAudioSampleRate = chunk.sampleRate
            decodedAudioDurationMilliseconds += chunk.durationMilliseconds
            lastDecodedAudioFrameIndex = chunk.frameIndex
        case .streamStopped:
            mimiDecodeStatus = "stopped"
        case let .streamFailed(message):
            mimiDecodeStatus = "failed: \(message)"
        }
    }

    mutating func recordPlayback(_ event: PlaybackEvent) {
        switch event {
        case .streamStarted:
            playbackStatus = "streaming"
        case let .chunk(chunk):
            playbackStatus = "streaming"
            playbackChunkCount += 1
            playbackDurationMilliseconds += chunk.durationMilliseconds
            lastPlaybackFrameIndex = chunk.frameIndex
        case .streamStopped:
            playbackStatus = "stopped"
        case let .streamFailed(message):
            playbackStatus = "failed: \(message)"
        }
    }

    mutating func recordHibikiInference(_ event: HibikiInferenceEvent) {
        switch event {
        case let .streamStarted(description):
            hibikiInferenceStatus = "ready"
            hibikiSamplingSummary = "temp \(description.configuration.temperature), top-k \(description.configuration.topK)"
        case .sourceTokens:
            hibikiInferenceStatus = "running"
        case .step:
            hibikiInferenceStatus = "running"
            hibikiStepCount += 1
        case let .text(text):
            hibikiTextTokenCount += 1
            if let piece = text.piece, !piece.isEmpty {
                hibikiVisibleTextCount += 1
                output += piece
            }
        case .generatedAudio:
            hibikiGeneratedAudioFrameCount += 1
        case .streamStopped:
            hibikiInferenceStatus = "stopped"
        case let .streamFailed(message):
            hibikiInferenceStatus = "failed: \(message)"
        }
    }
}

@MainActor
public protocol ExperimentBackend {
    func prepareEvents() async -> [ExperimentEvent]
    func prepareEvents(send: @escaping @MainActor (ExperimentEvent) -> Void) async
    func runEvents() async -> [ExperimentEvent]
    func stop()
}

public extension ExperimentBackend {
    func prepareEvents(send: @escaping @MainActor (ExperimentEvent) -> Void) async {
        for event in await prepareEvents() {
            send(event)
        }
    }

    func stop() {}
}

public struct ScriptedExperimentBackend: ExperimentBackend {
    private let prepareEventsScript: [ExperimentEvent]
    private let runEventsScript: [ExperimentEvent]

    public init(events: [ExperimentEvent]) {
        self.init(prepareEvents: events, runEvents: [])
    }

    public init(prepareEvents: [ExperimentEvent], runEvents: [ExperimentEvent]) {
        self.prepareEventsScript = prepareEvents
        self.runEventsScript = runEvents
    }

    public func prepareEvents() async -> [ExperimentEvent] {
        prepareEventsScript
    }

    public func runEvents() async -> [ExperimentEvent] {
        runEventsScript
    }
}

public enum ExperimentEvent: Equatable {
    case preparationProgress(Double)
    case artifactPreparationProgress(ArtifactPreparationProgress)
    case observation(String)
    case audioInput(AudioInputEvent)
    case mimiEncode(MimiEncodeEvent)
    case mimiDecode(MimiDecodeEvent)
    case playback(PlaybackEvent)
    case hibikiInference(HibikiInferenceEvent)
    case ready
    case stopped
    case failure(String)

    var name: String {
        switch self {
        case .preparationProgress:
            "preparationProgress"
        case .artifactPreparationProgress:
            "artifactPreparationProgress"
        case let .observation(name):
            name
        case let .audioInput(event):
            "audioInput:\(event.name)"
        case let .mimiEncode(event):
            "codec:\(event.name)"
        case let .mimiDecode(event):
            "codec:\(event.name)"
        case let .playback(event):
            "playback:\(event.name)"
        case let .hibikiInference(event):
            "hibiki:\(event.name)"
        case .ready:
            "ready"
        case .stopped:
            "stopped"
        case .failure:
            "failed"
        }
    }

    var shouldLog: Bool {
        switch self {
        case .ready, .stopped, .failure:
            true
        case let .audioInput(event):
            event.isStreamBoundary
        case let .mimiEncode(event):
            event.isStreamBoundary
        case let .mimiDecode(event):
            event.isStreamBoundary
        case let .playback(event):
            event.isStreamBoundary
        case let .hibikiInference(event):
            event.isStreamBoundary
        case .preparationProgress, .artifactPreparationProgress, .observation:
            false
        }
    }
}

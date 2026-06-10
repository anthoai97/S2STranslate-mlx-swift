import Combine
import Foundation

@MainActor
public final class ExperimentSession: ObservableObject {
    @Published public private(set) var state: ExperimentSessionState
    @Published public private(set) var observations: ExperimentObservations

    private let backend: ExperimentBackend

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
        state = .preparing
        for event in await backend.prepareEvents() {
            apply(event)
        }
    }

    public func start() async {
        guard state == .ready else { return }

        state = .running
        for event in await backend.runEvents() {
            apply(event)
        }
    }

    public func stop() {
        guard state == .running else { return }

        apply(.stopped)
        state = .stopped
    }

    public func newSession() {
        guard state.isTerminal else { return }

        state = .unloaded
        observations = ExperimentObservations()
    }

    public func triggerFailureDemo() {
        apply(.failure("Fake backend failure"))
    }

    private func apply(_ event: ExperimentEvent) {
        observations.record(event)

        switch event {
        case let .preparationProgress(progress):
            observations.progress = progress
        case .observation:
            break
        case .ready:
            state = .ready
        case .stopped:
            break
        case let .failure(message):
            state = .failed(message)
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
    public var eventCount: Int
    public var lastEventName: String
    public var output: String

    nonisolated public init(
        progress: Double = 0,
        eventCount: Int = 0,
        lastEventName: String = "none",
        output: String = ""
    ) {
        self.progress = progress
        self.eventCount = eventCount
        self.lastEventName = lastEventName
        self.output = output
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
}

@MainActor
public protocol ExperimentBackend {
    func prepareEvents() async -> [ExperimentEvent]
    func runEvents() async -> [ExperimentEvent]
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
    case observation(String)
    case ready
    case stopped
    case failure(String)

    var name: String {
        switch self {
        case .preparationProgress:
            "preparationProgress"
        case let .observation(name):
            name
        case .ready:
            "ready"
        case .stopped:
            "stopped"
        case .failure:
            "failed"
        }
    }
}

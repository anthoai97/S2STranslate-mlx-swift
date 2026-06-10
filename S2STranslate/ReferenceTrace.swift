import Foundation

public struct ReferenceTrace: Codable, Equatable, Sendable {
    public var schemaVersion: Int
    public var name: String
    public var source: ReferenceTraceSource
    public var events: [ReferenceTraceEvent]

    nonisolated public init(
        schemaVersion: Int = 1,
        name: String,
        source: ReferenceTraceSource,
        events: [ReferenceTraceEvent]
    ) {
        self.schemaVersion = schemaVersion
        self.name = name
        self.source = source
        self.events = events
    }

    public static func decode(from data: Data) throws -> ReferenceTrace {
        try JSONDecoder().decode(ReferenceTrace.self, from: data)
    }
}

public struct ReferenceTraceSource: Codable, Equatable, Sendable {
    public var reference: String
    public var referenceCommit: String?
    public var modelRevision: String?
    public var fixture: String?

    nonisolated public init(
        reference: String,
        referenceCommit: String? = nil,
        modelRevision: String? = nil,
        fixture: String? = nil
    ) {
        self.reference = reference
        self.referenceCommit = referenceCommit
        self.modelRevision = modelRevision
        self.fixture = fixture
    }
}

public struct ReferenceTraceEvent: Codable, Equatable, Sendable {
    public var stream: ReferenceTraceStream
    public var name: String
    public var frameIndex: Int?
    public var shape: [Int]?
    public var tokens: [Int]?
    public var cadenceMilliseconds: Double?

    nonisolated public init(
        stream: ReferenceTraceStream,
        name: String,
        frameIndex: Int? = nil,
        shape: [Int]? = nil,
        tokens: [Int]? = nil,
        cadenceMilliseconds: Double? = nil
    ) {
        self.stream = stream
        self.name = name
        self.frameIndex = frameIndex
        self.shape = shape
        self.tokens = tokens
        self.cadenceMilliseconds = cadenceMilliseconds
    }

    var eventSignature: String {
        "\(stream.rawValue):\(name)"
    }
}

public enum ReferenceTraceStream: String, Codable, Equatable, Sendable {
    case session
    case codec
    case model
    case audio
    case text
}

public struct ReferenceTraceComparisonOptions: Equatable, Sendable {
    public var compareEventOrder: Bool
    public var compareShapes: Bool
    public var compareTokens: Bool
    public var compareCadence: Bool
    public var frameTolerance: Int
    public var cadenceToleranceMilliseconds: Double

    nonisolated public init(
        compareEventOrder: Bool = true,
        compareShapes: Bool = true,
        compareTokens: Bool = true,
        compareCadence: Bool = true,
        frameTolerance: Int = 0,
        cadenceToleranceMilliseconds: Double = 0
    ) {
        self.compareEventOrder = compareEventOrder
        self.compareShapes = compareShapes
        self.compareTokens = compareTokens
        self.compareCadence = compareCadence
        self.frameTolerance = frameTolerance
        self.cadenceToleranceMilliseconds = cadenceToleranceMilliseconds
    }
}

public struct ReferenceTraceMismatch: Equatable, CustomStringConvertible, Sendable {
    public var eventIndex: Int?
    public var expected: String
    public var actual: String
    public var reason: ReferenceTraceMismatchReason

    nonisolated public init(
        eventIndex: Int?,
        expected: String,
        actual: String,
        reason: ReferenceTraceMismatchReason
    ) {
        self.eventIndex = eventIndex
        self.expected = expected
        self.actual = actual
        self.reason = reason
    }

    public var description: String {
        let prefix = eventIndex.map { "event \($0): " } ?? ""
        return "\(prefix)\(reason.rawValue) expected \(expected), got \(actual)"
    }
}

public enum ReferenceTraceMismatchReason: String, Equatable, Sendable {
    case eventCount
    case eventOrder
    case shape
    case tokens
    case frameCadence
    case timeCadence
}

public enum ReferenceTraceComparator {
    public static func compare(
        expected: ReferenceTrace,
        actual: ReferenceTrace,
        options: ReferenceTraceComparisonOptions = ReferenceTraceComparisonOptions()
    ) -> [ReferenceTraceMismatch] {
        var mismatches: [ReferenceTraceMismatch] = []

        if expected.events.count != actual.events.count {
            mismatches.append(
                ReferenceTraceMismatch(
                    eventIndex: nil,
                    expected: "\(expected.events.count) events",
                    actual: "\(actual.events.count) events",
                    reason: .eventCount
                )
            )
        }

        for index in 0..<min(expected.events.count, actual.events.count) {
            let expectedEvent = expected.events[index]
            let actualEvent = actual.events[index]

            if options.compareEventOrder,
               expectedEvent.eventSignature != actualEvent.eventSignature {
                mismatches.append(
                    ReferenceTraceMismatch(
                        eventIndex: index,
                        expected: expectedEvent.eventSignature,
                        actual: actualEvent.eventSignature,
                        reason: .eventOrder
                    )
                )
            }

            if options.compareShapes,
               let expectedShape = expectedEvent.shape,
               expectedShape != actualEvent.shape {
                mismatches.append(
                    ReferenceTraceMismatch(
                        eventIndex: index,
                        expected: "\(expectedShape)",
                        actual: "\(actualEvent.shape.map(String.init(describing:)) ?? "nil")",
                        reason: .shape
                    )
                )
            }

            if options.compareTokens,
               let expectedTokens = expectedEvent.tokens,
               expectedTokens != actualEvent.tokens {
                mismatches.append(
                    ReferenceTraceMismatch(
                        eventIndex: index,
                        expected: "\(expectedTokens)",
                        actual: "\(actualEvent.tokens.map(String.init(describing:)) ?? "nil")",
                        reason: .tokens
                    )
                )
            }

            if options.compareCadence {
                compareFrameCadence(
                    expected: expectedEvent,
                    actual: actualEvent,
                    options: options,
                    eventIndex: index,
                    mismatches: &mismatches
                )
                compareTimeCadence(
                    expected: expectedEvent,
                    actual: actualEvent,
                    options: options,
                    eventIndex: index,
                    mismatches: &mismatches
                )
            }
        }

        return mismatches
    }

    private static func compareFrameCadence(
        expected: ReferenceTraceEvent,
        actual: ReferenceTraceEvent,
        options: ReferenceTraceComparisonOptions,
        eventIndex: Int,
        mismatches: inout [ReferenceTraceMismatch]
    ) {
        guard let expectedFrame = expected.frameIndex else { return }
        guard let actualFrame = actual.frameIndex else {
            mismatches.append(
                ReferenceTraceMismatch(
                    eventIndex: eventIndex,
                    expected: "\(expectedFrame)",
                    actual: "nil",
                    reason: .frameCadence
                )
            )
            return
        }

        guard abs(expectedFrame - actualFrame) > options.frameTolerance else { return }

        mismatches.append(
            ReferenceTraceMismatch(
                eventIndex: eventIndex,
                expected: "\(expectedFrame) +/- \(options.frameTolerance)",
                actual: "\(actualFrame)",
                reason: .frameCadence
            )
        )
    }

    private static func compareTimeCadence(
        expected: ReferenceTraceEvent,
        actual: ReferenceTraceEvent,
        options: ReferenceTraceComparisonOptions,
        eventIndex: Int,
        mismatches: inout [ReferenceTraceMismatch]
    ) {
        guard let expectedCadence = expected.cadenceMilliseconds else { return }
        guard let actualCadence = actual.cadenceMilliseconds else {
            mismatches.append(
                ReferenceTraceMismatch(
                    eventIndex: eventIndex,
                    expected: "\(expectedCadence)",
                    actual: "nil",
                    reason: .timeCadence
                )
            )
            return
        }

        guard abs(expectedCadence - actualCadence) > options.cadenceToleranceMilliseconds else {
            return
        }

        mismatches.append(
            ReferenceTraceMismatch(
                eventIndex: eventIndex,
                expected: "\(expectedCadence) +/- \(options.cadenceToleranceMilliseconds)",
                actual: "\(actualCadence)",
                reason: .timeCadence
            )
        )
    }
}

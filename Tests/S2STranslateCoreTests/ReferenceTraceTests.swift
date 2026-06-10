import Foundation
import Testing

@testable import S2STranslateCore

@Suite("Reference Trace")
struct ReferenceTraceTests {
    @Test("fixture trace decodes from bundled JSON")
    func fixtureTraceDecodesFromBundledJSON() throws {
        let trace = try loadFixtureTrace()

        #expect(trace.schemaVersion == 1)
        #expect(trace.name == "small-french-audio-structure")
        #expect(trace.source.reference == "ref/hibiki-zero-mlx/src/infer_mlx_fast.py")
        #expect(trace.events.count == 5)
        #expect(trace.events[1].shape == [1, 16])
        #expect(trace.events[1].tokens == [101, 102, 103, 104])
    }

    @Test("identical traces compare without mismatches")
    func identicalTracesCompareWithoutMismatches() throws {
        let trace = try loadFixtureTrace()

        let mismatches = ReferenceTraceComparator.compare(expected: trace, actual: trace)

        #expect(mismatches.isEmpty)
    }

    @Test("shape and token mismatches are reported independently")
    func shapeAndTokenMismatchesAreReportedIndependently() throws {
        let expected = try loadFixtureTrace()
        var actual = expected
        actual.events[1].shape = [1, 15]
        actual.events[1].tokens = [101, 999]

        let mismatches = ReferenceTraceComparator.compare(expected: expected, actual: actual)

        #expect(mismatches.map(\.reason).contains(.shape))
        #expect(mismatches.map(\.reason).contains(.tokens))
    }

    @Test("event order mismatches are reported")
    func eventOrderMismatchesAreReported() throws {
        let expected = try loadFixtureTrace()
        var actual = expected
        actual.events.swapAt(1, 2)

        let mismatches = ReferenceTraceComparator.compare(expected: expected, actual: actual)

        #expect(mismatches.contains { $0.reason == .eventOrder })
    }

    @Test("cadence comparison supports frame and time tolerances")
    func cadenceComparisonSupportsTolerances() throws {
        let expected = try loadFixtureTrace()
        var actual = expected
        actual.events[1].frameIndex = 1
        actual.events[1].cadenceMilliseconds = 84

        let tolerant = ReferenceTraceComparator.compare(
            expected: expected,
            actual: actual,
            options: ReferenceTraceComparisonOptions(
                frameTolerance: 1,
                cadenceToleranceMilliseconds: 5
            )
        )
        let strict = ReferenceTraceComparator.compare(expected: expected, actual: actual)

        #expect(!tolerant.contains { $0.reason == .frameCadence || $0.reason == .timeCadence })
        #expect(strict.contains { $0.reason == .frameCadence })
        #expect(strict.contains { $0.reason == .timeCadence })
    }
}

func loadFixtureTrace() throws -> ReferenceTrace {
    let url = try #require(
        Bundle.module.url(
            forResource: "reference-trace-small",
            withExtension: "json"
        )
    )
    let data = try Data(contentsOf: url)
    return try ReferenceTrace.decode(from: data)
}

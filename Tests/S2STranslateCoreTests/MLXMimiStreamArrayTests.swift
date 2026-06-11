import Testing

@testable import S2STranslateCore

@Suite("MLX Mimi Stream Array")
struct MLXMimiStreamArrayTests {
    @Test("empty stream array preserves Moshi empty semantics")
    func emptyStreamArrayPreservesMoshiEmptySemantics() {
        let empty = MLXMimiStreamArray()

        #expect(empty.isEmpty)
        #expect(empty.shape == nil)
        #expect(empty.dim(-1) == 0)
        #expect(empty.asArray() == nil)
        #expect(empty.narrow(offset: 0, length: 1, axis: -1).isEmpty)

        let concatenated = empty.cat2(MLXMimiStreamArray(), axis: -1)
        #expect(concatenated.isEmpty)

        let (lhs, rhs) = empty.split(lhsLength: 1, axis: -1)
        #expect(lhs.isEmpty)
        #expect(rhs.isEmpty)
    }

    @Test("empty stream array map does not call transform")
    func emptyStreamArrayMapDoesNotCallTransform() {
        var transformCallCount = 0
        let mapped = MLXMimiStreamArray().map { array in
            transformCallCount += 1
            return array
        }

        #expect(mapped.isEmpty)
        #expect(transformCallCount == 0)
    }
}

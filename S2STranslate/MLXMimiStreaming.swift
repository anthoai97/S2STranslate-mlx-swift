import MLX
import MLXNN

public final class MLXMimiStreamArray {
    private let inner: MLXArray?

    public init(_ array: MLXArray? = nil) {
        self.inner = array
    }

    public var isEmpty: Bool {
        inner == nil
    }

    public var shape: [Int]? {
        inner?.shape
    }

    public func asArray() -> MLXArray? {
        inner
    }

    public func dim(_ axis: Int) -> Int {
        inner?.dim(axis) ?? 0
    }

    public func eval() {
        inner?.eval()
    }

    public func cat2(_ rhs: MLXMimiStreamArray, axis: Int) -> MLXMimiStreamArray {
        switch (inner, rhs.inner) {
        case (.none, .none):
            MLXMimiStreamArray()
        case let (.some(lhs), .none):
            MLXMimiStreamArray(lhs)
        case let (.none, .some(rhs)):
            MLXMimiStreamArray(rhs)
        case let (.some(lhs), .some(rhs)):
            MLXMimiStreamArray(concatenated([lhs, rhs], axis: axis))
        }
    }

    public func narrow(offset: Int, length: Int, axis: Int) -> MLXMimiStreamArray {
        guard let inner else { return MLXMimiStreamArray() }

        let totalLength = inner.dim(axis)
        guard length > 0 else { return MLXMimiStreamArray() }

        let split = inner.split(indices: [offset, min(totalLength, offset + length)], axis: axis)
        return MLXMimiStreamArray(split[1])
    }

    public func split(lhsLength: Int, axis: Int) -> (MLXMimiStreamArray, MLXMimiStreamArray) {
        guard let inner else {
            return (MLXMimiStreamArray(), MLXMimiStreamArray())
        }

        let totalLength = inner.dim(axis)
        let lhsLength = min(totalLength, lhsLength)

        if lhsLength == 0 {
            return (MLXMimiStreamArray(), MLXMimiStreamArray(inner))
        }
        if lhsLength == totalLength {
            return (MLXMimiStreamArray(inner), MLXMimiStreamArray())
        }

        let split = inner.split(indices: [lhsLength], axis: axis)
        return (MLXMimiStreamArray(split[0]), MLXMimiStreamArray(split[1]))
    }

    public func map(_ transform: (MLXArray) -> MLXArray) -> MLXMimiStreamArray {
        guard let inner else { return MLXMimiStreamArray() }
        return MLXMimiStreamArray(transform(inner))
    }

    public func elu() -> MLXMimiStreamArray {
        map { MLXNN.elu($0, alpha: 1.0) }
    }
}

final class MLXMimiStreamingAdd {
    var previousLeft = MLXMimiStreamArray()
    var previousRight = MLXMimiStreamArray()

    func resetState() {
        previousLeft = MLXMimiStreamArray()
        previousRight = MLXMimiStreamArray()
    }

    func step(_ left: MLXMimiStreamArray, _ right: MLXMimiStreamArray) -> MLXMimiStreamArray {
        let left = previousLeft.cat2(left, axis: -1)
        let right = previousRight.cat2(right, axis: -1)
        let commonLength = min(left.dim(-1), right.dim(-1))
        let (leftNow, leftLater) = left.split(lhsLength: commonLength, axis: -1)
        let (rightNow, rightLater) = right.split(lhsLength: commonLength, axis: -1)
        previousLeft = leftLater
        previousRight = rightLater

        switch (leftNow.asArray(), rightNow.asArray()) {
        case let (.some(leftArray), .some(rightArray)):
            return MLXMimiStreamArray(leftArray + rightArray)
        case (.none, .none):
            return MLXMimiStreamArray()
        case (.some, .none), (.none, .some):
            fatalError("Mimi streaming add reached mismatched non-empty operands")
        }
    }
}

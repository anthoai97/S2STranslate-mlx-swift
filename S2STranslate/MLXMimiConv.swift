import MLX
import MLXFast

final class MLXMimiConv1d {
    let weightShape: [Int]
    let biasShape: [Int]?
    private var weightStorage: MLXArray?
    private var biasStorage: MLXArray?
    var weight: MLXArray {
        get {
            if let weightStorage { return weightStorage }
            let weight = MLXArray.zeros(weightShape, type: Float32.self)
            weightStorage = weight
            return weight
        }
        set {
            weightStorage = newValue
        }
    }
    var bias: MLXArray? {
        get {
            guard let biasShape else { return nil }
            if let biasStorage { return biasStorage }
            let bias = MLXArray.zeros(biasShape, type: Float32.self)
            biasStorage = bias
            return bias
        }
        set {
            biasStorage = newValue
        }
    }
    let stride: Int
    let padding: Int
    let groups: Int
    let dilation: Int

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1,
        dilation: Int = 1,
        bias: Bool = true
    ) {
        self.weightShape = [outputChannels, kernelSize, inputChannels / groups]
        self.biasShape = bias ? [outputChannels] : nil
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv1d(
            x.swappedAxes(-1, -2),
            weight,
            stride: stride,
            padding: padding,
            dilation: dilation,
            groups: groups
        )
        if let bias {
            y = y + bias
        }
        return y.swappedAxes(-1, -2)
    }
}

final class MLXMimiNormConv1d {
    var conv: MLXMimiConv1d

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        groups: Int = 1,
        dilation: Int = 1,
        bias: Bool = true
    ) {
        self.conv = MLXMimiConv1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            groups: groups,
            dilation: dilation,
            bias: bias
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(x)
    }
}

final class MLXMimiConvTransposed1d {
    let weightShape: [Int]
    let biasShape: [Int]?
    private var weightStorage: MLXArray?
    private var biasStorage: MLXArray?
    private var expandedWeightStorage: MLXArray?
    private var expandedGroupsStorage: Int

    var weight: MLXArray {
        get {
            if let weightStorage { return weightStorage }
            let weight = MLXArray.zeros(weightShape, type: Float32.self)
            weightStorage = weight
            refreshExpandedWeight()
            return weight
        }
        set {
            weightStorage = newValue
            refreshExpandedWeight()
        }
    }
    var bias: MLXArray? {
        get {
            guard let biasShape else { return nil }
            if let biasStorage { return biasStorage }
            let bias = MLXArray.zeros(biasShape, type: Float32.self)
            biasStorage = bias
            return bias
        }
        set {
            biasStorage = newValue
        }
    }

    let stride: Int
    let padding: Int
    let groups: Int
    private let inputChannels: Int
    private let outputChannels: Int
    private let kernelSize: Int

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.weightShape = [outputChannels / groups, kernelSize, inputChannels]
        self.biasShape = bias ? [outputChannels] : nil
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.kernelSize = kernelSize
        self.expandedGroupsStorage = groups
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let rawWeight = weight
        let expandedWeight = expandedWeightStorage ?? rawWeight
        var y = convTransposed1d(
            x.swappedAxes(-1, -2),
            expandedWeight,
            stride: stride,
            padding: padding,
            groups: expandedGroupsStorage
        )
        if let bias {
            y = y + bias
        }
        return y.swappedAxes(-1, -2)
    }

    private func refreshExpandedWeight() {
        guard let weightStorage else {
            expandedWeightStorage = nil
            expandedGroupsStorage = groups
            return
        }

        if groups == inputChannels && groups == outputChannels {
            let eye = repeated(
                MLX.eye(outputChannels).asType(weightStorage.dtype).reshaped([outputChannels, 1, outputChannels]),
                count: kernelSize,
                axis: 1
            )
            expandedWeightStorage = repeated(weightStorage, count: groups, axis: 0) * eye
            expandedGroupsStorage = 1
        } else {
            expandedWeightStorage = weightStorage
            expandedGroupsStorage = groups
        }
    }
}

final class MLXMimiNormConvTranspose1d {
    var convtr: MLXMimiConvTransposed1d

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.convtr = MLXMimiConvTransposed1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            groups: groups,
            bias: bias
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        convtr(x)
    }
}

final class MLXMimiStreamableConv1d {
    let padMode: MLXMimiPadMode
    let causal: Bool
    let kernelSize: Int
    var leftPadApplied = false
    var previousInput = MLXMimiStreamArray()
    var conv: MLXMimiNormConv1d
    var weightShape: [Int] { conv.conv.weightShape }
    var stride: Int { conv.conv.stride }
    var dilation: Int { conv.conv.dilation }

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int,
        dilation: Int,
        groups: Int,
        bias: Bool,
        causal: Bool,
        padMode: MLXMimiPadMode
    ) {
        self.padMode = padMode
        self.causal = causal
        self.kernelSize = kernelSize
        self.conv = MLXMimiNormConv1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            groups: groups,
            dilation: dilation,
            bias: bias
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let effectiveKernelSize = (kernelSize - 1) * conv.conv.dilation + 1
        let paddingTotal = effectiveKernelSize - conv.conv.stride
        let extraPadding = getExtraPaddingForConv1d(
            x,
            kernelSize: effectiveKernelSize,
            stride: conv.conv.stride,
            paddingTotal: paddingTotal
        )
        let z = IntOrPair((0, 0))
        let widths: [IntOrPair]
        if causal {
            widths = [z, z, IntOrPair((paddingTotal, extraPadding))]
        } else {
            let paddingRight = paddingTotal / 2
            let paddingLeft = paddingTotal - paddingRight
            widths = [z, z, IntOrPair((paddingLeft, paddingRight + extraPadding))]
        }
        return conv(padded(x, widths: widths, mode: padMode.mlxPadMode))
    }

    func resetState() {
        previousInput = MLXMimiStreamArray()
        leftPadApplied = false
    }

    func step(_ x: MLXMimiStreamArray) -> MLXMimiStreamArray {
        guard var inner = x.asArray() else { return MLXMimiStreamArray() }

        let stride = conv.conv.stride
        let dilation = conv.conv.dilation
        if !leftPadApplied {
            leftPadApplied = true
            let effectiveKernelSize = (kernelSize - 1) * dilation + 1
            let paddingTotal = effectiveKernelSize - stride
            let z = IntOrPair((0, 0))
            inner = padded(
                inner,
                widths: [z, z, IntOrPair((paddingTotal, 0))],
                mode: padMode.mlxPadMode
            )
        }

        let effectiveKernelSize = (kernelSize - 1) * dilation + 1
        var stream = previousInput.cat2(MLXMimiStreamArray(inner), axis: -1)
        let sequenceLength = stream.dim(-1)
        let frameCount = max(sequenceLength + stride - effectiveKernelSize, 0) / stride
        guard frameCount > 0 else {
            previousInput = stream
            return MLXMimiStreamArray()
        }

        let offset = frameCount * stride
        previousInput = stream.narrow(offset: offset, length: sequenceLength - offset, axis: -1)
        let inputLength = (frameCount - 1) * stride + effectiveKernelSize
        stream = stream.narrow(offset: 0, length: inputLength, axis: -1)

        guard let executable = stream.asArray() else { return MLXMimiStreamArray() }
        return MLXMimiStreamArray(conv.conv(executable))
    }
}

final class MLXMimiStreamableConvTranspose1d {
    let causal: Bool
    let kernelSize: Int
    var previousOutput = MLXMimiStreamArray()
    var convtr: MLXMimiNormConvTranspose1d
    var weightShape: [Int] { convtr.convtr.weightShape }

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int,
        groups: Int,
        bias: Bool,
        causal: Bool
    ) {
        self.causal = causal
        self.kernelSize = kernelSize
        self.convtr = MLXMimiNormConvTranspose1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            groups: groups,
            bias: bias
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let stride = convtr.convtr.stride
        let paddingTotal = max(kernelSize - stride, 0)
        let x = convtr(x)
        if causal {
            return unpad1d(x, left: 0, right: paddingTotal)
        }

        let unpadRight = paddingTotal / 2
        let unpadLeft = paddingTotal - unpadRight
        return unpad1d(x, left: unpadLeft, right: unpadRight)
    }

    func resetState() {
        previousOutput = MLXMimiStreamArray()
    }

    func step(_ x: MLXMimiStreamArray) -> MLXMimiStreamArray {
        guard let input = x.asArray() else { return MLXMimiStreamArray() }

        var output = convtr(input)
        let outputLength = output.dim(-1)
        if var previous = previousOutput.asArray() {
            let previousLength = previous.dim(-1)
            if let bias = convtr.convtr.bias {
                previous = previous - bias[.newAxis, 0..., .newAxis]
            }
            let merged = output[.ellipsis, 0..<previousLength] + previous
            let tail = output[.ellipsis, previousLength...]
            output = concatenated([merged, tail], axis: -1)
        }

        let invalidSteps = kernelSize - convtr.convtr.stride
        let (ready, later) = MLXMimiStreamArray(output).split(
            lhsLength: outputLength - invalidSteps,
            axis: -1
        )
        previousOutput = later
        return ready
    }
}

public final class MLXMimiConvDownsample1d {
    public let stride: Int
    public let dimension: Int
    public let causal: Bool
    var conv: MLXMimiStreamableConv1d

    nonisolated public init(stride: Int, dimension: Int, causal: Bool) {
        self.stride = stride
        self.dimension = dimension
        self.causal = causal
        self.conv = MLXMimiStreamableConv1d(
            inputChannels: dimension,
            outputChannels: dimension,
            kernelSize: 2 * stride,
            stride: stride,
            dilation: 1,
            groups: 1,
            bias: false,
            causal: causal,
            padMode: .edge
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(x)
    }

    public func resetState() {
        conv.resetState()
    }

    public func step(_ x: MLXMimiStreamArray) -> MLXMimiStreamArray {
        conv.step(x)
    }
}

public final class MLXMimiConvTrUpsample1d {
    public let stride: Int
    public let dimension: Int
    public let causal: Bool
    var convtr: MLXMimiStreamableConvTranspose1d

    nonisolated public init(stride: Int, dimension: Int, causal: Bool) {
        self.stride = stride
        self.dimension = dimension
        self.causal = causal
        self.convtr = MLXMimiStreamableConvTranspose1d(
            inputChannels: dimension,
            outputChannels: dimension,
            kernelSize: 2 * stride,
            stride: stride,
            groups: dimension,
            bias: false,
            causal: causal
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        convtr(x)
    }

    public func resetState() {
        convtr.resetState()
    }

    public func step(_ x: MLXMimiStreamArray) -> MLXMimiStreamArray {
        convtr.step(x)
    }
}

private func getExtraPaddingForConv1d(
    _ x: MLXArray,
    kernelSize: Int,
    stride: Int,
    paddingTotal: Int
) -> Int {
    let length = x.dim(-1)
    let frameCount = Float(max(length + paddingTotal - kernelSize, 0)) / Float(stride) + 1.0
    let idealLength = (Int(frameCount.rounded(.up)) - 1) * stride + kernelSize - paddingTotal
    return max(0, idealLength - length)
}

private func unpad1d(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
    let length = x.dim(-1)
    return x[.ellipsis, left..<(length - right)]
}

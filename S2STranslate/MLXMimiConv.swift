import MLX

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

    nonisolated public init(stride: Int, dimension: Int, causal: Bool) {
        self.stride = stride
        self.dimension = dimension
        self.causal = causal
    }

    public func resetState() {}
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

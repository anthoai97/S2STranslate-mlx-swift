import MLX
import MLXNN

final class MLXMimiSeanetResnetBlock {
    private let skipAdd = MLXMimiStreamingAdd()
    var block: [MLXMimiStreamableConv1d]
    var shortcut: MLXMimiStreamableConv1d?

    init(
        _ configuration: MLXMimiSeanetConfiguration,
        dimension: Int,
        kernelSizesAndDilations: [(Int, Int)]
    ) {
        let hiddenDimension = dimension / configuration.compress
        var block: [MLXMimiStreamableConv1d] = []
        for (index, item) in kernelSizesAndDilations.enumerated() {
            let inputChannels = index == 0 ? dimension : hiddenDimension
            let outputChannels = index == kernelSizesAndDilations.count - 1
                ? dimension
                : hiddenDimension
            block.append(
                MLXMimiStreamableConv1d(
                    inputChannels: inputChannels,
                    outputChannels: outputChannels,
                    kernelSize: item.0,
                    stride: 1,
                    dilation: item.1,
                    groups: 1,
                    bias: true,
                    causal: configuration.causal,
                    padMode: configuration.padMode
                )
            )
        }

        self.block = block
        self.shortcut = configuration.trueSkip
            ? nil
            : MLXMimiStreamableConv1d(
                inputChannels: dimension,
                outputChannels: dimension,
                kernelSize: 1,
                stride: 1,
                dilation: 1,
                groups: 1,
                bias: true,
                causal: configuration.causal,
                padMode: configuration.padMode
            )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var x = x
        for layer in block {
            x = layer(elu(x, alpha: 1.0))
        }
        if let shortcut {
            return x + shortcut(residual)
        }
        return x + residual
    }

    func resetState() {
        skipAdd.resetState()
        block.forEach { $0.resetState() }
        shortcut?.resetState()
    }

    func step(_ x: MLXMimiStreamArray) -> MLXMimiStreamArray {
        let residual = x
        var x = x
        for layer in block {
            x = layer.step(x.elu())
        }
        if let shortcut {
            return skipAdd.step(x, shortcut.step(residual))
        }
        return skipAdd.step(x, residual)
    }
}

final class MLXMimiEncoderLayer {
    var residuals: [MLXMimiSeanetResnetBlock]
    var downsample: MLXMimiStreamableConv1d

    init(_ configuration: MLXMimiSeanetConfiguration, ratio: Int, multiplier: Int) {
        var residuals: [MLXMimiSeanetResnetBlock] = []
        var dilation = 1
        for _ in 0..<configuration.residualLayerCount {
            residuals.append(
                MLXMimiSeanetResnetBlock(
                    configuration,
                    dimension: multiplier * configuration.filterCount,
                    kernelSizesAndDilations: [
                        (configuration.residualKernelSize, dilation),
                        (1, 1),
                    ]
                )
            )
            dilation *= configuration.dilationBase
        }

        self.residuals = residuals
        self.downsample = MLXMimiStreamableConv1d(
            inputChannels: multiplier * configuration.filterCount,
            outputChannels: multiplier * configuration.filterCount * 2,
            kernelSize: ratio * 2,
            stride: ratio,
            dilation: 1,
            groups: 1,
            bias: true,
            causal: true,
            padMode: configuration.padMode
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for residual in residuals {
            x = residual(x)
        }
        return downsample(elu(x, alpha: 1.0))
    }

    func resetState() {
        residuals.forEach { $0.resetState() }
        downsample.resetState()
    }

    func step(_ x: MLXMimiStreamArray) -> MLXMimiStreamArray {
        var x = x
        for residual in residuals {
            x = residual.step(x)
        }
        return downsample.step(x.elu())
    }
}

final class MLXMimiDecoderLayer {
    var upsample: MLXMimiStreamableConvTranspose1d
    var residuals: [MLXMimiSeanetResnetBlock]

    init(_ configuration: MLXMimiSeanetConfiguration, ratio: Int, multiplier: Int) {
        var residuals: [MLXMimiSeanetResnetBlock] = []
        var dilation = 1
        for _ in 0..<configuration.residualLayerCount {
            residuals.append(
                MLXMimiSeanetResnetBlock(
                    configuration,
                    dimension: multiplier * configuration.filterCount / 2,
                    kernelSizesAndDilations: [
                        (configuration.residualKernelSize, dilation),
                        (1, 1),
                    ]
                )
            )
            dilation *= configuration.dilationBase
        }

        self.upsample = MLXMimiStreamableConvTranspose1d(
            inputChannels: multiplier * configuration.filterCount,
            outputChannels: multiplier * configuration.filterCount / 2,
            kernelSize: ratio * 2,
            stride: ratio,
            groups: 1,
            bias: true,
            causal: configuration.causal
        )
        self.residuals = residuals
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = upsample(elu(x, alpha: 1.0))
        for residual in residuals {
            x = residual(x)
        }
        return x
    }

    func resetState() {
        upsample.resetState()
        residuals.forEach { $0.resetState() }
    }

    func step(_ x: MLXMimiStreamArray) -> MLXMimiStreamArray {
        var x = upsample.step(x.elu())
        for residual in residuals {
            x = residual.step(x)
        }
        return x
    }
}

public final class MLXMimiSeanetEncoder {
    public let configuration: MLXMimiSeanetConfiguration
    var initConv1d: MLXMimiStreamableConv1d
    var layers: [MLXMimiEncoderLayer]
    var finalConv1d: MLXMimiStreamableConv1d

    nonisolated public init(_ configuration: MLXMimiSeanetConfiguration) {
        self.configuration = configuration
        var multiplier = 1
        self.initConv1d = MLXMimiStreamableConv1d(
            inputChannels: configuration.channels,
            outputChannels: multiplier * configuration.filterCount,
            kernelSize: configuration.kernelSize,
            stride: 1,
            dilation: 1,
            groups: 1,
            bias: true,
            causal: configuration.causal,
            padMode: configuration.padMode
        )

        var layers: [MLXMimiEncoderLayer] = []
        for ratio in configuration.ratios.reversed() {
            layers.append(MLXMimiEncoderLayer(configuration, ratio: ratio, multiplier: multiplier))
            multiplier *= 2
        }
        self.layers = layers
        self.finalConv1d = MLXMimiStreamableConv1d(
            inputChannels: multiplier * configuration.filterCount,
            outputChannels: configuration.dimension,
            kernelSize: configuration.lastKernelSize,
            stride: 1,
            dilation: 1,
            groups: 1,
            bias: true,
            causal: configuration.causal,
            padMode: configuration.padMode
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = initConv1d(x)
        for layer in layers {
            x = layer(x)
        }
        return finalConv1d(elu(x, alpha: 1.0))
    }

    public func resetState() {
        initConv1d.resetState()
        layers.forEach { $0.resetState() }
        finalConv1d.resetState()
    }

    public func step(_ x: MLXMimiStreamArray) -> MLXMimiStreamArray {
        var x = initConv1d.step(x)
        for layer in layers {
            x = layer.step(x)
        }
        return finalConv1d.step(x.elu())
    }
}

public final class MLXMimiSeanetDecoder {
    public let configuration: MLXMimiSeanetConfiguration
    var initConv1d: MLXMimiStreamableConv1d
    var layers: [MLXMimiDecoderLayer]
    var finalConv1d: MLXMimiStreamableConv1d

    nonisolated public init(_ configuration: MLXMimiSeanetConfiguration) {
        self.configuration = configuration
        var multiplier = 1 << configuration.ratios.count
        self.initConv1d = MLXMimiStreamableConv1d(
            inputChannels: configuration.dimension,
            outputChannels: multiplier * configuration.filterCount,
            kernelSize: configuration.kernelSize,
            stride: 1,
            dilation: 1,
            groups: 1,
            bias: true,
            causal: configuration.causal,
            padMode: configuration.padMode
        )

        var layers: [MLXMimiDecoderLayer] = []
        for ratio in configuration.ratios {
            layers.append(MLXMimiDecoderLayer(configuration, ratio: ratio, multiplier: multiplier))
            multiplier /= 2
        }
        self.layers = layers
        self.finalConv1d = MLXMimiStreamableConv1d(
            inputChannels: configuration.filterCount,
            outputChannels: configuration.channels,
            kernelSize: configuration.lastKernelSize,
            stride: 1,
            dilation: 1,
            groups: 1,
            bias: true,
            causal: configuration.causal,
            padMode: configuration.padMode
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = initConv1d(x)
        for layer in layers {
            x = layer(x)
        }
        return finalConv1d(elu(x, alpha: 1.0))
    }

    public func resetState() {
        initConv1d.resetState()
        layers.forEach { $0.resetState() }
        finalConv1d.resetState()
    }

    public func step(_ x: MLXMimiStreamArray) -> MLXMimiStreamArray {
        var x = initConv1d.step(x)
        for layer in layers {
            x = layer.step(x)
        }
        return finalConv1d.step(x.elu())
    }
}

import MLX
import Testing

@testable import S2STranslateCore

@Suite("MLX Mimi Model")
struct MLXMimiModelTests {
    @Test("default Mimi 2024 07 configuration matches MoshiLib codec metadata")
    func defaultMimi202407ConfigurationMatchesMoshiLibCodecMetadata() {
        let configuration = MLXMimiConfiguration.mimi202407()

        #expect(configuration.channels == 1)
        #expect(configuration.sampleRate == 24_000)
        #expect(configuration.frameRate == 12.5)
        #expect(configuration.samplesPerFrame == 1_920)
        #expect(configuration.quantizerCodebookCount == 16)
        #expect(configuration.quantizerBins == 2_048)
        #expect(configuration.quantizerDimension == 256)
        #expect(configuration.seanet.dimension == 512)
        #expect(configuration.seanet.ratios == [8, 6, 5, 4])
        #expect(configuration.transformer.modelDimension == 512)
        #expect(configuration.transformer.headCount == 8)
        #expect(configuration.transformer.layerCount == 8)
        #expect(configuration.transformer.context == 250)
        #expect(configuration.transformer.usesRotatingKVCache)
    }

    @Test("model shell instantiates the Mimi component graph")
    func modelShellInstantiatesMimiComponentGraph() {
        let configuration = MLXMimiConfiguration.mimi202407(codebookCount: 16)
        let model = MLXMimiModel(configuration: configuration, batchSize: 1)

        #expect(model.configuration == configuration)
        #expect(model.batchSize == 1)
        #expect(model.encoder.configuration == configuration.seanet)
        #expect(model.decoder.configuration == configuration.seanet)
        #expect(model.encoderTransformer.configuration == configuration.transformer)
        #expect(model.decoderTransformer.configuration == configuration.transformer)
        #expect(model.downsample.stride == 2)
        #expect(model.upsample.stride == 2)
        #expect(model.quantizer.codebookCount == 16)
        #expect(model.quantizer.bins == 2_048)
        #expect(model.quantizer.inputDimension == 512)
        #expect(model.quantizer.outputDimension == 512)
    }

    @Test("downsample keeps empty streaming input empty")
    func downsampleKeepsEmptyStreamingInputEmpty() {
        let downsample = MLXMimiConvDownsample1d(stride: 2, dimension: 512, causal: true)

        let output = downsample.step(MLXMimiStreamArray())

        #expect(output.isEmpty)
        #expect(downsample.conv.leftPadApplied == false)
    }

    @Test("downsample buffers insufficient streaming input without placeholder output")
    func downsampleBuffersInsufficientStreamingInputWithoutPlaceholderOutput() {
        let downsample = MLXMimiConvDownsample1d(stride: 2, dimension: 1, causal: true)
        let input = MLXMimiStreamArray(MLXArray([Float(0)]).reshaped([1, 1, 1]))

        let output = downsample.step(input)

        #expect(output.isEmpty)
        #expect(downsample.conv.leftPadApplied)
        #expect(downsample.conv.previousInput.shape == [1, 1, 3])
    }

    @Test("model reset clears encoder streaming state")
    func modelResetClearsEncoderStreamingState() {
        let model = MLXMimiModel()
        model.downsample.conv.leftPadApplied = true

        model.resetEncodeState()

        #expect(model.downsample.conv.leftPadApplied == false)
    }

    @Test("Seanet encoder owns executable Moshi topology")
    func seanetEncoderOwnsExecutableMoshiTopology() {
        let encoder = MLXMimiSeanetEncoder(.v0_1())

        #expect(encoder.initConv1d.weightShape == [64, 7, 1])
        #expect(encoder.layers.count == 4)
        #expect(encoder.layers[0].downsample.weightShape == [128, 8, 64])
        #expect(encoder.layers[1].downsample.weightShape == [256, 10, 128])
        #expect(encoder.layers[2].downsample.weightShape == [512, 12, 256])
        #expect(encoder.layers[3].downsample.weightShape == [1_024, 16, 512])
        #expect(encoder.finalConv1d.weightShape == [512, 3, 1_024])
    }

    @Test("deterministic Mimi encoder and decoder remain independent from MLX shell")
    func deterministicMimiEncoderAndDecoderRemainIndependentFromMLXShell() async throws {
        let encoder = DeterministicMimiStreamingEncoder()
        let decoder = DeterministicMimiStreamingDecoder()
        let chunk = PCMChunk(
            frameIndex: 0,
            timestampMilliseconds: 0,
            sampleRate: 24_000,
            samples: Array(repeating: 0.1, count: 1_920)
        )

        let encodedFrame = try #require(try await encoder.encode(chunk).first)
        let decodedChunk = try await decoder.decode(encodedFrame)

        #expect(encodedFrame.tokens.count == 16)
        #expect(decodedChunk.sampleRate == 24_000)
        #expect(decodedChunk.samples.count == 1_920)
    }
}

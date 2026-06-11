import Foundation
import Testing

@testable import S2STranslateCore

@Suite("MLX Mimi Weight Loading")
struct MLXMimiWeightLoadingTests {
    @Test("maps Moshi PyTorch keys to MLX Mimi parameter keys")
    func mapsMoshiPyTorchKeysToMLXMimiParameterKeys() {
        #expect(
            MLXMimiWeightKeyMapper.map("encoder.model.0.conv.weight")
                == "encoder.init_conv1d.conv.weight"
        )
        #expect(
            MLXMimiWeightKeyMapper.map("encoder.model.1.block.1.conv.weight")
                == "encoder.layers.0.residuals.0.block.0.conv.weight"
        )
        #expect(
            MLXMimiWeightKeyMapper.map("encoder.model.3.conv.weight")
                == "encoder.layers.0.downsample.conv.weight"
        )
        #expect(
            MLXMimiWeightKeyMapper.map("decoder.model.2.convtr.weight")
                == "decoder.layers.0.upsample.convtr.weight"
        )
        #expect(
            MLXMimiWeightKeyMapper.map("decoder.model.3.block.3.conv.weight")
                == "decoder.layers.0.residuals.0.block.1.conv.weight"
        )
        #expect(
            MLXMimiWeightKeyMapper.map("encoder_transformer.layers.0.self_attn.in_proj_weight")
                == "encoder_transformer.layers.0.self_attn.in_proj.weight"
        )
        #expect(
            MLXMimiWeightKeyMapper.map("decoder_transformer.layers.0.linear1.weight")
                == "decoder_transformer.layers.0.gating.linear1.weight"
        )
        #expect(
            MLXMimiWeightKeyMapper.map("decoder_transformer.layers.0.linear2.weight")
                == "decoder_transformer.layers.0.gating.linear2.weight"
        )
    }

    @Test("maps tensor layouts for convolution weights")
    func mapsTensorLayoutsForConvolutionWeights() throws {
        let conv = MLXMimiWeightTensor(shape: [2, 3, 4])
        let transposedConv = MLXMimiWeightTensor(shape: [2, 3, 4])

        let mappedConv = MLXMimiTensorLayoutMapper.map(
            conv,
            mappedKey: "encoder.init_conv1d.conv.weight"
        )
        let mappedTransposedConv = MLXMimiTensorLayoutMapper.map(
            transposedConv,
            mappedKey: "decoder.layers.0.upsample.convtr.weight"
        )

        #expect(mappedConv.shape == [2, 4, 3])
        #expect(mappedTransposedConv.shape == [3, 4, 2])
    }

    @Test("loads prepared Mimi safetensors through injected array reader")
    func loadsPreparedMimiSafetensorsThroughInjectedArrayReader() throws {
        let directory = try makeMimiWeightsTemporaryDirectory()
        let weightsURL = directory.appendingPathComponent("mimi.safetensors")
        try Data().write(to: weightsURL)
        let artifacts = PreparedModelArtifacts(
            manifest: ModelRuntimeManifest(
                modelRepo: "repo",
                revision: "rev",
                requiredFiles: [
                    ModelArtifactRequirement(role: "mimiWeights", fileName: "mimi.safetensors"),
                ]
            ),
            files: [
                PreparedModelArtifact(
                    role: "mimiWeights",
                    fileName: "mimi.safetensors",
                    location: weightsURL.path,
                    source: .prepared
                ),
            ]
        )
        let loader = MLXMimiWeightLoader(arrayReader: { url in
            #expect(url.path == weightsURL.path)
            return [
                "encoder.model.0.conv.weight": MLXMimiWeightTensor(shape: [2, 3, 4]),
                "decoder.model.2.convtr.weight": MLXMimiWeightTensor(shape: [2, 3, 4]),
            ]
        })

        let loaded = try loader.load(from: artifacts)

        #expect(loaded.artifact.fileName == "mimi.safetensors")
        #expect(loaded.mappedTensors.keys.sorted() == [
            "decoder.layers.0.upsample.convtr.weight",
            "encoder.init_conv1d.conv.weight",
        ])
        #expect(loaded.mappedTensors["encoder.init_conv1d.conv.weight"]?.shape == [2, 4, 3])
        #expect(loaded.mappedTensors["decoder.layers.0.upsample.convtr.weight"]?.shape == [3, 4, 2])
    }

    @Test("weight loader reports unexpected and incompatible mapped arrays")
    func weightLoaderReportsUnexpectedAndIncompatibleMappedArrays() throws {
        let unexpected = MLXMimiWeightLoadError.unexpectedKey("extra.weight")
        let incompatible = MLXMimiWeightLoadError.incompatibleShape(
            key: "encoder.init_conv1d.conv.weight",
            expected: [2, 4, 3],
            actual: [2, 3, 4]
        )

        #expect(unexpected.userVisibleMessage == "Mimi weight key unexpected: extra.weight")
        #expect(
            incompatible.userVisibleMessage
                == "Mimi weight shape incompatible for encoder.init_conv1d.conv.weight: expected [2, 4, 3], got [2, 3, 4]"
        )
    }

    @Test("weight loader validates expected mapped keys and shapes")
    func weightLoaderValidatesExpectedMappedKeysAndShapes() throws {
        let artifacts = try makePreparedMimiArtifacts()
        let loader = MLXMimiWeightLoader(
            expectedShapes: [
                "encoder.init_conv1d.conv.weight": [2, 4, 3],
            ],
            arrayReader: { _ in
                [
                    "encoder.model.0.conv.weight": MLXMimiWeightTensor(shape: [2, 3, 4]),
                ]
            }
        )

        let loaded = try loader.load(from: artifacts)

        #expect(loaded.mappedTensors["encoder.init_conv1d.conv.weight"]?.shape == [2, 4, 3])
    }

    @Test("weight loader reports missing expected mapped key")
    func weightLoaderReportsMissingExpectedMappedKey() throws {
        let artifacts = try makePreparedMimiArtifacts()
        let loader = MLXMimiWeightLoader(
            expectedShapes: [
                "encoder.init_conv1d.conv.weight": [2, 4, 3],
            ],
            arrayReader: { _ in [:] }
        )

        #expect(throws: MLXMimiWeightLoadError.missingKey("encoder.init_conv1d.conv.weight")) {
            try loader.load(from: artifacts)
        }
    }

    @Test("weight loader reports incompatible mapped shape")
    func weightLoaderReportsIncompatibleMappedShape() throws {
        let artifacts = try makePreparedMimiArtifacts()
        let loader = MLXMimiWeightLoader(
            expectedShapes: [
                "encoder.init_conv1d.conv.weight": [2, 4, 3],
            ],
            arrayReader: { _ in
                [
                    "encoder.model.0.conv.weight": MLXMimiWeightTensor(shape: [2, 4, 3]),
                ]
            }
        )

        #expect(
            throws: MLXMimiWeightLoadError.incompatibleShape(
                key: "encoder.init_conv1d.conv.weight",
                expected: [2, 4, 3],
                actual: [2, 3, 4]
            )
        ) {
            try loader.load(from: artifacts)
        }
    }

    @Test("weight loader can reject unexpected mapped keys in strict mode")
    func weightLoaderCanRejectUnexpectedMappedKeysInStrictMode() throws {
        let artifacts = try makePreparedMimiArtifacts()
        let loader = MLXMimiWeightLoader(
            expectedShapes: [
                "encoder.init_conv1d.conv.weight": [2, 4, 3],
            ],
            allowsUnexpectedKeys: false,
            arrayReader: { _ in
                [
                    "encoder.model.0.conv.weight": MLXMimiWeightTensor(shape: [2, 3, 4]),
                    "unexpected.weight": MLXMimiWeightTensor(shape: [1]),
                ]
            }
        )

        #expect(throws: MLXMimiWeightLoadError.unexpectedKey("unexpected.weight")) {
            try loader.load(from: artifacts)
        }
    }

    @Test("shape only loaded weights report missing MLX payload before module parameter conversion")
    func shapeOnlyLoadedWeightsReportMissingMLXPayloadBeforeModuleParameterConversion() throws {
        let artifacts = try makePreparedMimiArtifacts()
        let loader = MLXMimiWeightLoader(
            arrayReader: { _ in
                [
                    "encoder.model.0.conv.weight": MLXMimiWeightTensor(shape: [2, 3, 4]),
                ]
            }
        )
        let loaded = try loader.load(from: artifacts)

        #expect(
            throws: MLXMimiWeightLoadError.loadFailed(
                "Mapped tensor has no MLX array payload: encoder.init_conv1d.conv.weight"
            )
        ) {
            _ = try loaded.moduleParameters()
        }
    }

    @Test("weight loader reports missing role missing file and reader failures")
    func weightLoaderReportsMissingRoleMissingFileAndReaderFailures() throws {
        let missingRole = PreparedModelArtifacts(
            manifest: ModelRuntimeManifest(modelRepo: "repo", revision: "rev", requiredFiles: []),
            files: []
        )
        let missingFile = PreparedModelArtifacts(
            manifest: ModelRuntimeManifest(
                modelRepo: "repo",
                revision: "rev",
                requiredFiles: [
                    ModelArtifactRequirement(role: "mimiWeights", fileName: "mimi.safetensors"),
                ]
            ),
            files: [
                PreparedModelArtifact(
                    role: "mimiWeights",
                    fileName: "mimi.safetensors",
                    location: "/tmp/missing-\(UUID().uuidString).safetensors",
                    source: .prepared
                ),
            ]
        )
        let readerFailure = MLXMimiWeightLoader(
            arrayReader: { _ in throw ExampleReaderError.failed }
        )

        #expect(throws: MLXMimiWeightLoadError.missingArtifactRole("mimiWeights")) {
            try MLXMimiWeightLoader().load(from: missingRole)
        }
        #expect(throws: MLXMimiWeightLoadError.missingArtifactFile("mimi.safetensors")) {
            try MLXMimiWeightLoader().load(from: missingFile)
        }
        #expect(throws: MLXMimiWeightLoadError.loadFailed("failed")) {
            try readerFailure.load(from: try makePreparedMimiArtifacts())
        }
    }
}

private func makeMimiWeightsTemporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func makePreparedMimiArtifacts() throws -> PreparedModelArtifacts {
    let directory = try makeMimiWeightsTemporaryDirectory()
    let weightsURL = directory.appendingPathComponent("mimi.safetensors")
    try Data().write(to: weightsURL)
    return PreparedModelArtifacts(
        manifest: ModelRuntimeManifest(
            modelRepo: "repo",
            revision: "rev",
            requiredFiles: [
                ModelArtifactRequirement(role: "mimiWeights", fileName: "mimi.safetensors"),
            ]
        ),
        files: [
            PreparedModelArtifact(
                role: "mimiWeights",
                fileName: "mimi.safetensors",
                location: weightsURL.path,
                source: .prepared
            ),
        ]
    )
}

private enum ExampleReaderError: Error, CustomStringConvertible {
    case failed

    var description: String {
        "failed"
    }
}

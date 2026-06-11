// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "S2STranslate",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
    ],
    products: [
        .library(name: "S2STranslateCore", targets: ["S2STranslateCore"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/ml-explore/mlx-swift",
            revision: "70dbb62128a5a1471a5ab80363430adb33470cab"
        ),
    ],
    targets: [
        .target(
            name: "S2STranslateCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "S2STranslate",
            exclude: [
                "Assets.xcassets",
                "ContentView.swift",
                "ModelRuntimeManifest.json",
                "S2STranslateApp.swift",
            ],
            sources: [
                "AVAudioPlaybackSink.swift",
                "ExperimentSession.swift",
                "FileAudioInput.swift",
                "MLXHibikiGraphParameters.swift",
                "MLXHibikiModel.swift",
                "MLXHibikiRuntime.swift",
                "MLXMimiConv.swift",
                "MLXMimiGraphParameters.swift",
                "MLXMimiModel.swift",
                "MLXMimiQuantization.swift",
                "MLXMimiRuntime.swift",
                "MLXMimiSeanet.swift",
                "MLXMimiStreaming.swift",
                "MLXMimiTransformer.swift",
                "MLXMimiWeightLoading.swift",
                "ModelArtifactPreparation.swift",
                "ReferenceTrace.swift",
                "StreamingHibikiInference.swift",
                "StreamingMimiDecode.swift",
                "StreamingMimiEncode.swift",
                "StreamingAudioInput.swift",
            ]
        ),
        .testTarget(
            name: "S2STranslateCoreTests",
            dependencies: ["S2STranslateCore"],
            resources: [
                .process("Fixtures"),
            ]
        ),
    ]
)

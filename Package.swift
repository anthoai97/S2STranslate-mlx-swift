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
    targets: [
        .target(
            name: "S2STranslateCore",
            path: "S2STranslate",
            exclude: [
                "Assets.xcassets",
                "ContentView.swift",
                "ModelRuntimeManifest.json",
                "S2STranslateApp.swift",
            ],
            sources: [
                "ExperimentSession.swift",
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

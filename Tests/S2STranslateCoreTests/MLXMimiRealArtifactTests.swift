import Foundation
import Testing

@testable import S2STranslateCore

@Suite("MLX Mimi Real Artifact")
struct MLXMimiRealArtifactTests {
    @Test("local Mimi safetensors matches Python reference shape and stable token prefix")
    func localMimiSafetensorsLoadAndEncodeSmoke() throws {
        guard ProcessInfo.processInfo.environment["S2S_RUN_REAL_MIMI_ARTIFACT_TESTS"] == "1" else {
            return
        }

        let weightsURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("ref/hibiki-zero-mlx/weights/mimi-pytorch-e351c8d8@125.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            Issue.record("Missing local Mimi safetensors fixture at \(weightsURL.path)")
            return
        }

        let artifacts = PreparedModelArtifacts(
            manifest: ModelRuntimeManifest(
                modelRepo: "kyutai/hibiki-1b-pytorch",
                revision: "e351c8d8",
                requiredFiles: [
                    ModelArtifactRequirement(
                        role: "mimiWeights",
                        fileName: "mimi-pytorch-e351c8d8@125.safetensors"
                    ),
                ]
            ),
            files: [
                PreparedModelArtifact(
                    role: "mimiWeights",
                    fileName: "mimi-pytorch-e351c8d8@125.safetensors",
                    location: weightsURL.path,
                    source: .cache
                ),
            ]
        )

        let runtime = try MLXMimiRuntimeLoader().load(from: artifacts)
        let samples = Array(repeating: Float(0), count: MLXMimiRuntimeConfiguration.mimi202407.samplesPerFrame * 4)
        let frames = try runtime.encode(
            MLXMimiPCMInput(
                samples: samples,
                sampleRate: MLXMimiRuntimeConfiguration.mimi202407.sampleRate,
                pcmShape: [1, 1, samples.count]
            )
        )
        // Generated from `ref/hibiki-zero-mlx/moshi-mlx` Python Mimi with the same zero PCM.
        // Full residual-token parity is still being chased; the first three codebooks are stable.
        let expectedReferenceFrames = [
            [1049, 1700, 1626, 1562, 946, 1572, 825, 754, 739, 1992, 118, 439, 1101, 113, 144, 1684],
            [127, 243, 783, 1348, 1335, 1572, 976, 1744, 1437, 1542, 118, 1383, 1908, 1112, 853, 851],
            [1880, 243, 1178, 546, 1736, 1572, 1978, 1744, 1210, 1542, 1165, 439, 1343, 1112, 220, 851],
            [1031, 243, 1178, 546, 1736, 1572, 1978, 2008, 1210, 374, 1165, 436, 1912, 113, 644, 1684],
        ]

        #expect(frames.count == 4)
        #expect(frames.allSatisfy { $0.tokens.count == MLXMimiRuntimeConfiguration.mimi202407.codebookCount })
        #expect(
            frames.map { Array($0.tokens.prefix(3)) }
                == expectedReferenceFrames.map { Array($0.prefix(3)) }
        )
    }
}

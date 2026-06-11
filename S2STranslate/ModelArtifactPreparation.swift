import Foundation

public struct ModelRuntimeManifest: Equatable, Sendable {
    public var modelRepo: String
    public var revision: String
    public var requiredFiles: [ModelArtifactRequirement]

    nonisolated public init(
        modelRepo: String,
        revision: String,
        requiredFiles: [ModelArtifactRequirement]
    ) {
        self.modelRepo = modelRepo
        self.revision = revision
        self.requiredFiles = requiredFiles
    }

    public static func decode(from data: Data) throws -> ModelRuntimeManifest {
        let decoded = try JSONDecoder().decode(ModelRuntimeManifestDTO.self, from: data)
        return ModelRuntimeManifest(
            modelRepo: decoded.modelRepo,
            revision: decoded.revision,
            requiredFiles: decoded.requiredFiles
                .map { ModelArtifactRequirement(role: $0.key, fileName: $0.value) }
                .sorted { $0.role < $1.role }
        )
    }

    public static let hibikiQ4Default = ModelRuntimeManifest(
        modelRepo: "anquachdev/hbk-zero-3b-mlx-q4",
        revision: "558daadd9272df9432642783b57b02756ff34d5b",
        requiredFiles: [
            ModelArtifactRequirement(role: "architectureConfig", fileName: "config.json"),
            ModelArtifactRequirement(role: "hibikiWeights", fileName: "hibiki.q4.safetensors"),
            ModelArtifactRequirement(role: "mimiWeights", fileName: "mimi-pytorch-e351c8d8@125.safetensors"),
            ModelArtifactRequirement(role: "tokenizer", fileName: "tokenizer_spm_48k_multi6_2.model"),
        ]
    )
}

private struct ModelRuntimeManifestDTO: Decodable {
    var modelRepo: String
    var revision: String
    var requiredFiles: [String: String]
}

public struct ModelArtifactRequirement: Equatable, Sendable {
    public var role: String
    public var fileName: String

    nonisolated public init(role: String, fileName: String) {
        self.role = role
        self.fileName = fileName
    }
}

public struct ModelArtifactHandle: Equatable, Sendable {
    public var fileName: String
    public var location: String
    public var byteCount: Int64
    public var integrity: ModelArtifactIntegrity

    nonisolated public init(
        fileName: String,
        location: String,
        byteCount: Int64,
        integrity: ModelArtifactIntegrity = .valid
    ) {
        self.fileName = fileName
        self.location = location
        self.byteCount = byteCount
        self.integrity = integrity
    }
}

public enum ModelArtifactIntegrity: Equatable, Sendable {
    case valid
    case corrupt
    case incompatible
    case tooLarge
}

public enum ModelArtifactSource: Equatable, Sendable {
    case cache
    case prepared
}

public struct PreparedModelArtifact: Equatable, Sendable {
    public var role: String
    public var fileName: String
    public var location: String
    public var source: ModelArtifactSource
}

public struct PreparedModelArtifacts: Equatable, Sendable {
    public var manifest: ModelRuntimeManifest
    public var files: [PreparedModelArtifact]
}

public enum ArtifactPreparationPhase: Equatable, Sendable {
    case checkingCache
    case downloading
    case validating
    case completed
}

public struct ArtifactPreparationProgress: Equatable, Sendable {
    public var fileName: String
    public var phase: ArtifactPreparationPhase
    public var completedFileCount: Int
    public var totalFileCount: Int
    public var downloadedByteCount: Int64?
    public var expectedByteCount: Int64?

    nonisolated public init(
        fileName: String,
        phase: ArtifactPreparationPhase,
        completedFileCount: Int,
        totalFileCount: Int,
        downloadedByteCount: Int64? = nil,
        expectedByteCount: Int64? = nil
    ) {
        self.fileName = fileName
        self.phase = phase
        self.completedFileCount = completedFileCount
        self.totalFileCount = totalFileCount
        self.downloadedByteCount = downloadedByteCount
        self.expectedByteCount = expectedByteCount
    }

    public var fileFractionCompleted: Double? {
        guard let downloadedByteCount, let expectedByteCount, expectedByteCount > 0 else {
            return nil
        }
        return min(1, max(0, Double(downloadedByteCount) / Double(expectedByteCount)))
    }

    public var overallFractionCompleted: Double {
        guard totalFileCount > 0 else { return 1 }
        let fileFraction = fileFractionCompleted ?? (phase == .completed ? 1 : 0)
        return min(1, max(0, (Double(completedFileCount) + fileFraction) / Double(totalFileCount)))
    }

    public var summary: String {
        let count = "\(completedFileCount)/\(totalFileCount)"
        switch phase {
        case .checkingCache:
            return "Checking \(fileName) (\(count))"
        case .downloading:
            let byteSummary = byteProgressSummary.map { " - \($0)" } ?? ""
            if let fileFractionCompleted {
                return "Downloading \(fileName) \(fileFractionCompleted.formatted(.percent.precision(.fractionLength(0))))\(byteSummary) (\(count))"
            }
            return "Downloading \(fileName)\(byteSummary) (\(count))"
        case .validating:
            return "Validating \(fileName) (\(count))"
        case .completed:
            return "Prepared \(fileName) (\(min(completedFileCount + 1, totalFileCount))/\(totalFileCount))"
        }
    }

    private var byteProgressSummary: String? {
        guard let downloadedByteCount else { return nil }
        guard let expectedByteCount, expectedByteCount > 0 else {
            return Self.formatByteCount(downloadedByteCount)
        }
        return "\(Self.formatByteCount(downloadedByteCount)) / \(Self.formatByteCount(expectedByteCount))"
    }

    private static func formatByteCount(_ byteCount: Int64) -> String {
        guard byteCount >= 1024 else { return "\(byteCount) B" }

        let units = ["KB", "MB", "GB", "TB"]
        var value = Double(byteCount)
        var unitIndex = -1
        repeat {
            value /= 1024
            unitIndex += 1
        } while value >= 1024 && unitIndex < units.count - 1

        let precision = value >= 10 ? 0 : 1
        return "\(value.formatted(.number.precision(.fractionLength(precision)))) \(units[unitIndex])"
    }
}

private actor ArtifactPreparationProgressCollector {
    private var progressEvents: [ArtifactPreparationProgress] = []

    func append(_ progress: ArtifactPreparationProgress) {
        progressEvents.append(progress)
    }

    func values() -> [ArtifactPreparationProgress] {
        progressEvents
    }
}

public enum ModelArtifactPreparationError: Error, Equatable, Sendable {
    case missing(String)
    case inaccessible(String)
    case corrupt(String)
    case incompatible(String)
    case tooLarge(String)

    public var userVisibleMessage: String {
        switch self {
        case let .missing(fileName):
            "Model artifact missing: \(fileName)"
        case let .inaccessible(fileName):
            "Model artifact inaccessible: \(fileName)"
        case let .corrupt(fileName):
            "Model artifact corrupt: \(fileName)"
        case let .incompatible(fileName):
            "Model artifact incompatible: \(fileName)"
        case let .tooLarge(fileName):
            "Model artifact too large for this device: \(fileName)"
        }
    }
}

public protocol ModelArtifactProviding: Sendable {
    func cachedArtifact(named fileName: String) async throws -> ModelArtifactHandle?
    func prepareArtifact(
        named fileName: String,
        from modelRepo: String,
        revision: String
    ) async throws -> ModelArtifactHandle
}

public protocol ModelArtifactRepositoryCachingProviding: ModelArtifactProviding {
    func cachedArtifact(
        named fileName: String,
        from modelRepo: String,
        revision: String
    ) async throws -> ModelArtifactHandle?
}

public protocol ModelArtifactProgressReportingProviding: ModelArtifactProviding {
    func prepareArtifact(
        named fileName: String,
        from modelRepo: String,
        revision: String,
        progress: @escaping @Sendable (Int64, Int64?) async -> Void
    ) async throws -> ModelArtifactHandle
}

public struct ModelArtifactPreparationResult: Equatable, Sendable {
    public var progressEvents: [Double]
    public var artifactProgressEvents: [ArtifactPreparationProgress]
    public var artifacts: PreparedModelArtifacts?
    public var failure: ModelArtifactPreparationError?

    public var succeeded: Bool {
        artifacts != nil && failure == nil
    }
}

public struct ModelArtifactPreparer: Sendable {
    private let manifest: ModelRuntimeManifest
    private let provider: any ModelArtifactProviding

    nonisolated public init(manifest: ModelRuntimeManifest, provider: any ModelArtifactProviding) {
        self.manifest = manifest
        self.provider = provider
    }

    public func prepare() async -> ModelArtifactPreparationResult {
        await prepare(progress: nil)
    }

    public func prepare(
        progress: (@MainActor @Sendable (ArtifactPreparationProgress) async -> Void)?
    ) async -> ModelArtifactPreparationResult {
        var progressEvents: [Double] = [0]
        let artifactProgressEvents = ArtifactPreparationProgressCollector()
        var preparedFiles: [PreparedModelArtifact] = []
        let requiredFiles = manifest.requiredFiles

        guard !requiredFiles.isEmpty else {
            return ModelArtifactPreparationResult(
                progressEvents: [1],
                artifactProgressEvents: [],
                artifacts: PreparedModelArtifacts(manifest: manifest, files: []),
                failure: nil
            )
        }

        for (index, requirement) in requiredFiles.enumerated() {
            do {
                let artifact = try await resolveArtifact(
                    for: requirement,
                    completedFileCount: index,
                    totalFileCount: requiredFiles.count
                ) { artifactProgress in
                    await artifactProgressEvents.append(artifactProgress)
                    await progress?(artifactProgress)
                }
                preparedFiles.append(artifact)
                progressEvents.append(Double(index + 1) / Double(requiredFiles.count))
            } catch let error as ModelArtifactPreparationError {
                return ModelArtifactPreparationResult(
                    progressEvents: progressEvents,
                    artifactProgressEvents: await artifactProgressEvents.values(),
                    artifacts: nil,
                    failure: error
                )
            } catch {
                return ModelArtifactPreparationResult(
                    progressEvents: progressEvents,
                    artifactProgressEvents: await artifactProgressEvents.values(),
                    artifacts: nil,
                    failure: .inaccessible(requirement.fileName)
                )
            }
        }

        return ModelArtifactPreparationResult(
            progressEvents: progressEvents,
            artifactProgressEvents: await artifactProgressEvents.values(),
            artifacts: PreparedModelArtifacts(manifest: manifest, files: preparedFiles),
            failure: nil
        )
    }

    private func resolveArtifact(
        for requirement: ModelArtifactRequirement,
        completedFileCount: Int,
        totalFileCount: Int,
        progress: @escaping @MainActor @Sendable (ArtifactPreparationProgress) async -> Void
    ) async throws -> PreparedModelArtifact {
        await progress(ArtifactPreparationProgress(
            fileName: requirement.fileName,
            phase: .checkingCache,
            completedFileCount: completedFileCount,
            totalFileCount: totalFileCount
        ))

        let cachedArtifact: ModelArtifactHandle?
        if let repositoryCachingProvider = provider as? any ModelArtifactRepositoryCachingProviding {
            cachedArtifact = try await repositoryCachingProvider.cachedArtifact(
                named: requirement.fileName,
                from: manifest.modelRepo,
                revision: manifest.revision
            )
        } else {
            cachedArtifact = try await provider.cachedArtifact(named: requirement.fileName)
        }

        if let cached = cachedArtifact {
            await progress(ArtifactPreparationProgress(
                fileName: requirement.fileName,
                phase: .validating,
                completedFileCount: completedFileCount,
                totalFileCount: totalFileCount
            ))
            try validate(cached, for: requirement)
            await progress(ArtifactPreparationProgress(
                fileName: requirement.fileName,
                phase: .completed,
                completedFileCount: completedFileCount,
                totalFileCount: totalFileCount,
                downloadedByteCount: cached.byteCount,
                expectedByteCount: cached.byteCount
            ))
            return PreparedModelArtifact(
                role: requirement.role,
                fileName: cached.fileName,
                location: cached.location,
                source: .cache
            )
        }

        let prepared: ModelArtifactHandle
        if let progressProvider = provider as? any ModelArtifactProgressReportingProviding {
            prepared = try await progressProvider.prepareArtifact(
                named: requirement.fileName,
                from: manifest.modelRepo,
                revision: manifest.revision
            ) { downloadedByteCount, expectedByteCount in
                await progress(ArtifactPreparationProgress(
                    fileName: requirement.fileName,
                    phase: .downloading,
                    completedFileCount: completedFileCount,
                    totalFileCount: totalFileCount,
                    downloadedByteCount: downloadedByteCount,
                    expectedByteCount: expectedByteCount
                ))
            }
        } else {
            await progress(ArtifactPreparationProgress(
                fileName: requirement.fileName,
                phase: .downloading,
                completedFileCount: completedFileCount,
                totalFileCount: totalFileCount
            ))
            prepared = try await provider.prepareArtifact(
                named: requirement.fileName,
                from: manifest.modelRepo,
                revision: manifest.revision
            )
        }

        await progress(ArtifactPreparationProgress(
            fileName: requirement.fileName,
            phase: .validating,
            completedFileCount: completedFileCount,
            totalFileCount: totalFileCount,
            downloadedByteCount: prepared.byteCount,
            expectedByteCount: prepared.byteCount
        ))
        try validate(prepared, for: requirement)
        await progress(ArtifactPreparationProgress(
            fileName: requirement.fileName,
            phase: .completed,
            completedFileCount: completedFileCount,
            totalFileCount: totalFileCount,
            downloadedByteCount: prepared.byteCount,
            expectedByteCount: prepared.byteCount
        ))
        return PreparedModelArtifact(
            role: requirement.role,
            fileName: prepared.fileName,
            location: prepared.location,
            source: .prepared
        )
    }

    private func validate(
        _ handle: ModelArtifactHandle,
        for requirement: ModelArtifactRequirement
    ) throws {
        guard handle.fileName == requirement.fileName else {
            throw ModelArtifactPreparationError.incompatible(requirement.fileName)
        }

        switch handle.integrity {
        case .valid:
            guard handle.byteCount > 0 else {
                throw ModelArtifactPreparationError.corrupt(requirement.fileName)
            }
        case .corrupt:
            throw ModelArtifactPreparationError.corrupt(requirement.fileName)
        case .incompatible:
            throw ModelArtifactPreparationError.incompatible(requirement.fileName)
        case .tooLarge:
            throw ModelArtifactPreparationError.tooLarge(requirement.fileName)
        }
    }
}

public struct ModelArtifactExperimentBackend: ExperimentBackend, Sendable {
    private let preparer: ModelArtifactPreparer
    private let runEventsScript: [ExperimentEvent]

    public init(
        preparer: ModelArtifactPreparer,
        runEvents: [ExperimentEvent] = []
    ) {
        self.preparer = preparer
        self.runEventsScript = runEvents
    }

    public func prepareEvents() async -> [ExperimentEvent] {
        let result = await preparer.prepare()
        return events(for: result)
    }

    public func prepareEvents(send: @escaping @MainActor (ExperimentEvent) -> Void) async {
        let result = await preparer.prepare { artifactProgress in
            await MainActor.run {
                send(.artifactPreparationProgress(artifactProgress))
                send(.preparationProgress(artifactProgress.overallFractionCompleted))
            }
        }

        for event in terminalEvents(for: result) {
            send(event)
        }
    }

    private func events(for result: ModelArtifactPreparationResult) -> [ExperimentEvent] {
        var events = result.artifactProgressEvents.flatMap { artifactProgress in
            [
                ExperimentEvent.artifactPreparationProgress(artifactProgress),
                ExperimentEvent.preparationProgress(artifactProgress.overallFractionCompleted),
            ]
        }
        if events.isEmpty {
            events = result.progressEvents.map(ExperimentEvent.preparationProgress)
        }
        events.append(contentsOf: terminalEvents(for: result))
        return events
    }

    private func terminalEvents(for result: ModelArtifactPreparationResult) -> [ExperimentEvent] {
        if let failure = result.failure {
            return [.failure(failure.userVisibleMessage)]
        } else if let artifacts = result.artifacts {
            let cachedCount = artifacts.files.filter { $0.source == .cache }.count
            let preparedCount = artifacts.files.filter { $0.source == .prepared }.count
            return [
                .observation("Prepared \(artifacts.files.count) model artifacts (\(cachedCount) cached, \(preparedCount) prepared)."),
                .ready,
            ]
        }

        return []
    }

    public func runEvents() async -> [ExperimentEvent] {
        runEventsScript
    }
}

public protocol ModelArtifactDownloading: Sendable {
    func download(
        from url: URL,
        to destinationURL: URL,
        progress: @escaping @Sendable (Int64, Int64?) async -> Void
    ) async throws -> Int64
}

public enum ModelArtifactDownloadError: Error, Sendable {
    case invalidResponse
    case httpStatus(Int)
    case insufficientDiskSpace(requiredBytes: Int64, availableBytes: Int64)
}

public struct URLSessionModelArtifactDownloader: ModelArtifactDownloading, @unchecked Sendable {
    private let configuration: URLSessionConfiguration

    nonisolated public init(configuration: URLSessionConfiguration = .default) {
        self.configuration = configuration
    }

    public func download(
        from url: URL,
        to destinationURL: URL,
        progress: @escaping @Sendable (Int64, Int64?) async -> Void
    ) async throws -> Int64 {
        let delegate = URLSessionArtifactDownloadDelegate(
            destinationURL: destinationURL,
            progress: progress
        )
        let session = URLSession(
            configuration: configuration,
            delegate: delegate,
            delegateQueue: nil
        )
        defer {
            session.invalidateAndCancel()
        }

        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                delegate.start(continuation: continuation)
                session.downloadTask(with: url).resume()
            }
        } onCancel: {
            session.invalidateAndCancel()
        }
    }
}

private final class URLSessionArtifactDownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private static let progressReportByteInterval: Int64 = 8 * 1024 * 1024

    private let destinationURL: URL
    private let progress: @Sendable (Int64, Int64?) async -> Void
    private let lock = NSLock()

    private var continuation: CheckedContinuation<Int64, Error>?
    private var downloadedByteCount: Int64 = 0
    private var expectedByteCount: Int64?
    private var lastReportedByteCount: Int64 = 0
    private var pendingError: Error?
    private var isFinished = false
    private var didCheckDiskSpace = false

    private struct DownloadCompletion {
        var continuation: CheckedContinuation<Int64, Error>
        var result: Result<Int64, Error>
        var expectedByteCount: Int64?
    }

    init(
        destinationURL: URL,
        progress: @escaping @Sendable (Int64, Int64?) async -> Void
    ) {
        self.destinationURL = destinationURL
        self.progress = progress
    }

    func start(continuation: CheckedContinuation<Int64, Error>) {
        lock.withLock {
            self.continuation = continuation
        }
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        let expectedByteCount = totalBytesExpectedToWrite > 0 ? totalBytesExpectedToWrite : nil
        var shouldReport = false
        var reportedExpectedByteCount: Int64?
        var shouldCancel = false

        lock.withLock {
            downloadedByteCount = totalBytesWritten
            if let expectedByteCount {
                self.expectedByteCount = expectedByteCount
            }
            reportedExpectedByteCount = self.expectedByteCount

            if !didCheckDiskSpace,
               let expectedByteCount = self.expectedByteCount,
               let availableBytes = availableDiskBytes(at: destinationURL.deletingLastPathComponent()),
               availableBytes <= expectedByteCount - totalBytesWritten {
                didCheckDiskSpace = true
                pendingError = ModelArtifactDownloadError.insufficientDiskSpace(
                    requiredBytes: expectedByteCount,
                    availableBytes: availableBytes
                )
                shouldCancel = true
            } else if self.expectedByteCount != nil {
                didCheckDiskSpace = true
            }

            if shouldReportProgress(
                downloadedByteCount: totalBytesWritten,
                expectedByteCount: self.expectedByteCount,
                lastReportedByteCount: lastReportedByteCount
            ) {
                shouldReport = true
                lastReportedByteCount = totalBytesWritten
            }
        }

        if shouldReport {
            Task {
                await progress(totalBytesWritten, reportedExpectedByteCount)
            }
        }
        if shouldCancel {
            downloadTask.cancel()
        }
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        do {
            guard let httpResponse = downloadTask.response as? HTTPURLResponse else {
                record(error: ModelArtifactDownloadError.invalidResponse)
                return
            }
            guard (200..<300).contains(httpResponse.statusCode) else {
                record(error: ModelArtifactDownloadError.httpStatus(httpResponse.statusCode))
                return
            }

            if FileManager.default.fileExists(atPath: destinationURL.path) {
                try FileManager.default.removeItem(at: destinationURL)
            }
            try FileManager.default.moveItem(at: location, to: destinationURL)
            let byteCount = try byteCount(at: destinationURL)
            lock.withLock {
                downloadedByteCount = byteCount
            }
        } catch {
            record(error: error)
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        let completion: DownloadCompletion? = lock.withLock {
            guard !isFinished, let continuation else { return nil }
            isFinished = true
            self.continuation = nil

            let result: Result<Int64, Error>
            if let pendingError {
                result = .failure(pendingError)
            } else if let error {
                result = .failure(error)
            } else {
                result = .success(downloadedByteCount)
            }
            return DownloadCompletion(
                continuation: continuation,
                result: result,
                expectedByteCount: expectedByteCount
            )
        }

        guard let completion else { return }
        switch completion.result {
        case let .success(byteCount):
            Task {
                await progress(byteCount, completion.expectedByteCount)
            }
            completion.continuation.resume(returning: byteCount)
        case let .failure(error):
            completion.continuation.resume(throwing: error)
        }
    }

    private func record(error: Error) {
        lock.withLock {
            pendingError = error
        }
    }

    private func shouldReportProgress(
        downloadedByteCount: Int64,
        expectedByteCount: Int64?,
        lastReportedByteCount: Int64
    ) -> Bool {
        guard downloadedByteCount > lastReportedByteCount else { return false }

        if downloadedByteCount - lastReportedByteCount >= Self.progressReportByteInterval {
            return true
        }

        guard let expectedByteCount, expectedByteCount > 0 else {
            return false
        }
        let previousPercent = Int((Double(lastReportedByteCount) / Double(expectedByteCount)) * 100)
        let currentPercent = Int((Double(downloadedByteCount) / Double(expectedByteCount)) * 100)
        return currentPercent > previousPercent
    }

    private func availableDiskBytes(at directory: URL) -> Int64? {
        guard let values = try? directory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]) else {
            return nil
        }
        return values.volumeAvailableCapacityForImportantUsage
    }

    private func byteCount(at url: URL) throws -> Int64 {
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        if let size = attributes[.size] as? NSNumber {
            return size.int64Value
        }
        if let size = attributes[.size] as? Int64 {
            return size
        }
        if let size = attributes[.size] as? UInt64 {
            return Int64(size)
        }
        return 0
    }
}

public actor HuggingFaceModelArtifactProvider: ModelArtifactProgressReportingProviding, ModelArtifactRepositoryCachingProviding {
    private let storeRootURL: URL
    private let downloader: any ModelArtifactDownloading

    public init(
        storeRootURL: URL? = nil,
        downloader: any ModelArtifactDownloading = URLSessionModelArtifactDownloader()
    ) {
        self.downloader = downloader
        self.storeRootURL = storeRootURL ?? Self.defaultStoreRootURL(fileManager: .default)
    }

    public func cachedArtifact(named fileName: String) async throws -> ModelArtifactHandle? {
        nil
    }

    public func cachedArtifact(
        named fileName: String,
        from modelRepo: String,
        revision: String
    ) async throws -> ModelArtifactHandle? {
        let directoryURL = try artifactDirectoryURL(modelRepo: modelRepo, revision: revision)
        try cleanupTemporaryFiles(in: directoryURL)
        let finalURL = directoryURL.appendingPathComponent(fileName, isDirectory: false)
        return try handleIfValidFileExists(fileName: fileName, at: finalURL)
    }

    public func prepareArtifact(
        named fileName: String,
        from modelRepo: String,
        revision: String
    ) async throws -> ModelArtifactHandle {
        try await prepareArtifact(
            named: fileName,
            from: modelRepo,
            revision: revision,
            progress: { _, _ in }
        )
    }

    public func prepareArtifact(
        named fileName: String,
        from modelRepo: String,
        revision: String,
        progress: @escaping @Sendable (Int64, Int64?) async -> Void
    ) async throws -> ModelArtifactHandle {
        let directoryURL = try artifactDirectoryURL(modelRepo: modelRepo, revision: revision)
        try cleanupTemporaryFiles(in: directoryURL)

        let finalURL = directoryURL.appendingPathComponent(fileName, isDirectory: false)
        if let cached = try handleIfValidFileExists(fileName: fileName, at: finalURL) {
            return cached
        }

        let temporaryURL = directoryURL.appendingPathComponent(".\(fileName).download", isDirectory: false)
        if FileManager.default.fileExists(atPath: temporaryURL.path) {
            try FileManager.default.removeItem(at: temporaryURL)
        }

        do {
            let downloadedByteCount = try await downloader.download(
                from: downloadURL(modelRepo: modelRepo, revision: revision, fileName: fileName),
                to: temporaryURL,
                progress: progress
            )
            guard downloadedByteCount > 0 else {
                throw ModelArtifactPreparationError.corrupt(fileName)
            }
            if FileManager.default.fileExists(atPath: finalURL.path) {
                try FileManager.default.removeItem(at: finalURL)
            }
            try FileManager.default.moveItem(at: temporaryURL, to: finalURL)
            return ModelArtifactHandle(
                fileName: fileName,
                location: finalURL.path,
                byteCount: downloadedByteCount
            )
        } catch let error as ModelArtifactPreparationError {
            try? FileManager.default.removeItem(at: temporaryURL)
            throw error
        } catch let error as ModelArtifactDownloadError {
            try? FileManager.default.removeItem(at: temporaryURL)
            throw preparationError(from: error, fileName: fileName)
        } catch {
            try? FileManager.default.removeItem(at: temporaryURL)
            throw ModelArtifactPreparationError.inaccessible(fileName)
        }
    }

    private static func defaultStoreRootURL(fileManager: FileManager) -> URL {
        let applicationSupportURL = try? fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )
        return (applicationSupportURL ?? fileManager.temporaryDirectory)
            .appendingPathComponent("ModelArtifacts", isDirectory: true)
    }

    private func artifactDirectoryURL(modelRepo: String, revision: String) throws -> URL {
        let directoryURL = storeRootURL
            .appendingPathComponent(safePathComponent(modelRepo), isDirectory: true)
            .appendingPathComponent(safePathComponent(revision), isDirectory: true)
        try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
        return directoryURL
    }

    private func cleanupTemporaryFiles(in directoryURL: URL) throws {
        guard let fileURLs = try? FileManager.default.contentsOfDirectory(
            at: directoryURL,
            includingPropertiesForKeys: nil
        ) else {
            return
        }

        for fileURL in fileURLs where fileURL.lastPathComponent.hasSuffix(".download") {
            try FileManager.default.removeItem(at: fileURL)
        }
    }

    private func handleIfValidFileExists(fileName: String, at url: URL) throws -> ModelArtifactHandle? {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return nil
        }
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        let byteCount: Int64
        if let size = attributes[.size] as? NSNumber {
            byteCount = size.int64Value
        } else if let size = attributes[.size] as? Int64 {
            byteCount = size
        } else if let size = attributes[.size] as? UInt64 {
            byteCount = Int64(size)
        } else {
            byteCount = 0
        }
        guard byteCount > 0 else {
            throw ModelArtifactPreparationError.corrupt(fileName)
        }
        return ModelArtifactHandle(fileName: fileName, location: url.path, byteCount: byteCount)
    }

    private func downloadURL(modelRepo: String, revision: String, fileName: String) -> URL {
        var components = URLComponents()
        components.scheme = "https"
        components.host = "huggingface.co"
        components.path = "/\(modelRepo)/resolve/\(revision)/\(fileName)"
        return components.url!
    }

    private func safePathComponent(_ value: String) -> String {
        value
            .replacingOccurrences(of: "/", with: "__")
            .replacingOccurrences(of: ":", with: "_")
    }

    private func preparationError(
        from error: ModelArtifactDownloadError,
        fileName: String
    ) -> ModelArtifactPreparationError {
        switch error {
        case .insufficientDiskSpace:
            return .tooLarge(fileName)
        case .invalidResponse, .httpStatus:
            return .inaccessible(fileName)
        }
    }
}

public actor DemoModelArtifactProvider: ModelArtifactProviding {
    public init() {}

    public func cachedArtifact(named fileName: String) async throws -> ModelArtifactHandle? {
        nil
    }

    public func prepareArtifact(
        named fileName: String,
        from modelRepo: String,
        revision: String
    ) async throws -> ModelArtifactHandle {
        ModelArtifactHandle(
            fileName: fileName,
            location: "demo-cache://\(modelRepo)/\(revision)/\(fileName)",
            byteCount: 1
        )
    }
}

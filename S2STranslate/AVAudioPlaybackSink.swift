@preconcurrency import AVFoundation
import OSLog
import Synchronization

private let audioPlaybackLogger = Logger(subsystem: "andyq.S2STranslate", category: "AudioPlayback")

protocol AudioPlaybackEngine: Sendable {
    func start(sampleRate: Int) throws
    func schedule(samples: [Float], sampleRate: Int) throws
    func finish() async throws
    func stop()
    func reset()
}

public final class AVAudioPlaybackSink: PlaybackSink, @unchecked Sendable {
    private let engine: any AudioPlaybackEngine
    private let state = Mutex(AVAudioPlaybackSinkState())

    public convenience init() {
        self.init(engine: AVFoundationAudioPlaybackEngine())
    }

    init(engine: any AudioPlaybackEngine) {
        self.engine = engine
    }

    public func start(sampleRate: Int) async throws {
        do {
            try engine.start(sampleRate: sampleRate)
            state.withLock { state in
                state.sampleRate = sampleRate
                state.started = true
                state.stopped = false
            }
        } catch {
            throw PlaybackSinkError.unavailable(String(describing: error))
        }
    }

    public func receive(_ chunk: DecodedAudioChunk) async throws {
        let sampleRate = try state.withLock { state in
            guard state.started, !state.stopped else {
                throw PlaybackSinkError.stopped
            }
            guard state.sampleRate == chunk.sampleRate else {
                throw PlaybackSinkError.unavailable(
                    "sample rate changed from \(state.sampleRate ?? 0) to \(chunk.sampleRate)"
                )
            }
            return chunk.sampleRate
        }

        do {
            try engine.schedule(samples: chunk.samples, sampleRate: sampleRate)
        } catch let error as PlaybackSinkError {
            throw error
        } catch {
            throw PlaybackSinkError.unavailable(String(describing: error))
        }
    }

    public func finish() async throws {
        let shouldFinish = state.withLock { state in
            state.started && !state.stopped
        }
        guard shouldFinish else { return }

        do {
            try await engine.finish()
        } catch let error as PlaybackSinkError {
            throw error
        } catch {
            throw PlaybackSinkError.unavailable(String(describing: error))
        }
    }

    public func stop() {
        state.withLock { state in
            state.stopped = true
            state.started = false
        }
        engine.stop()
    }

    public func reset() {
        state.withLock { state in
            state.sampleRate = nil
            state.started = false
            state.stopped = false
        }
        engine.reset()
    }
}

public final class DeferredAudioPlaybackSink: PlaybackSink, @unchecked Sendable {
    private let wrapped: any PlaybackSink
    private let state = Mutex(DeferredAudioPlaybackSinkState())

    public convenience init() {
        self.init(wrapped: AVAudioPlaybackSink())
    }

    init(wrapped: any PlaybackSink) {
        self.wrapped = wrapped
    }

    public func start(sampleRate: Int) async throws {
        state.withLock { state in
            state.sampleRate = sampleRate
            state.started = true
            state.stopped = false
            state.chunks.removeAll()
        }
    }

    public func receive(_ chunk: DecodedAudioChunk) async throws {
        try state.withLock { state in
            guard state.started, !state.stopped else {
                throw PlaybackSinkError.stopped
            }
            guard state.sampleRate == chunk.sampleRate else {
                throw PlaybackSinkError.unavailable(
                    "sample rate changed from \(state.sampleRate ?? 0) to \(chunk.sampleRate)"
                )
            }
            state.chunks.append(chunk)
        }
    }

    public func finish() async throws {
        let snapshot = try state.withLock { state in
            guard state.started, !state.stopped else {
                throw PlaybackSinkError.stopped
            }
            guard let sampleRate = state.sampleRate else {
                throw PlaybackSinkError.stopped
            }
            return (sampleRate, state.chunks)
        }
        guard !snapshot.1.isEmpty else { return }
        let startedAt = Date()
        let sampleCount = snapshot.1.reduce(0) { $0 + $1.samples.count }
        let durationSeconds = Double(sampleCount) / Double(snapshot.0)
        audioPlaybackLogger.info("deferred playback begin chunks=\(snapshot.1.count, privacy: .public) samples=\(sampleCount, privacy: .public) durationSeconds=\(durationSeconds, privacy: .public)")

        wrapped.reset()
        try await wrapped.start(sampleRate: snapshot.0)
        for chunk in snapshot.1 {
            try await wrapped.receive(chunk)
        }
        audioPlaybackLogger.info("deferred playback scheduled chunks=\(snapshot.1.count, privacy: .public) elapsedSeconds=\(Date().timeIntervalSince(startedAt), privacy: .public)")
        try await wrapped.finish()
        audioPlaybackLogger.info("deferred playback end elapsedSeconds=\(Date().timeIntervalSince(startedAt), privacy: .public)")
    }

    public func stop() {
        state.withLock { state in
            state.stopped = true
            state.started = false
            state.chunks.removeAll()
        }
        wrapped.stop()
    }

    public func reset() {
        state.withLock { state in
            state.sampleRate = nil
            state.started = false
            state.stopped = false
            state.chunks.removeAll()
        }
        wrapped.reset()
    }
}

public final class BufferedStreamingAudioPlaybackSink: PlaybackSink, @unchecked Sendable {
    private let wrapped: any PlaybackSink
    private let prebufferDurationSeconds: Double
    private let state = Mutex(BufferedStreamingAudioPlaybackSinkState())

    public convenience init(prebufferDurationSeconds: Double = 2) {
        self.init(
            wrapped: AVAudioPlaybackSink(),
            prebufferDurationSeconds: prebufferDurationSeconds
        )
    }

    init(wrapped: any PlaybackSink, prebufferDurationSeconds: Double) {
        self.wrapped = wrapped
        self.prebufferDurationSeconds = prebufferDurationSeconds
    }

    public func start(sampleRate: Int) async throws {
        state.withLock { state in
            state.sampleRate = sampleRate
            state.started = true
            state.stopped = false
            state.wrappedStarted = false
            state.pendingChunks.removeAll()
        }
        wrapped.reset()
        audioPlaybackLogger.info("buffered pseudo-streaming playback armed prebufferSeconds=\(self.prebufferDurationSeconds, privacy: .public)")
    }

    public func receive(_ chunk: DecodedAudioChunk) async throws {
        let action = try state.withLock { state -> BufferedStreamingPlaybackAction in
            guard state.started, !state.stopped else {
                throw PlaybackSinkError.stopped
            }
            guard state.sampleRate == chunk.sampleRate else {
                throw PlaybackSinkError.unavailable(
                    "sample rate changed from \(state.sampleRate ?? 0) to \(chunk.sampleRate)"
                )
            }
            guard !state.wrappedStarted else {
                return .schedule([chunk])
            }

            state.pendingChunks.append(chunk)
            let bufferedSeconds = state.pendingDurationSeconds
            guard bufferedSeconds >= prebufferDurationSeconds else {
                return .wait
            }

            state.wrappedStarted = true
            let chunks = state.pendingChunks
            state.pendingChunks.removeAll()
            return .startAndSchedule(sampleRate: chunk.sampleRate, chunks: chunks, bufferedSeconds: bufferedSeconds)
        }

        try await perform(action)
    }

    public func finish() async throws {
        let action = try state.withLock { state -> BufferedStreamingPlaybackAction in
            guard state.started, !state.stopped else {
                throw PlaybackSinkError.stopped
            }
            guard let sampleRate = state.sampleRate else {
                throw PlaybackSinkError.stopped
            }

            if state.wrappedStarted {
                let chunks = state.pendingChunks
                state.pendingChunks.removeAll()
                return chunks.isEmpty ? .finish : .scheduleThenFinish(chunks)
            }

            let chunks = state.pendingChunks
            state.pendingChunks.removeAll()
            guard !chunks.isEmpty else { return .finish }
            state.wrappedStarted = true
            return .startScheduleThenFinish(sampleRate: sampleRate, chunks: chunks)
        }

        try await perform(action)
    }

    public func stop() {
        state.withLock { state in
            state.started = false
            state.stopped = true
            state.wrappedStarted = false
            state.pendingChunks.removeAll()
        }
        wrapped.stop()
    }

    public func reset() {
        state.withLock { state in
            state.sampleRate = nil
            state.started = false
            state.stopped = false
            state.wrappedStarted = false
            state.pendingChunks.removeAll()
        }
        wrapped.reset()
    }

    private func perform(_ action: BufferedStreamingPlaybackAction) async throws {
        switch action {
        case .wait:
            return
        case let .startAndSchedule(sampleRate, chunks, bufferedSeconds):
            audioPlaybackLogger.info("buffered pseudo-streaming playback start chunks=\(chunks.count, privacy: .public) bufferedSeconds=\(bufferedSeconds, privacy: .public)")
            try await wrapped.start(sampleRate: sampleRate)
            for chunk in chunks {
                try await wrapped.receive(chunk)
            }
        case let .schedule(chunks):
            for chunk in chunks {
                try await wrapped.receive(chunk)
            }
        case let .scheduleThenFinish(chunks):
            for chunk in chunks {
                try await wrapped.receive(chunk)
            }
            try await wrapped.finish()
        case let .startScheduleThenFinish(sampleRate, chunks):
            audioPlaybackLogger.info("buffered pseudo-streaming playback finish starts short stream chunks=\(chunks.count, privacy: .public)")
            try await wrapped.start(sampleRate: sampleRate)
            for chunk in chunks {
                try await wrapped.receive(chunk)
            }
            try await wrapped.finish()
        case .finish:
            try await wrapped.finish()
        }
    }
}

private enum BufferedStreamingPlaybackAction {
    case wait
    case startAndSchedule(sampleRate: Int, chunks: [DecodedAudioChunk], bufferedSeconds: Double)
    case schedule([DecodedAudioChunk])
    case scheduleThenFinish([DecodedAudioChunk])
    case startScheduleThenFinish(sampleRate: Int, chunks: [DecodedAudioChunk])
    case finish
}

private struct BufferedStreamingAudioPlaybackSinkState: Sendable {
    var sampleRate: Int?
    var started = false
    var stopped = false
    var wrappedStarted = false
    var pendingChunks: [DecodedAudioChunk] = []

    var pendingDurationSeconds: Double {
        guard let sampleRate, sampleRate > 0 else { return 0 }
        let sampleCount = pendingChunks.reduce(0) { $0 + $1.samples.count }
        return Double(sampleCount) / Double(sampleRate)
    }
}

private struct DeferredAudioPlaybackSinkState: Sendable {
    var sampleRate: Int?
    var started = false
    var stopped = false
    var chunks: [DecodedAudioChunk] = []
}

private struct AVAudioPlaybackSinkState: Sendable {
    var sampleRate: Int?
    var started = false
    var stopped = false
}

private final class AVFoundationAudioPlaybackEngine: AudioPlaybackEngine, @unchecked Sendable {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let pendingLock = NSLock()
    private var currentFormat: AVAudioFormat?
    private var attached = false
    private var pendingBufferCount = 0
    private var pendingSampleCount = 0
    private var finishContinuations: [CheckedContinuation<Void, Never>] = []

    func start(sampleRate: Int) throws {
        stop()
        #if os(iOS) || os(tvOS)
        try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
        try AVAudioSession.sharedInstance().setActive(true)
        #endif

        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false
        ) else {
            throw PlaybackSinkError.unavailable("could not create AVAudioFormat for \(sampleRate) Hz")
        }

        if !attached {
            engine.attach(player)
            attached = true
        }
        engine.connect(player, to: engine.mainMixerNode, format: format)
        engine.prepare()
        try engine.start()
        player.play()
        currentFormat = format
    }

    func schedule(samples: [Float], sampleRate: Int) throws {
        guard let currentFormat else {
            throw PlaybackSinkError.stopped
        }
        guard Int(currentFormat.sampleRate) == sampleRate else {
            throw PlaybackSinkError.unavailable(
                "sample rate changed from \(Int(currentFormat.sampleRate)) to \(sampleRate)"
            )
        }
        guard !samples.isEmpty else { return }
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: currentFormat,
            frameCapacity: AVAudioFrameCount(samples.count)
        ) else {
            throw PlaybackSinkError.unavailable("could not allocate playback buffer")
        }

        buffer.frameLength = AVAudioFrameCount(samples.count)
        guard let channel = buffer.floatChannelData?[0] else {
            throw PlaybackSinkError.unavailable("playback buffer has no float channel")
        }
        samples.withUnsafeBufferPointer { source in
            channel.update(from: source.baseAddress!, count: samples.count)
        }
        pendingLock.lock()
        pendingBufferCount += 1
        pendingSampleCount += samples.count
        pendingLock.unlock()
        let scheduledSampleCount = samples.count
        player.scheduleBuffer(buffer) { [weak self] in
            self?.completePendingBuffer(sampleCount: scheduledSampleCount)
        }
    }

    func finish() async throws {
        let waitSeconds = finishWaitTimeoutSeconds()
        audioPlaybackLogger.info("av playback finish wait begin pendingBuffers=\(self.pendingBufferSnapshot().buffers, privacy: .public) pendingSamples=\(self.pendingBufferSnapshot().samples, privacy: .public) timeoutSeconds=\(waitSeconds, privacy: .public)")
        let startedAt = Date()
        let timedOut = Mutex(false)
        let timeoutTask = Task { [weak self] in
            let nanoseconds = UInt64(waitSeconds * 1_000_000_000)
            try? await Task.sleep(nanoseconds: nanoseconds)
            guard !Task.isCancelled, let self, self.pendingBufferSnapshot().buffers > 0 else {
                return
            }
            timedOut.withLock { $0 = true }
            self.resolvePendingBuffers()
        }
        await waitForPendingBuffers()
        timeoutTask.cancel()
        if timedOut.withLock({ $0 }) {
            audioPlaybackLogger.error("av playback finish timed out pendingBuffers=\(self.pendingBufferSnapshot().buffers, privacy: .public) pendingSamples=\(self.pendingBufferSnapshot().samples, privacy: .public) elapsedSeconds=\(Date().timeIntervalSince(startedAt), privacy: .public)")
        } else {
            audioPlaybackLogger.info("av playback finish wait end elapsedSeconds=\(Date().timeIntervalSince(startedAt), privacy: .public)")
        }
    }

    func stop() {
        player.stop()
        engine.stop()
        engine.reset()
        currentFormat = nil
        resolvePendingBuffers()
    }

    func reset() {
        stop()
    }

    private func completePendingBuffer(sampleCount: Int) {
        pendingLock.lock()
        pendingBufferCount = max(0, pendingBufferCount - 1)
        pendingSampleCount = max(0, pendingSampleCount - sampleCount)
        let continuations = pendingBufferCount == 0 ? finishContinuations : []
        if pendingBufferCount == 0 {
            finishContinuations.removeAll()
        }
        pendingLock.unlock()

        continuations.forEach { $0.resume() }
    }

    private func resolvePendingBuffers() {
        pendingLock.lock()
        pendingBufferCount = 0
        pendingSampleCount = 0
        let continuations = finishContinuations
        finishContinuations.removeAll()
        pendingLock.unlock()

        continuations.forEach { $0.resume() }
    }

    private func waitForPendingBuffers() async {
        await withCheckedContinuation { continuation in
            pendingLock.lock()
            if pendingBufferCount == 0 {
                pendingLock.unlock()
                continuation.resume()
            } else {
                finishContinuations.append(continuation)
                pendingLock.unlock()
            }
        }
    }

    private func finishWaitTimeoutSeconds() -> Double {
        let snapshot = pendingBufferSnapshot()
        guard let sampleRate = currentFormat?.sampleRate, sampleRate > 0 else {
            return 5
        }
        let pendingDuration = Double(snapshot.samples) / sampleRate
        return min(max(pendingDuration + 5, 5), 30)
    }

    private func pendingBufferSnapshot() -> (buffers: Int, samples: Int) {
        pendingLock.lock()
        defer { pendingLock.unlock() }
        return (pendingBufferCount, pendingSampleCount)
    }
}

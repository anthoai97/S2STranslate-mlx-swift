@preconcurrency import AVFoundation
import Synchronization

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

        wrapped.reset()
        try await wrapped.start(sampleRate: snapshot.0)
        for chunk in snapshot.1 {
            try await wrapped.receive(chunk)
        }
        try await wrapped.finish()
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
        pendingLock.unlock()
        player.scheduleBuffer(buffer) { [weak self] in
            self?.completePendingBuffer()
        }
    }

    func finish() async throws {
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

    private func completePendingBuffer() {
        pendingLock.lock()
        pendingBufferCount = max(0, pendingBufferCount - 1)
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
        let continuations = finishContinuations
        finishContinuations.removeAll()
        pendingLock.unlock()

        continuations.forEach { $0.resume() }
    }
}

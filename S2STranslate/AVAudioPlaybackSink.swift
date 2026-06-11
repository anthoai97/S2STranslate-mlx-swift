@preconcurrency import AVFoundation
import Synchronization

protocol AudioPlaybackEngine: Sendable {
    func start(sampleRate: Int) throws
    func schedule(samples: [Float], sampleRate: Int) throws
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

private struct AVAudioPlaybackSinkState: Sendable {
    var sampleRate: Int?
    var started = false
    var stopped = false
}

private final class AVFoundationAudioPlaybackEngine: AudioPlaybackEngine, @unchecked Sendable {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private var currentFormat: AVAudioFormat?
    private var attached = false

    func start(sampleRate: Int) throws {
        stop()

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
        player.scheduleBuffer(buffer, completionHandler: nil)
    }

    func stop() {
        player.stop()
        engine.stop()
        engine.reset()
        currentFormat = nil
    }

    func reset() {
        stop()
    }
}

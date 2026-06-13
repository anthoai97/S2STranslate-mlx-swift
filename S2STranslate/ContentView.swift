//
//  ContentView.swift
//  S2STranslate
//
//  Created by An Quach on 10/6/26.
//

import Combine
import SwiftUI

struct ContentView: View {
    @StateObject private var inputSelection: DemoAudioInputSelection
    @StateObject private var runMode: DemoRunModeSelection
    @StateObject private var playbackMode: DemoTranslationPlaybackModeSelection
    @StateObject private var session: ExperimentSession
    @State private var showSettings = false
    @State private var showPlaceholderObservations = false

    init() {
        let inputSelection = DemoAudioInputSelection(fixtures: FileAudioFixtureCatalog.frenchFixtures)
        let runMode = DemoRunModeSelection()
        let playbackMode = DemoTranslationPlaybackModeSelection()
        let artifactPreparer = ModelArtifactPreparer(
            manifest: .hibikiQ4Default,
            provider: HuggingFaceModelArtifactProvider()
        )
        let generationConfiguration = HibikiGenerationConfiguration(
            textTemperature: 0.4,
            textTopK: 25,
            tailSilenceFrameCount: 100,
            postInputPaddingStopFrameCount: 12
        )
        _inputSelection = StateObject(wrappedValue: inputSelection)
        _runMode = StateObject(wrappedValue: runMode)
        _playbackMode = StateObject(wrappedValue: playbackMode)
        _session = StateObject(
            wrappedValue: ExperimentSession(
                backend: Self.defaultBackend(
                    runMode: runMode,
                    playbackMode: playbackMode,
                    artifactPreparer: artifactPreparer,
                    audioSource: inputSelection.source,
                    generationConfiguration: generationConfiguration
                )
            )
        )
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                runConfigurationControls

                if showPlaceholderObservations && hasActivity {
                    PlaceholderObservationsPanel(observations: session.observations)
                        .transition(.push(from: .top))
                }

                if session.observations.output.isEmpty {
                    ExperimentInfoPanel(
                        mode: runMode.selectedMode,
                        statusText: statusText,
                        inputTitle: inputSelection.selectedFixture.title
                    )
                        .transition(.opacity)
                } else {
                    ExperimentOutputPanel(output: session.observations.output)
                        .transition(.opacity)
                }

                if hasActivity {
                    ExperimentStatusPanel(
                        state: session.state,
                        progress: session.observations.progress,
                        artifactSummary: session.observations.artifactPreparationSummary,
                        artifactFileProgress: session.observations.artifactFileProgress,
                        eventCount: session.observations.eventCount,
                        lastEventName: session.observations.lastEventName,
                        statusText: statusText
                    )
                    .transition(.push(from: .bottom))
                }

                Spacer(minLength: 0)

                bottomControls
            }
            .padding()
            .navigationTitle("S2STranslate")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .animation(.easeInOut(duration: 0.2), value: session.state)
            .animation(.easeInOut(duration: 0.2), value: session.observations.output)
        }
    }

    private static func defaultBackend(
        runMode: DemoRunModeSelection,
        playbackMode: DemoTranslationPlaybackModeSelection,
        artifactPreparer: ModelArtifactPreparer,
        audioSource: any AudioInputSource,
        generationConfiguration: HibikiGenerationConfiguration
    ) -> any ExperimentBackend {
        DemoModeExperimentBackend(
            runMode: runMode,
            sourcePlaybackBackend: SourceAudioPlaybackExperimentBackend(
                source: audioSource,
                playbackSink: AVAudioPlaybackSink()
            ),
            translateBackend: defaultTranslateBackend(
                playbackMode: playbackMode,
                artifactPreparer: artifactPreparer,
                audioSource: audioSource,
                generationConfiguration: generationConfiguration
            )
        )
    }

    private static func defaultTranslateBackend(
        playbackMode: DemoTranslationPlaybackModeSelection,
        artifactPreparer: ModelArtifactPreparer,
        audioSource: any AudioInputSource,
        generationConfiguration: HibikiGenerationConfiguration
    ) -> any ExperimentBackend {
        #if targetEnvironment(simulator)
        let message = "Real MLX translation is unavailable in iOS Simulator because MLX Metal aborts during GPU initialization; run on device or use the macOS smoke test."
        return ScriptedExperimentBackend(prepareEvents: [
            .mimiEncode(.streamFailed(message)),
            .mimiDecode(.streamFailed(message)),
            .hibikiInference(.streamFailed(message)),
            .failure(message),
        ], runEvents: [])
        #else
        return RealFileHibikiTranslationExperimentBackend(
            artifactPreparer: artifactPreparer,
            audioSource: audioSource,
            playbackRouteProvider: {
                RealtimeOutputPolicy().routePlayback(
                    generatedRealtimeFactor: 0.23,
                    livePlaybackSink: AVAudioPlaybackSink(),
                    forceDiagnosticLivePlayback: playbackMode.forceDiagnosticLivePlayback()
                )
            },
            generationConfiguration: generationConfiguration
        )
        #endif
    }

    private var bottomControls: some View {
        VStack(spacing: 12) {
            if shouldShowPreloadButton {
                Button {
                    Task { await preloadModel() }
                } label: {
                    Label("Preload Model", systemImage: "externaldrive.fill")
                        .frame(minWidth: 180)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
            }

            ZStack {
                Button(action: performPrimaryAction) {
                    Label(primaryActionTitle, systemImage: primaryActionIcon)
                        .font(.title2)
                        .frame(minWidth: 140)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(session.state == .preparing)

                HStack {
                    Spacer()

                    Button {
                        showSettings.toggle()
                    } label: {
                        Image(systemName: "gear")
                            .font(.title2)
                    }
                    .buttonStyle(.borderless)
                    .popover(isPresented: $showSettings, arrowEdge: .bottom) {
                        VStack(alignment: .leading, spacing: 16) {
                            Toggle(isOn: $showPlaceholderObservations) {
                                Label("Session Observations", systemImage: "chart.bar")
                            }

                            Toggle(isOn: diagnosticLivePlaybackBinding) {
                                Label("Diagnostic Live Playback", systemImage: "speaker.wave.2.fill")
                            }
                            .disabled(!canChangeConfiguration)

                            Button(role: .destructive) {
                                showSettings = false
                                session.triggerFailureDemo()
                            } label: {
                                Label("Trigger Failure Demo", systemImage: "exclamationmark.triangle.fill")
                            }
                            .disabled(isTerminal)
                        }
                        .padding()
                        .frame(minWidth: 260)
                        .presentationCompactAdaptation(.popover)
                    }
                }
            }
        }
        .padding(.vertical, 8)
    }

    private var runConfigurationControls: some View {
        VStack(alignment: .leading, spacing: 10) {
            Picker("Mode", selection: runModeBinding) {
                ForEach(DemoRunMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .disabled(!canChangeConfiguration)

            Picker("Sample", selection: selectedFixtureBinding) {
                ForEach(inputSelection.fixtures) { fixture in
                    Text(fixture.title).tag(fixture.id)
                }
            }
            .pickerStyle(.menu)
            .disabled(!canChangeConfiguration)
        }
    }

    private var runModeBinding: Binding<DemoRunMode> {
        Binding(
            get: { runMode.selectedMode },
            set: { mode in
                resetTerminalSessionIfNeeded()
                runMode.select(mode: mode)
            }
        )
    }

    private var selectedFixtureBinding: Binding<String> {
        Binding(
            get: { inputSelection.selectedFixtureID },
            set: { id in
                resetTerminalSessionIfNeeded()
                inputSelection.selectFixture(id: id)
            }
        )
    }

    private var diagnosticLivePlaybackBinding: Binding<Bool> {
        Binding(
            get: { playbackMode.diagnosticLivePlaybackEnabled },
            set: { enabled in
                resetTerminalSessionIfNeeded()
                playbackMode.setDiagnosticLivePlaybackEnabled(enabled)
            }
        )
    }

    private func performPrimaryAction() {
        switch session.state {
        case .unloaded:
            Task { await prepareAndStart() }
        case .ready:
            Task { await startAndStopWhenFinished() }
        case .running:
            session.stop()
        case .stopped, .failed:
            session.newSession()
        case .preparing:
            break
        }
    }

    private func preloadModel() async {
        await session.prepare()
    }

    private func prepareAndStart() async {
        await session.prepare()
        if session.state == .ready {
            await startAndStopWhenFinished()
        }
    }

    private func startAndStopWhenFinished() async {
        await session.start()
        if session.state == .running {
            session.stop()
        }
    }

    private func resetTerminalSessionIfNeeded() {
        guard isTerminal else { return }
        session.newSession()
    }

    private var primaryActionTitle: String {
        switch session.state {
        case .unloaded:
            runMode.selectedMode.primaryActionTitle
        case .preparing:
            "Preparing"
        case .ready:
            runMode.selectedMode.readyActionTitle
        case .running:
            "Stop"
        case .stopped, .failed:
            "Reset"
        }
    }

    private var primaryActionIcon: String {
        switch session.state {
        case .unloaded:
            runMode.selectedMode.primaryActionIcon
        case .preparing:
            "hourglass"
        case .ready:
            "play.circle.fill"
        case .running:
            "stop.circle.fill"
        case .stopped, .failed:
            "plus.circle.fill"
        }
    }

    private var statusText: String {
        switch session.state {
        case .unloaded:
            "Unloaded"
        case .preparing:
            "Preparing"
        case .ready:
            "Ready"
        case .running:
            "Running"
        case .stopped:
            "Stopped"
        case let .failed(message):
            "Failed: \(message)"
        }
    }

    private var hasActivity: Bool {
        session.state != .unloaded || session.observations.eventCount > 0
    }

    private var shouldShowPreloadButton: Bool {
        runMode.selectedMode == .translate && session.state == .unloaded
    }

    private var canChangeConfiguration: Bool {
        switch session.state {
        case .unloaded, .stopped, .failed:
            true
        case .preparing, .ready, .running:
            false
        }
    }

    private var isTerminal: Bool {
        switch session.state {
        case .stopped, .failed:
            true
        case .unloaded, .preparing, .ready, .running:
            false
        }
    }
}

@MainActor
private final class DemoAudioInputSelection: ObservableObject {
    @Published private(set) var selectedFixtureID: String

    let fixtures: [FileAudioFixture]
    let source: ConfigurableAudioInputSource

    init(fixtures: [FileAudioFixture]) {
        let fallback = FileAudioFixtureCatalog.frenchShortForm[0]
        let initialFixture = fixtures.first ?? fallback
        self.fixtures = fixtures.isEmpty ? [fallback] : fixtures
        self.selectedFixtureID = initialFixture.id
        self.source = ConfigurableAudioInputSource(
            source: RemoteAudioFileInputSource(fixture: initialFixture)
        )
    }

    var selectedFixture: FileAudioFixture {
        fixtures.first { $0.id == selectedFixtureID } ?? fixtures[0]
    }

    func selectFixture(id: String) {
        guard let fixture = fixtures.first(where: { $0.id == id }) else { return }
        selectedFixtureID = fixture.id
        source.update(source: RemoteAudioFileInputSource(fixture: fixture))
    }
}

private enum DemoRunMode: String, CaseIterable, Identifiable, Sendable {
    case source
    case translate

    var id: String { rawValue }

    var title: String {
        switch self {
        case .source:
            "Sample"
        case .translate:
            "Translate"
        }
    }

    var panelTitle: String {
        switch self {
        case .source:
            "Sample Playback"
        case .translate:
            "Translation"
        }
    }

    var panelDescription: String {
        switch self {
        case .source:
            "Plays the selected source audio file."
        case .translate:
            "Streams the selected file through Mimi, Hibiki, Mimi decode, and playback."
        }
    }

    var primaryActionTitle: String {
        switch self {
        case .source:
            "Play Sample"
        case .translate:
            "Translate"
        }
    }

    var primaryActionIcon: String {
        switch self {
        case .source:
            "speaker.wave.2.fill"
        case .translate:
            "waveform.and.mic"
        }
    }

    var readyActionTitle: String {
        switch self {
        case .source:
            "Play"
        case .translate:
            "Start Translation"
        }
    }
}

private final class DemoRunModeSelection: ObservableObject, @unchecked Sendable {
    @Published private(set) var selectedMode: DemoRunMode
    private let lock = NSLock()
    private var storedMode: DemoRunMode

    init(initialMode: DemoRunMode = .source) {
        self.selectedMode = initialMode
        self.storedMode = initialMode
    }

    func select(mode: DemoRunMode) {
        lock.lock()
        storedMode = mode
        lock.unlock()
        selectedMode = mode
    }

    func currentMode() -> DemoRunMode {
        lock.lock()
        defer { lock.unlock() }
        return storedMode
    }
}

private final class DemoTranslationPlaybackModeSelection: ObservableObject, @unchecked Sendable {
    @Published private(set) var diagnosticLivePlaybackEnabled = false
    private let lock = NSLock()
    private var storedDiagnosticLivePlaybackEnabled = false

    func setDiagnosticLivePlaybackEnabled(_ enabled: Bool) {
        lock.lock()
        storedDiagnosticLivePlaybackEnabled = enabled
        lock.unlock()
        diagnosticLivePlaybackEnabled = enabled
    }

    func forceDiagnosticLivePlayback() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return storedDiagnosticLivePlaybackEnabled
    }
}

private struct DemoModeExperimentBackend: ExperimentBackend, Sendable {
    private let runMode: DemoRunModeSelection
    private let sourcePlaybackBackend: any ExperimentBackend
    private let translateBackend: any ExperimentBackend

    init(
        runMode: DemoRunModeSelection,
        sourcePlaybackBackend: any ExperimentBackend,
        translateBackend: any ExperimentBackend
    ) {
        self.runMode = runMode
        self.sourcePlaybackBackend = sourcePlaybackBackend
        self.translateBackend = translateBackend
    }

    func prepareEvents() async -> [ExperimentEvent] {
        await selectedBackend.prepareEvents()
    }

    func prepareEvents(send: @escaping @Sendable (ExperimentEvent) async -> Void) async {
        await selectedBackend.prepareEvents(send: send)
    }

    func runEvents() async -> [ExperimentEvent] {
        await selectedBackend.runEvents()
    }

    func runEvents(send: @escaping @Sendable (ExperimentEvent) async -> Void) async {
        await selectedBackend.runEvents(send: send)
    }

    func stop() {
        sourcePlaybackBackend.stop()
        translateBackend.stop()
    }

    private var selectedBackend: any ExperimentBackend {
        switch runMode.currentMode() {
        case .source:
            sourcePlaybackBackend
        case .translate:
            translateBackend
        }
    }
}

private struct ExperimentInfoPanel: View {
    let mode: DemoRunMode
    let statusText: String
    let inputTitle: String

    var body: some View {
        VStack(spacing: 24) {
            Image(systemName: "waveform.and.mic")
                .font(.system(size: 64))
                .foregroundStyle(.blue)

            VStack(spacing: 12) {
                Text(mode.panelTitle)
                    .font(.title)
                    .bold()

                Text(statusText)
                    .font(.headline)

                Text(inputTitle)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                Text(mode.panelDescription)
                    .font(.body)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(RoundedRectangle(cornerRadius: 8).fill(.secondary.opacity(0.1)))
    }
}

private struct ExperimentOutputPanel: View {
    let output: String

    var body: some View {
        ScrollView(.vertical) {
            ScrollViewReader { proxy in
                Text(output)
                    .textSelection(.enabled)
                    .multilineTextAlignment(.leading)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                    .onChange(of: output) { _, _ in
                        withAnimation {
                            proxy.scrollTo("bottom", anchor: .bottom)
                        }
                    }

                Color.clear
                    .frame(height: 1)
                    .id("bottom")
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(RoundedRectangle(cornerRadius: 8).fill(.secondary.opacity(0.1)))
    }
}

private struct ExperimentStatusPanel: View {
    let state: ExperimentSessionState
    let progress: Double
    let artifactSummary: String
    let artifactFileProgress: Double?
    let eventCount: Int
    let lastEventName: String
    let statusText: String

    var body: some View {
        VStack(spacing: 8) {
            if state == .preparing {
                VStack(alignment: .leading, spacing: 6) {
                    ProgressView(value: progress) {
                        Text(artifactSummary == "n/a" ? "Preparing artifacts" : artifactSummary)
                            .font(.callout)
                            .lineLimit(2)
                            .minimumScaleFactor(0.85)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    if let artifactFileProgress {
                        ProgressView(value: artifactFileProgress)
                            .tint(.secondary)
                    }
                }
                .transition(.slide)
            }

            HStack(spacing: 12) {
                Label(statusText, systemImage: statusIcon)

                if state == .running {
                    Image(systemName: "waveform.circle.fill")
                        .foregroundStyle(.blue)
                        .font(.title2)
                        .symbolEffect(.bounce, options: .repeating)
                }

                VStack(alignment: .leading) {
                    Text("Events")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("\(eventCount) events")
                        .font(.body.monospacedDigit())
                }
            }
            .padding()
            .background(RoundedRectangle(cornerRadius: 8).fill(statusTint.opacity(0.1)))

            Text("Last event: \(lastEventName)")
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var statusIcon: String {
        switch state {
        case .unloaded:
            "circle"
        case .preparing:
            "hourglass"
        case .ready:
            "checkmark.circle.fill"
        case .running:
            "play.circle.fill"
        case .stopped:
            "stop.circle.fill"
        case .failed:
            "exclamationmark.triangle.fill"
        }
    }

    private var statusTint: Color {
        switch state {
        case .failed:
            .red
        case .running:
            .blue
        case .ready:
            .green
        case .stopped:
            .secondary
        case .unloaded, .preparing:
            .orange
        }
    }
}

private struct PlaceholderObservationsPanel: View {
    let observations: ExperimentObservations

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Session Observations")
                .font(.headline)

            ScrollView(.vertical) {
                Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 8) {
                    GridRow {
                        Text("Progress")
                            .foregroundStyle(.secondary)
                        Text(observations.progress.formatted(.percent.precision(.fractionLength(0))))
                    }

                    GridRow {
                        Text("Artifact")
                            .foregroundStyle(.secondary)
                        metricValue(observations.artifactPreparationSummary)
                    }

                    GridRow {
                        Text("Artifact File")
                            .foregroundStyle(.secondary)
                        metricValue(observations.artifactFileName)
                    }

                    GridRow {
                        Text("File Progress")
                            .foregroundStyle(.secondary)
                        Text(artifactFileProgressText)
                    }

                    GridRow {
                        Text("Events")
                            .foregroundStyle(.secondary)
                        Text("\(observations.eventCount)")
                    }

                    GridRow {
                        Text("Last")
                            .foregroundStyle(.secondary)
                        metricValue(observations.lastEventName)
                    }

                    GridRow {
                        Text("Output Strategy")
                            .foregroundStyle(.secondary)
                        metricValue(observations.outputStrategySummary)
                    }

                    GridRow {
                        Text("Audio")
                            .foregroundStyle(.secondary)
                        metricValue(observations.audioInputStatus)
                    }

                    GridRow {
                        Text("Chunks")
                            .foregroundStyle(.secondary)
                        Text("\(observations.audioChunkCount)")
                    }

                    GridRow {
                        Text("Sample Rate")
                            .foregroundStyle(.secondary)
                        Text(sampleRateText)
                    }

                    GridRow {
                        Text("Duration")
                            .foregroundStyle(.secondary)
                        Text(durationText)
                    }

                    GridRow {
                        Text("Frame")
                            .foregroundStyle(.secondary)
                        Text(frameText)
                    }

                    GridRow {
                        Text("Mimi")
                            .foregroundStyle(.secondary)
                        metricValue(observations.mimiEncodeStatus)
                    }

                    GridRow {
                        Text("Mimi Frames")
                            .foregroundStyle(.secondary)
                        Text("\(observations.mimiEncodedFrameCount)")
                    }

                    GridRow {
                        Text("Codebooks")
                            .foregroundStyle(.secondary)
                        Text(codebookText)
                    }

                    GridRow {
                        Text("Tokens")
                            .foregroundStyle(.secondary)
                        Text("\(observations.mimiTokenCount)")
                    }

                    GridRow {
                        Text("Mimi Frame")
                            .foregroundStyle(.secondary)
                        Text(mimiFrameText)
                    }

                    GridRow {
                        Text("Hibiki")
                            .foregroundStyle(.secondary)
                        metricValue(observations.hibikiInferenceStatus)
                    }

                    GridRow {
                        Text("Steps")
                            .foregroundStyle(.secondary)
                        Text("\(observations.hibikiStepCount)")
                    }

                    GridRow {
                        Text("Text Tokens")
                            .foregroundStyle(.secondary)
                        Text("\(observations.hibikiTextTokenCount)")
                    }

                    GridRow {
                        Text("Visible Text")
                            .foregroundStyle(.secondary)
                        Text("\(observations.hibikiVisibleTextCount)")
                    }

                    GridRow {
                        Text("Gen Audio")
                            .foregroundStyle(.secondary)
                        Text("\(observations.hibikiGeneratedAudioFrameCount)")
                    }

                    GridRow {
                        Text("Sampling")
                            .foregroundStyle(.secondary)
                        metricValue(observations.hibikiSamplingSummary)
                    }

                    GridRow {
                        Text("Decode")
                            .foregroundStyle(.secondary)
                        metricValue(observations.mimiDecodeStatus)
                    }

                    GridRow {
                        Text("Decoded")
                            .foregroundStyle(.secondary)
                        Text("\(observations.decodedAudioChunkCount)")
                    }

                    GridRow {
                        Text("Decoded Rate")
                            .foregroundStyle(.secondary)
                        Text(decodedRateText)
                    }

                    GridRow {
                        Text("Decoded Dur")
                            .foregroundStyle(.secondary)
                        Text(decodedDurationText)
                    }

                    GridRow {
                        Text("Playback")
                            .foregroundStyle(.secondary)
                        metricValue(observations.playbackStatus)
                    }

                    GridRow {
                        Text("Played")
                            .foregroundStyle(.secondary)
                        Text("\(observations.playbackChunkCount)")
                    }

                    GridRow {
                        Text("Queued")
                            .foregroundStyle(.secondary)
                        Text(playbackScheduledText)
                    }

                    GridRow {
                        Text("Completed")
                            .foregroundStyle(.secondary)
                        Text(playbackCompletedText)
                    }

                    GridRow {
                        Text("Pending")
                            .foregroundStyle(.secondary)
                        Text(playbackPendingText)
                    }

                    GridRow {
                        Text("Gap")
                            .foregroundStyle(.secondary)
                        Text(playbackGapText)
                    }

                    GridRow {
                        Text("Underruns")
                            .foregroundStyle(.secondary)
                        Text("\(observations.playbackUnderrunCount)")
                    }
                }
                .font(.system(.body, design: .monospaced))
                .frame(maxWidth: .infinity, alignment: .leading)

                Text("Real artifact preparation, file PCM chunks, MLX Mimi encode/decode, Hibiki text/audio tokens, decoded PCM chunks, and playback delivery are measured. Audible output and translation quality still require a full smoke run.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.top, 6)
            }
            .frame(maxHeight: 220)
            .scrollIndicators(.visible)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 8).fill(.secondary.opacity(0.1)))
    }

    private func metricValue(_ value: String) -> some View {
        Text(value)
            .lineLimit(2)
            .minimumScaleFactor(0.8)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var sampleRateText: String {
        guard observations.audioSampleRate > 0 else { return "n/a" }
        return "\(observations.audioSampleRate) Hz"
    }

    private var artifactFileProgressText: String {
        guard let artifactFileProgress = observations.artifactFileProgress else { return "n/a" }
        return artifactFileProgress.formatted(.percent.precision(.fractionLength(0)))
    }

    private var durationText: String {
        observations.audioDurationMilliseconds.formatted(.number.precision(.fractionLength(0))) + " ms"
    }

    private var frameText: String {
        guard let frame = observations.lastAudioFrameIndex else { return "n/a" }
        return "\(frame)"
    }

    private var codebookText: String {
        guard observations.mimiCodebookCount > 0 else { return "n/a" }
        return "\(observations.mimiCodebookCount)"
    }

    private var mimiFrameText: String {
        guard let frame = observations.lastMimiFrameIndex else { return "n/a" }
        return "\(frame)"
    }

    private var decodedRateText: String {
        guard observations.decodedAudioSampleRate > 0 else { return "n/a" }
        return "\(observations.decodedAudioSampleRate) Hz"
    }

    private var decodedDurationText: String {
        observations.decodedAudioDurationMilliseconds.formatted(.number.precision(.fractionLength(0))) + " ms"
    }

    private var playbackScheduledText: String {
        observations.playbackScheduledDurationMilliseconds.formatted(.number.precision(.fractionLength(0))) + " ms"
    }

    private var playbackCompletedText: String {
        observations.playbackCompletedDurationMilliseconds.formatted(.number.precision(.fractionLength(0))) + " ms"
    }

    private var playbackPendingText: String {
        observations.playbackPendingDurationMilliseconds.formatted(.number.precision(.fractionLength(0))) + " ms"
    }

    private var playbackGapText: String {
        guard let gap = observations.playbackScheduleGapMilliseconds else { return "n/a" }
        return gap.formatted(.number.precision(.fractionLength(0))) + " ms"
    }
}

#Preview {
    ContentView()
}

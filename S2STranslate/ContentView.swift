//
//  ContentView.swift
//  S2STranslate
//
//  Created by An Quach on 10/6/26.
//

import SwiftUI

struct ContentView: View {
    @StateObject private var session = ExperimentSession(
        backend: ScriptedExperimentBackend(
            prepareEvents: [
                .preparationProgress(0.35),
                .preparationProgress(1.0),
                .ready,
            ],
            runEvents: [
                .observation("fake backend tick"),
                .observation("placeholder output event"),
            ]
        )
    )
    @State private var showSettings = false
    @State private var showPlaceholderObservations = true

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                if showPlaceholderObservations && hasActivity {
                    PlaceholderObservationsPanel(observations: session.observations)
                        .transition(.push(from: .top))
                }

                if session.observations.output.isEmpty {
                    ExperimentInfoPanel(statusText: statusText)
                        .transition(.opacity)
                } else {
                    ExperimentOutputPanel(output: session.observations.output)
                        .transition(.opacity)
                }

                if hasActivity {
                    ExperimentStatusPanel(
                        state: session.state,
                        progress: session.observations.progress,
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

    private var bottomControls: some View {
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
                            Label("Placeholder Observations", systemImage: "chart.bar")
                        }

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
        .padding(.vertical, 8)
    }

    private func performPrimaryAction() {
        switch session.state {
        case .unloaded:
            Task { await session.prepare() }
        case .ready:
            Task { await session.start() }
        case .running:
            session.stop()
        case .stopped, .failed:
            session.newSession()
        case .preparing:
            break
        }
    }

    private var primaryActionTitle: String {
        switch session.state {
        case .unloaded:
            "Prepare"
        case .preparing:
            "Preparing"
        case .ready:
            "Start"
        case .running:
            "Stop"
        case .stopped, .failed:
            "New Session"
        }
    }

    private var primaryActionIcon: String {
        switch session.state {
        case .unloaded:
            "arrow.clockwise"
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

    private var isTerminal: Bool {
        switch session.state {
        case .stopped, .failed:
            true
        case .unloaded, .preparing, .ready, .running:
            false
        }
    }
}

private struct ExperimentInfoPanel: View {
    let statusText: String

    var body: some View {
        VStack(spacing: 24) {
            Image(systemName: "waveform.and.mic")
                .font(.system(size: 64))
                .foregroundStyle(.blue)

            VStack(spacing: 12) {
                Text("Experiment Session")
                    .font(.title)
                    .bold()

                Text(statusText)
                    .font(.headline)

                Text("Fake backend only. Real model inference is not wired in this slice.")
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
    let eventCount: Int
    let lastEventName: String
    let statusText: String

    var body: some View {
        VStack(spacing: 8) {
            if state == .preparing {
                ProgressView(value: progress)
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
                    Text("Script")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Gauge(value: Double(eventCount), in: 0...5) {
                        EmptyView()
                    } currentValueLabel: {
                        Text("\(eventCount) events")
                    }
                    .tint(.blue)
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
            Text("Placeholder Observations")
                .font(.headline)

            Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 8) {
                GridRow {
                    Text("Progress")
                        .foregroundStyle(.secondary)
                    Text(observations.progress.formatted(.percent.precision(.fractionLength(0))))
                }

                GridRow {
                    Text("Events")
                        .foregroundStyle(.secondary)
                    Text("\(observations.eventCount)")
                }

                GridRow {
                    Text("Last")
                        .foregroundStyle(.secondary)
                    Text(observations.lastEventName)
                }
            }
            .font(.system(.body, design: .monospaced))

            Text("No model latency, memory, frame cadence, token count, audio chunks, or translation quality is measured yet.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 8).fill(.secondary.opacity(0.1)))
    }
}

#Preview {
    ContentView()
}

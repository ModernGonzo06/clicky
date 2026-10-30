//
//  SystemTTSClient.swift
//  leanring-buddy
//
//  Text-to-speech client backed by the macOS system voice
//  (NSSpeechSynthesizer). Mirrors the `ElevenLabsTTSClient` interface so the
//  companion pipeline can fall back to it when the Cloudflare Worker — and
//  therefore ElevenLabs — is unreachable, e.g. in a fully-local setup that
//  uses a local LLM backend with no deployed worker at all.
//

import AppKit
import Foundation

@MainActor
final class SystemTTSClient {
    /// The active system-voice synthesizer. Retained for the duration of
    /// playback so the audio finishes even if the caller drops the client.
    private var speechSynthesizer: NSSpeechSynthesizer?

    /// Speaks `text` in the macOS system voice. Never throws: the system
    /// synthesizer has no network dependency.
    func speakText(_ text: String) {
        stopPlayback()

        let synthesizer = NSSpeechSynthesizer()
        synthesizer.startSpeaking(text)
        speechSynthesizer = synthesizer
        print("🔊 System TTS: speaking reply in the system voice")
    }

    /// Whether the system voice is currently speaking.
    var isSpeaking: Bool {
        speechSynthesizer?.isSpeaking ?? false
    }

    /// Stops any in-progress speech immediately.
    func stopPlayback() {
        speechSynthesizer?.stopSpeaking()
        speechSynthesizer = nil
    }
}

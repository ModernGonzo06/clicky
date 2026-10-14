//
//  BuddyChatProvider.swift
//  leanring-buddy
//
//  Shared protocol surface for the "brain" behind the companion — anything that
//  can take a transcript + screenshot images and stream back a spoken reply.
//
//  Mirrors the `BuddyTranscriptionProvider` protocol on the input side, so the
//  companion pipeline can swap the underlying brain without touching the call
//  sites. Today two implementations exist:
//    - `ClaudeAPI`         (cloud, via the Cloudflare Worker proxy)
//    - `LocalLLMProvider`  (local model server: Ollama, llama.cpp, LM Studio, ...)
//

import Foundation

/// A chat "brain" that can analyze one or more labeled screenshots and stream
/// back a text reply. Implementations may be cloud- or local-backed.
protocol BuddyChatProvider: AnyObject {
    /// Human-friendly label shown in the menu bar panel (e.g. "Claude", "Local · qwen2.5vl").
    var displayName: String { get }

    /// Whether this provider's wire format can carry image inputs. Both
    /// Ollama and OpenAI-compatible formats support vision; whether images are
    /// actually *sent* is the caller's choice (pass an empty `images` array).
    var supportsVision: Bool { get }

    /// Send a vision request and stream the reply back via `onTextChunk`.
    ///
    /// - `images`: zero or more captured screenshots, each with a human label
    ///   describing which display it came from and its pixel dimensions.
    /// - `systemPrompt`: the behavioral system prompt (defines Clicky's voice).
    /// - `conversationHistory`: prior (user, assistant) exchanges, oldest first.
    /// - `userPrompt`: the user's latest transcript.
    /// - `onTextChunk`: called on the main actor with the accumulated text so
    ///   far each time new text arrives.
    ///
    /// Returns the full accumulated text and the total request duration.
    func analyzeImageStreaming(
        images: [(data: Data, label: String)],
        systemPrompt: String,
        conversationHistory: [(userPlaceholder: String, assistantResponse: String)],
        userPrompt: String,
        onTextChunk: @MainActor @Sendable (String) -> Void
    ) async throws -> (text: String, duration: TimeInterval)
}

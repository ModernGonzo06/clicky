//
//  LocalLLMChatProvider.swift
//  leanring-buddy
//
//  Chat provider that talks to a locally-running LLM server instead of the
//  cloud. Supports two wire formats and auto-detects which one to speak from
//  the base URL:
//
//    - Ollama native        -> POST {base}/api/chat
//        Streams newline-delimited JSON objects; the final one has "done": true.
//        Images go in the user message under an "images" array of base64 strings.
//
//    - OpenAI-compatible     -> POST {base}/v1/chat/completions
//        (LM Studio, llama.cpp/llama-server, vLLM, Ollama's /v1 alias, ...)
//        Streams Server-Sent Events ("data: {json}" lines ending in "[DONE]").
//        Images go in the user message as {type:"image_url"} content blocks.
//
//  Because everything runs on localhost there is no TLS warmup and no proxy.
//  Vision works with either format; whether images are actually sent is the
//  caller's decision via an empty `images` array.
//

import Foundation

/// Which local-server API dialect to speak. Derived from the base URL.
enum LocalChatWireFormat {
    /// Ollama's native `/api/chat` endpoint.
    case ollamaNative
    /// OpenAI's `/v1/chat/completions` endpoint.
    case openAICompatible

    /// Auto-detect the dialect from the configured base URL:
    ///   - any URL containing "/v1" or "chat/completions" -> OpenAI-compatible
    ///   - everything else (e.g. http://localhost:11434)   -> Ollama native
    static func detect(from baseURL: String) -> LocalChatWireFormat {
        let lowercased = baseURL.lowercased()
        if lowercased.contains("chat/completions")
            || lowercased.contains("/v1")
            || lowercased.hasSuffix("/v1") {
            return .openAICompatible
        }
        return .ollamaNative
    }
}

/// Errors surfaced when talking to a local LLM server. Kept small and
/// user-friendly; the companion's `speakCreditsErrorFallback` plays a
/// hardcoded line when any of these throw mid-interaction.
enum LocalLLMProviderError: LocalizedError {
    case invalidEndpoint
    case invalidResponse
    case http(statusCode: Int, body: String)

    var errorDescription: String? {
        switch self {
        case .invalidEndpoint:
            return "Check the local server base URL in settings."
        case .invalidResponse:
            return "The local server returned an unexpected response."
        case .http(let statusCode, let body):
            return "Local server error (\(statusCode)): \(body)"
        }
    }
}

/// One model a local server reports as available, enriched with the metadata
/// Clicky's panel shows: whether it can see images, its size, and its runner.
struct LocalModelDetail: Equatable, Hashable {
    /// Full name as the server reports it (e.g. "qwen3.8:27b-mlx", "llava:latest").
    let name: String

    /// Size in bytes. 0 when the server doesn't report it (OpenAI-compatible servers usually don't).
    let sizeBytes: Int

    /// Capabilities the server advertises for this model (Ollama: "vision",
    /// "thinking", "tools", ...). Empty for OpenAI-compatible servers, which
    /// don't report capabilities.
    let capabilities: [String]

    /// Inference runner, e.g. "mlx" or "cuda" (Ollama only).
    let runner: String

    /// Quantization level, e.g. "q4_0" or "nvfp4" (Ollama only).
    let quantization: String

    var isVisionCapable: Bool { capabilities.contains("vision") }
    var isThinkingCapable: Bool { capabilities.contains("thinking") }

    /// Human-readable size; "—" when unknown.
    var sizeLabel: String {
        let bytes = Double(sizeBytes)
        if bytes >= 1_000_000_000 {
            return String(format: "%.1f GB", bytes / 1_000_000_000)
        }
        if bytes >= 1_000_000 {
            return String(format: "%.1f MB", bytes / 1_000_000)
        }
        return bytes > 0 ? String(format: "%.1f KB", bytes / 1_000) : "—"
    }

    /// A compact, panel-friendly descriptor, e.g. "vision · nvfp4 · mlx".
    var descriptor: String {
        var parts: [String] = []
        parts.append(isVisionCapable ? "vision" : "text")
        if !quantization.isEmpty { parts.append(quantization) }
        if !runner.isEmpty { parts.append(runner) }
        if sizeLabel != "—" { parts.append(sizeLabel) }
        return parts.joined(separator: " · ")
    }
}

/// The result of a "Refresh" connection probe: whether the server answered, the
/// models it reported, and whether the configured model is among them.
struct LocalModelCatalog: Equatable {
    var connected: Bool
    var models: [LocalModelDetail]
    var configuredModelAvailable: Bool
    var configuredModelDetail: LocalModelDetail?
    var reason: String?

    static var empty: LocalModelCatalog {
        LocalModelCatalog(connected: false, models: [], configuredModelAvailable: false,
                          configuredModelDetail: nil, reason: nil)
    }

    static func error(reason: String) -> LocalModelCatalog {
        LocalModelCatalog(connected: false, models: [], configuredModelAvailable: false,
                          configuredModelDetail: nil, reason: reason)
    }
}

/// Local LLM chat provider conforming to `BuddyChatProvider`. Talks to either
/// Ollama (native) or an OpenAI-compatible server, auto-detected from `baseURL`.
final class LocalLLMProvider: BuddyChatProvider {

    /// Label shown in the menu bar panel. Includes the model name so the user
    /// can tell which local model is active at a glance.
    var displayName: String { "Local · \(model)" }

    /// Both supported wire formats can carry images; whether they are sent is
    /// the caller's choice.
    let supportsVision = true

    /// Whether Ollama's "thinking" phase should be allowed. Thinking-capable
    /// models (qwen3.8, ...) stream a private "thinking" phase before the
    /// answer, adding a second or more of latency to every spoken turn with no
    /// benefit for a short spoken reply — so this defaults off. Exposed as a
    /// property so a future UI toggle is trivial.
    var enableThinking = false

    /// Base URL of the local server, e.g. http://localhost:11434 (Ollama) or
    /// http://localhost:1234 (LM Studio). Mutating it re-derives `wireFormat`.
    var baseURL: String {
        didSet { recomputeWireFormat() }
    }

    /// Model id to request, e.g. "llava", "qwen2.5vl:7b", "llama3.2:3b".
    var model: String

    /// Which API dialect to speak, derived from `baseURL`.
    private(set) var wireFormat: LocalChatWireFormat

    private let session: URLSession

    init(baseURL: String, model: String) {
        self.baseURL = baseURL
        self.model = model
        self.wireFormat = LocalChatWireFormat.detect(from: baseURL)

        // localhost: .default config is fine. No cookie/url cache needed and
        // there's no cold-TLS problem the way there is for the cloud proxy.
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 300
        config.timeoutIntervalForResource = 600
        config.waitsForConnectivity = true
        config.urlCache = nil
        config.httpCookieStorage = nil
        self.session = URLSession(configuration: config)
    }

    private func recomputeWireFormat() {
        wireFormat = LocalChatWireFormat.detect(from: baseURL)
    }

    // MARK: - URL helpers

    /// The configured base URL with any trailing slash stripped, so joining an
    /// endpoint onto it is predictable.
    private var normalizedBaseURL: String {
        var result = baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        while result.hasSuffix("/") {
            result = String(result.dropLast())
        }
        return result
    }

    /// The full chat-completion endpoint URL for the current wire format.
    private var chatEndpointURL: URL? {
        let raw = normalizedBaseURL
        let base = URL(string: raw)

        // If the user pasted a fully-qualified endpoint, respect it as-is.
        if raw.contains("/api/chat") || raw.contains("/chat/completions") {
            return URL(string: raw)
        }

        if wireFormat == .ollamaNative {
            return base?.appendingPathComponent("api/chat")
        } else {
            // OpenAI-compatible.
            if raw.hasSuffix("/v1") {
                return base?.appendingPathComponent("chat/completions")
            }
            return base?.appendingPathComponent("v1/chat/completions")
        }
    }

    /// Endpoint used by the panel's "Test" button. It serves two purposes:
    /// checking that the server is reachable AND confirming that the
    /// configured model is actually available. Ollama lists its pulled models
    /// at `/api/tags`; OpenAI-compatible servers list them at `/v1/models`.
    private var modelListURL: URL? {
        let raw = normalizedBaseURL
        let base = URL(string: raw)

        if wireFormat == .ollamaNative {
            return base?.appendingPathComponent("api/tags")
        } else {
            if raw.hasSuffix("/v1") {
                return base?.appendingPathComponent("models")
            }
            return base?.appendingPathComponent("v1/models")
        }
    }

    // MARK: - BuddyChatProvider

    func analyzeImageStreaming(
        images: [(data: Data, label: String)],
        systemPrompt: String,
        conversationHistory: [(userPlaceholder: String, assistantResponse: String)],
        userPrompt: String,
        onTextChunk: @MainActor @Sendable (String) -> Void
    ) async throws -> (text: String, duration: TimeInterval) {
        let startTime = Date()

        guard let request = makeRequest(
            images: images,
            systemPrompt: systemPrompt,
            conversationHistory: conversationHistory,
            userPrompt: userPrompt
        ) else {
            throw LocalLLMProviderError.invalidEndpoint
        }

        let (byteStream, response) = try await session.bytes(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw LocalLLMProviderError.invalidResponse
        }

        // On a non-2xx, drain the body so the user sees the real error message.
        if !(200...299).contains(httpResponse.statusCode) {
            var errorLines: [String] = []
            for try await line in byteStream.lines {
                errorLines.append(line)
            }
            throw LocalLLMProviderError.http(
                statusCode: httpResponse.statusCode,
                body: errorLines.joined(separator: "\n")
            )
        }

        var accumulated = ""

        for try await line in byteStream.lines {
            var producedChunk = false
            var shouldStop = false

            switch wireFormat {
            case .ollamaNative:
                (producedChunk, shouldStop) = parseOllamaLine(line, &accumulated)
            case .openAICompatible:
                (producedChunk, shouldStop) = parseOpenAICompatibleLine(line, &accumulated)
            }

            if producedChunk {
                let snapshot = accumulated
                await onTextChunk(snapshot)
            }
            if shouldStop {
                break
            }
        }

        return (text: accumulated, duration: Date().timeIntervalSince(startTime))
    }

    // MARK: - Request building

    private func makeRequest(
        images: [(data: Data, label: String)],
        systemPrompt: String,
        conversationHistory: [(userPlaceholder: String, assistantResponse: String)],
        userPrompt: String
    ) -> URLRequest? {
        guard let url = chatEndpointURL else {
            return nil
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 300
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body = buildBody(
            images: images,
            systemPrompt: systemPrompt,
            conversationHistory: conversationHistory,
            userPrompt: userPrompt
        )
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        return request
    }

    /// Build the JSON request body for the active wire format.
    ///
    /// Both formats share the message shape (system first, then history pairs,
    /// then the current user turn), but the image encoding differs:
    ///   - Ollama: a flat "images" array of base64 strings on the user message.
    ///   - OpenAI: {type:"image_url"} content blocks interleaved with text.
    private func buildBody(
        images: [(data: Data, label: String)],
        systemPrompt: String,
        conversationHistory: [(userPlaceholder: String, assistantResponse: String)],
        userPrompt: String
    ) -> [String: Any] {
        var messages: [[String: Any]] = []

        if !systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            messages.append(["role": "system", "content": systemPrompt])
        }

        for (userText, assistantText) in conversationHistory {
            messages.append(["role": "user", "content": userText])
            messages.append(["role": "assistant", "content": assistantText])
        }

        switch wireFormat {
        case .openAICompatible:
            var contentBlocks: [[String: Any]] = []
            for image in images {
                contentBlocks.append([
                    "type": "text",
                    "text": image.label
                ])
                contentBlocks.append([
                    "type": "image_url",
                    "image_url": [
                        "url": "data:image/jpeg;base64,\(image.data.base64EncodedString())"
                    ]
                ])
            }
            contentBlocks.append([
                "type": "text",
                "text": userPrompt
            ])
            messages.append([
                "role": "user",
                "content": contentBlocks
            ])

        case .ollamaNative:
            var userMessage: [String: Any] = [
                "role": "user",
                "content": userPrompt
            ]
            let base64Images = images.map { $0.data.base64EncodedString() }
            if !base64Images.isEmpty {
                userMessage["images"] = base64Images
            }
            messages.append(userMessage)
        }

        // No max_tokens/max_completion_tokens: local servers have widely
        // differing defaults and may reject unknown keys. Let the server decide.
        var body: [String: Any] = [
            "model": model,
            "stream": true,
            "messages": messages
        ]

        // Thinking-capable Ollama models stream a private "thinking" phase
        // before the answer, adding latency to every spoken turn. Disable it
        // for a snappier reply. Ollama-specific — OpenAI-compatible servers
        // don't understand this flag, so only send it for Ollama.
        if wireFormat == .ollamaNative {
            body["think"] = enableThinking
        }

        return body
    }

    // MARK: - Stream parsing

    /// Parse one Ollama native stream line. Each line is a JSON object of the
    /// shape `{"message": {"content": "..."}, "done": true/false}`.
    /// Returns whether a text chunk was produced and whether to stop streaming.
    private func parseOllamaLine(
        _ line: String,
        _ accumulated: inout String
    ) -> (producedChunk: Bool, shouldStop: Bool) {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              let data = trimmed.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return (false, false)
        }

        var producedChunk = false
        if let message = object["message"] as? [String: Any],
           let content = message["content"] as? String,
           !content.isEmpty {
            accumulated += content
            producedChunk = true
        }

        let shouldStop = (object["done"] as? Bool) ?? false
        return (producedChunk, shouldStop)
    }

    /// Parse one OpenAI-compatible SSE line. Lines look like `data: {json}` and
    /// end with `data: [DONE]`. Returns whether a chunk was produced and whether
    /// to stop.
    private func parseOpenAICompatibleLine(
        _ line: String,
        _ accumulated: inout String
    ) -> (producedChunk: Bool, shouldStop: Bool) {
        guard line.hasPrefix("data:") else {
            return (false, false)
        }

        let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
        guard !payload.isEmpty else {
            return (false, false)
        }

        if payload == "[DONE]" {
            return (false, true)
        }

        guard let data = payload.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = object["choices"] as? [[String: Any]],
              let first = choices.first,
              let delta = first["delta"] as? [String: Any],
              let content = delta["content"] as? String,
              !content.isEmpty
        else {
            return (false, false)
        }

        accumulated += content
        return (true, false)
    }

    // MARK: - Reachability

    /// Connectivity + model probe for the panel's "Test" button. Returns whether
    /// the local server answered with a 2xx AND the configured model is
    /// present in its model list, plus a human reason on failure.
    func checkReachable() async -> (reachable: Bool, reason: String?) {
        guard let url = modelListURL else {
            return (false, "Invalid base URL")
        }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = 12

        do {
            let (data, response) = try await session.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse else {
                return (false, "Unexpected response")
            }
            guard (200...299).contains(httpResponse.statusCode) else {
                return (false, "HTTP \(httpResponse.statusCode)")
            }

            // Confirm the configured model is actually available. The server is
            // considered "reachable" for Clicky's purposes only when it can
            // serve the requested model.
            let availableModels = parseAvailableModelNames(from: data)
            guard modelExists(configuredModel: model, in: availableModels) else {
                if availableModels.isEmpty {
                    return (false, "Model '\(model)' not available (server returned no models)")
                }
                let preview = availableModels.prefix(6).joined(separator: ", ")
                return (false, "Model '\(model)' not available. Found: \(preview)")
            }

            return (true, nil)
        } catch {
            return (false, error.localizedDescription)
        }
    }

    // MARK: - Model list helpers

    /// Parse the model names out of a server's model-list response.
    /// - Ollama `/api/tags`:  { "models": [ { "name": "llava:latest", ... } ] }
    /// - OpenAI `/v1/models`:  { "data":   [ { "id": "gpt-4o", ... } ] }
    private func parseAvailableModelNames(from data: Data) -> [String] {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return []
        }

        if wireFormat == .ollamaNative {
            guard let models = object["models"] as? [[String: Any]] else { return [] }
            return models.compactMap { entry in
                if let name = entry["name"] as? String { return name }
                if let name = entry["model"] as? String { return name }
                return nil
            }
        } else {
            guard let models = object["data"] as? [[String: Any]] else { return [] }
            return models.compactMap { $0["id"] as? String }
        }
    }

    /// Whether the configured model matches an entry returned by the server.
    /// Ollama names usually include a tag (`llava:latest`), while users often
    /// type the bare name (`llava`), so we compare the name before the first
    /// colon in both directions to be forgiving.
    private func modelExists(configuredModel: String, in availableModels: [String]) -> Bool {
        let configured = configuredModel.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !configured.isEmpty else { return false }

        let configuredBase = configured.split(separator: ":", maxSplits: 1).first.map(String.init) ?? configured

        return availableModels.contains { available in
            let availableTrimmed = available.trimmingCharacters(in: .whitespacesAndNewlines)
            if availableTrimmed == configured { return true }
            let availableBase = availableTrimmed.split(separator: ":", maxSplits: 1).first.map(String.init) ?? availableTrimmed
            return availableBase == configuredBase
        }
    }

    // MARK: - Model discovery

    /// Probe the server's model list (Ollama `/api/tags`, OpenAI-compatible
    /// `/v1/models`) and return a catalog: reachability, the available models
    /// with their capabilities, and whether the configured model is among them.
    /// This is the richer "Refresh" path the panel displays — it answers "what
    /// model is running and can it see my screen?" in one call.
    func discoverModels() async -> LocalModelCatalog {
        guard let url = modelListURL else {
            return .error(reason: "Invalid base URL")
        }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = 12

        do {
            let (data, response) = try await session.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse else {
                return .error(reason: "Unexpected response")
            }
            guard (200...299).contains(httpResponse.statusCode) else {
                return .error(reason: "HTTP \(httpResponse.statusCode)")
            }

            let details = parseModelCatalog(from: data)
            let matched = matchModelDetail(for: model, in: details)
            return LocalModelCatalog(
                connected: true,
                models: details,
                configuredModelAvailable: matched != nil,
                configuredModelDetail: matched,
                reason: nil
            )
        } catch {
            return .error(reason: error.localizedDescription)
        }
    }

    /// Parse a server's model-list response into enriched details.
    /// - Ollama `/api/tags`:    { "models": [ { "name","size","capabilities":[...],"details":{...} } ] }
    /// - OpenAI `/v1/models`:    { "data":   [ { "id" } ] } (no capabilities reported)
    private func parseModelCatalog(from data: Data) -> [LocalModelDetail] {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return []
        }

        if wireFormat == .ollamaNative {
            guard let models = object["models"] as? [[String: Any]] else { return [] }
            let details = models.compactMap { entry -> LocalModelDetail? in
                guard let name = (entry["name"] as? String) ?? (entry["model"] as? String),
                      !name.isEmpty
                else {
                    return nil
                }
                let size = (entry["size"] as? Int) ?? 0
                let capabilities = (entry["capabilities"] as? [String]) ?? []
                let details = entry["details"] as? [String: Any]
                let runner = (details?["runner"] as? String) ?? ""
                let quantization = (details?["quantization_level"] as? String)
                    ?? (details?["quantization"] as? String) ?? ""
                return LocalModelDetail(
                    name: name,
                    sizeBytes: size,
                    capabilities: capabilities,
                    runner: runner,
                    quantization: quantization
                )
            }
            return deduplicateModels(details)
        } else {
            guard let models = object["data"] as? [[String: Any]] else { return [] }
            let details = models.compactMap { entry -> LocalModelDetail? in
                guard let id = entry["id"] as? String, !id.isEmpty else {
                    return nil
                }
                return LocalModelDetail(
                    name: id,
                    sizeBytes: 0,
                    capabilities: [],
                    runner: "",
                    quantization: ""
                )
            }
            return deduplicateModels(details)
        }
    }

    /// Drop duplicate model names, keeping the first occurrence.
    private func deduplicateModels(_ models: [LocalModelDetail]) -> [LocalModelDetail] {
        var seen = Set<String>()
        var result: [LocalModelDetail] = []
        for model in models {
            if seen.insert(model.name).inserted {
                result.append(model)
            }
        }
        return result
    }

    /// Find the catalog entry matching the configured model, tolerating the
    /// bare-vs-tagged mismatch users hit (typing "qwen3.8" vs "qwen3.8:27b-mlx").
    private func matchModelDetail(
        for configuredModel: String,
        in details: [LocalModelDetail]
    ) -> LocalModelDetail? {
        let configured = configuredModel.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !configured.isEmpty else {
            return nil
        }
        let configuredBase = configured.split(separator: ":", maxSplits: 1).first.map(String.init) ?? configured

        if let exact = details.first(where: { $0.name == configured }) {
            return exact
        }
        return details.first { detail in
            let base = detail.name.split(separator: ":", maxSplits: 1).first.map(String.init) ?? detail.name
            return base == configuredBase
        }
    }
}

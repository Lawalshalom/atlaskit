/**
 * AtlasKit JavaScript SDK — zero dependencies, works in Node 18+, Deno, Bun and browsers.
 *
 *   import { AtlasKit } from "atlaskit";
 *   const atlas = new AtlasKit({ baseUrl: "https://<pod>-8080.proxy.runpod.net", apiKey: "..." });
 *   const { text } = await atlas.chat("Ẹ kú àárọ̀", { language: "yo" });
 */

export const LANGUAGES = Object.freeze({ en: "Nigerian English", yo: "Yoruba", ha: "Hausa", ig: "Igbo" });

export class AtlasKitError extends Error {
  /** @param {string} message @param {number} [status] @param {string} [code] */
  constructor(message, status, code) {
    super(message);
    this.name = "AtlasKitError";
    this.status = status;
    this.code = code;
  }
}

/**
 * @typedef {{role: "system"|"user"|"assistant", content: string}} Message
 * @typedef {{language?: string, system?: string, temperature?: number, maxTokens?: number, signal?: AbortSignal, [k: string]: any}} ChatOptions
 * @typedef {{text: string, finishReason: string|null, usage: object, raw: object}} ChatResponse
 */

export class AtlasKit {
  /**
   * @param {{baseUrl?: string, apiKey?: string, timeoutMs?: number, maxRetries?: number, fetch?: typeof fetch}} [opts]
   */
  constructor(opts = {}) {
    const env = typeof process !== "undefined" && process.env ? process.env : {};
    this.baseUrl = (opts.baseUrl || env.ATLASKIT_BASE_URL || "http://localhost:8080").replace(/\/+$/, "").replace(/\/v1$/, "");
    this.apiKey = opts.apiKey ?? env.ATLASKIT_API_KEY;
    this.timeoutMs = opts.timeoutMs ?? 120_000;
    this.maxRetries = opts.maxRetries ?? 2;
    this._fetch = opts.fetch || globalThis.fetch.bind(globalThis);
  }

  _headers(extra = {}) {
    const h = { Accept: "application/json", ...extra };
    if (this.apiKey) h.Authorization = `Bearer ${this.apiKey}`;
    return h;
  }

  async _request(method, path, { body, headers, signal } = {}) {
    let attempt = 0;
    for (;;) {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(new Error("timeout")), this.timeoutMs);
      if (signal) signal.addEventListener("abort", () => ctrl.abort(signal.reason), { once: true });
      let res;
      try {
        res = await this._fetch(`${this.baseUrl}${path}`, { method, body, headers: this._headers(headers), signal: ctrl.signal });
      } catch (err) {
        clearTimeout(timer);
        if (signal?.aborted) throw err;
        if (attempt++ < this.maxRetries) { await sleep(500 * 2 ** attempt); continue; }
        throw new AtlasKitError(`Cannot reach AtlasKit gateway at ${this.baseUrl}: ${err.message}`);
      }
      clearTimeout(timer);
      if (res.ok) return res;
      if ([502, 503, 504].includes(res.status) && attempt++ < this.maxRetries) { await sleep(500 * 2 ** attempt); continue; }
      let message = await res.text(), code;
      try { const e = JSON.parse(message).error; message = e.message; code = e.code; } catch { /* plain text */ }
      throw new AtlasKitError(`HTTP ${res.status}: ${message}`, res.status, code);
    }
  }

  async _json(method, path, payload, signal) {
    const res = await this._request(method, path, {
      body: payload === undefined ? undefined : JSON.stringify(payload),
      headers: payload === undefined ? {} : { "Content-Type": "application/json" },
      signal,
    });
    return res.json();
  }

  static _chatBody(messages, opts = {}) {
    const { language, system, temperature, maxTokens, signal, ...rest } = opts;
    let msgs = typeof messages === "string" ? [{ role: "user", content: messages }] : [...messages];
    if (system) msgs = [{ role: "system", content: system }, ...msgs.filter((m) => m.role !== "system")];
    const body = { model: "n-atlas", messages: msgs, ...rest };
    if (language) body.atlas_language = language;
    if (temperature !== undefined) body.temperature = temperature;
    if (maxTokens !== undefined) body.max_tokens = maxTokens;
    return body;
  }

  /** @returns {Promise<object>} */
  health() { return this._json("GET", "/health").catch((e) => (e.status === 503 ? { status: "degraded" } : Promise.reject(e))); }

  /** @returns {Promise<object[]>} */
  async models() { return (await this._json("GET", "/v1/models")).data; }

  /**
   * @param {string|Message[]} messages
   * @param {ChatOptions} [opts]
   * @returns {Promise<ChatResponse>}
   */
  async chat(messages, opts = {}) {
    const raw = await this._json("POST", "/v1/chat/completions", AtlasKit._chatBody(messages, opts), opts.signal);
    const c = raw.choices[0];
    return { text: c.message.content, finishReason: c.finish_reason ?? null, usage: raw.usage || {}, raw };
  }

  /**
   * Stream text deltas.
   * @param {string|Message[]} messages
   * @param {ChatOptions} [opts]
   * @returns {AsyncGenerator<string>}
   */
  async *stream(messages, opts = {}) {
    const body = { ...AtlasKit._chatBody(messages, opts), stream: true };
    const res = await this._request("POST", "/v1/chat/completions", {
      body: JSON.stringify(body),
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      signal: opts.signal,
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 1);
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") return;
        let chunk;
        try { chunk = JSON.parse(payload); } catch { continue; }
        for (const choice of chunk.choices || []) {
          const delta = choice.delta?.content;
          if (delta) yield delta;
        }
      }
    }
  }

  /**
   * Speech to text with the official N-ATLaS ASR model.
   * @param {Blob|ArrayBuffer|Uint8Array} audio
   * @param {string} language en|yo|ha|ig
   * @param {{filename?: string, signal?: AbortSignal}} [opts]
   */
  transcribe(audio, language, opts = {}) {
    return this._multipart("/v1/audio/transcriptions", audio, { language }, opts);
  }

  /** Audio question in, text answer in the same language out. */
  voiceChat(audio, language, opts = {}) {
    return this._multipart("/v1/atlas/voice-chat", audio, { language, system: opts.system }, opts);
  }

  async _multipart(path, audio, fields, opts) {
    const form = new FormData();
    for (const [k, v] of Object.entries(fields)) if (v != null) form.append(k, v);
    const blob = audio instanceof Blob ? audio : new Blob([audio]);
    form.append("file", blob, opts.filename || audio.name || "audio.wav");
    const res = await this._request("POST", path, { body: form, signal: opts.signal });
    return res.json();
  }

  /** @returns {Promise<string>} */
  async translate(text, target, source, { model } = {}) {
    return (await this._json("POST", "/v1/atlas/translate", { text, target, source, model })).translation;
  }

  /** @returns {Promise<string>} */
  async summarize(text, { language, maxWords = 80, model } = {}) {
    return (await this._json("POST", "/v1/atlas/summarize", { text, language, max_words: maxWords, model })).summary;
  }

  detectLanguage(text) { return this._json("POST", "/v1/atlas/detect-language", { text }); }

  async feedback(rating, { language, tag = "" } = {}) {
    await this._json("POST", "/v1/atlas/feedback", { rating, language, tag });
  }

  stats() { return this._json("GET", "/v1/atlas/stats"); }

  conversation(opts = {}) { return new Conversation(this, opts); }
}

export class Conversation {
  /** @param {AtlasKit} client @param {{language?: string, system?: string}} [opts] */
  constructor(client, { language, system } = {}) {
    this.client = client;
    this.language = language;
    /** @type {Message[]} */
    this.messages = system ? [{ role: "system", content: system }] : [];
  }

  async say(text, opts = {}) {
    this.messages.push({ role: "user", content: text });
    const { text: reply } = await this.client.chat(this.messages, { language: this.language, ...opts });
    this.messages.push({ role: "assistant", content: reply });
    return reply;
  }

  async *stream(text, opts = {}) {
    this.messages.push({ role: "user", content: text });
    let reply = "";
    for await (const d of this.client.stream(this.messages, { language: this.language, ...opts })) {
      reply += d;
      yield d;
    }
    this.messages.push({ role: "assistant", content: reply });
  }

  reset() { this.messages = this.messages.filter((m) => m.role === "system"); }
}

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

export default AtlasKit;

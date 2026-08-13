import {ModelRequest, TypedModelRequest} from "@korabench/core";
import * as dns from "node:dns/promises";
import {createLogRetryHandler, RetryOptions, withRetry} from "../retry.js";
import {Model} from "./model.js";

const MAI_THINKING_SLUG = "maithinking";

// The model's context window is 262144 tokens total (input + output). Cap
// escalating retries well under that so max_new_tokens alone never exceeds
// it — a completion request over the limit is rejected outright (400) with
// no chance to recover, worse than the "ended before completing its
// reasoning" error it would otherwise retry past.
const MAX_NEW_TOKENS_CEILING = 200000;

export function isMaiThinkingSlug(slug: string): boolean {
  return slug === MAI_THINKING_SLUG;
}

interface MaiThinkingModelConfig {
  url: string;
  deploymentName: string;
  maxTokens?: number;
  temperature?: number;
  retry?: RetryOptions;
}

// ChatML turn: <|im_start|>{role}\n{content}<|im_end|>\n
function renderTurn(role: string, content: string): string {
  return `<|im_start|>${role}\n${content}<|im_end|>\n`;
}

function renderPrompt(messages: ModelRequest["messages"]): string {
  const rendered = messages
    .map(m => renderTurn(m.role, m.content))
    .join("");
  return `${rendered}<|im_start|>assistant\n`;
}

interface GenerateResponse {
  text: string;
  meta_info?: {finish_reason?: {type?: string}};
}

// The local DNS resolver (observed here: Tailscale MagicDNS) intermittently
// fails to resolve this internal hostname when many concurrent lookups fire
// at once — concurrent requests otherwise fail with ECONNREFUSED/"could not
// resolve host" before ever reaching the server, which looks identical to a
// server-side outage. Resolving the hostname once upfront and connecting to
// the IP directly (with the original hostname sent via the Host header)
// sidesteps the repeated-lookup bottleneck entirely.
interface ResolvedEndpoint {
  url: string;
  hostHeader: string;
}

// Cached by url so concurrent Model instances (kora creates one per test)
// share a single resolution instead of each racing their own DNS lookup —
// which is the exact failure mode this sidesteps.
const resolvedEndpoints = new Map<string, Promise<ResolvedEndpoint>>();

function resolveEndpointOnce(url: string): Promise<ResolvedEndpoint> {
  const cached = resolvedEndpoints.get(url);
  if (cached) return cached;

  const promise = (async () => {
    const parsed = new URL(url);
    const hostHeader = parsed.host;
    const {address} = await dns.lookup(parsed.hostname);
    parsed.hostname = address;
    return {url: parsed.toString(), hostHeader};
  })();

  resolvedEndpoints.set(url, promise);
  return promise;
}

// The server conditionally emits a reasoning trace: when present, the raw
// text (only visible with skip_special_tokens: false — the default silently
// strips these markers and merges reasoning into the answer) looks like
// `type=thought<|im_end|>{reasoning}<|im_end|>{answer}`. When the model
// answers directly with no reasoning, it's just `{answer}` (optionally with
// a trailing `<|im_end|>`). Passing stop: ["<|im_start|>"] prevents the
// model from hallucinating additional fake turns past its real answer,
// which it otherwise does indefinitely once the true answer is complete.
//
// Detecting a reasoning response by matching the exact literal "type=thought"
// prefix is fragile — it was observed to sometimes emit a garbled header
// (e.g. "=thought" instead of "type=thought"), which fell through to the
// "direct answer" branch and leaked the entire reasoning trace into the
// conversation as if it were the real answer. Match "=thought" appearing
// within the first ~10 characters instead — tolerant of the "type" prefix
// being garbled or dropped, while still requiring the "=thought" marker
// itself so a genuine answer that happens to start with the word "thought"
// (e.g. "I thought about this...") isn't misidentified as a reasoning header.
const THOUGHT_HEADER_PATTERN = /^.{0,10}=thought/i;

function extractFinalAnswer(rawText: string): string {
  const parts = rawText.split("<|im_end|>");
  const hasThoughtHeader = THOUGHT_HEADER_PATTERN.test(parts[0]!);

  if (!hasThoughtHeader) {
    // No reasoning header: a direct answer, optionally followed by a
    // trailing stop-marker delimiter (or hallucinated extra turns beyond
    // it, which are discarded).
    return parts[0]!.trim();
  }

  if (parts.length < 3) {
    // Reasoning header present but the response was cut off before its
    // closing delimiter (even with zero delimiters at all, i.e. cut off
    // before writing any delimiter) — the reasoning itself is incomplete,
    // so parts[0] is NOT a usable answer. Surface this as retryable rather
    // than returning the partial reasoning dump.
    throw new Error(
      `maithinking response ended before completing its reasoning (got ${parts.length} of 3 expected segments); ` +
        `increase maxTokens. Raw: ${rawText.slice(0, 200)}`
    );
  }

  // Completed reasoning trace: everything after the second delimiter is
  // the real answer.
  return parts.slice(2).join("<|im_end|>").trim();
}

export function createMaiThinkingModel(config: MaiThinkingModelConfig): Model {
  const retryOptions: RetryOptions = {
    maxRetries: 5,
    initialDelayMs: 1000,
    maxDelayMs: 60000,
    backoffMultiplier: 2,
    jitterFactor: 0.2,
    onRetry: createLogRetryHandler(MAI_THINKING_SLUG),
    ...config.retry,
  };

  async function callOnce(
    request: ModelRequest,
    maxNewTokens: number
  ): Promise<string> {
    const text = renderPrompt(request.messages);
    const temperature = request.temperature ?? config.temperature;
    const {url, hostHeader} = await resolveEndpointOnce(config.url);

    let r: Response;
    try {
      r = await fetch(url, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-deployment-name": config.deploymentName,
          host: hostHeader,
        },
        body: JSON.stringify({
          text,
          sampling_params: {
            max_new_tokens: maxNewTokens,
            temperature,
            stop: ["<|im_start|>"],
            skip_special_tokens: false,
          },
        }),
      });
    } catch (error) {
      // The cached IP may be stale (server restarted elsewhere, DHCP
      // change, etc.) — drop it so the next attempt re-resolves instead of
      // failing against a dead address forever.
      resolvedEndpoints.delete(config.url);
      throw error;
    }

    if (!r.ok) {
      throw new Error(
        `maithinking POST ${config.url} failed: ${r.status} ${await r.text()}`
      );
    }

    const data = (await r.json()) as GenerateResponse;
    return extractFinalAnswer(data.text);
  }

  async function generate(request: ModelRequest): Promise<string> {
    const baseMaxNewTokens = request.maxTokens ?? config.maxTokens ?? 16000;

    // A cut-off reasoning trace (extractFinalAnswer's error) will fail
    // identically on a verbatim retry — the model isn't non-deterministic
    // enough to reliably finish faster on attempt 2. So each retry doubles
    // the token budget instead of resending the same request.
    let attempt = 0;
    return withRetry(async () => {
      const maxNewTokens = Math.min(
        baseMaxNewTokens * 2 ** attempt,
        MAX_NEW_TOKENS_CEILING
      );
      attempt++;
      return callOnce(request, maxNewTokens);
    }, retryOptions);
  }

  return {
    getTextResponse: generate,

    async getStructuredResponse<T>(_request: TypedModelRequest<T>): Promise<T> {
      throw new Error(
        `maithinking does not support structured output (requested for a judge/user model). ` +
          `Use it only as the target model.`
      );
    },
  };
}

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

export interface MaiThinkingModelConfig {
  url: string;
  deploymentName: string;
  maxTokens?: number;
  temperature?: number;
  retry?: RetryOptions;
  /** Label used in retry/error messages. Defaults to the maithinking slug;
   * rc34 reuses this transport and overrides it so logs name the right slug. */
  label?: string;
}

// ChatML turn: <|im_start|>{role}\n{content}<|im_end|>\n
function renderTurn(role: string, content: string): string {
  return `<|im_start|>${role}\n${content}<|im_end|>\n`;
}

/** Render messages as a ChatML prompt primed for an assistant turn. Exported
 * so debug commands can print the exact string the gateway receives. */
export function renderChatMlPrompt(messages: ModelRequest["messages"]): string {
  const rendered = messages.map(m => renderTurn(m.role, m.content)).join("");
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
// conversation as if it were the real answer. Match "=thought" near the start
// instead — tolerant of the "type" prefix being garbled, dropped, or preceded
// by a special token, while still requiring the "=thought" marker itself so a
// genuine answer that happens to start with the word "thought" (e.g. "I
// thought about this...") isn't misidentified as a reasoning header.
//
// The window is 40 characters rather than ~10 because rc34 (same serving
// stack, different checkpoint) prefixes the header with a special token:
// "<|im_sep|>type=thought" is 22 characters, and at the old width it fell
// through to the direct-answer branch and leaked the whole trace.
const THOUGHT_HEADER_PATTERN = /^.{0,40}?=thought/i;

// rc34 sometimes finishes its answer and immediately begins ANOTHER thought
// block, which the stop token then truncates — leaving e.g.
// "...right here with you.<|im_sep|>type=thought" as the final segment. Without
// stripping it, that marker was returned as part of the answer and reached both
// the judge and the recorded transcript verbatim. Anchored to the END and
// requires the special-token form or the bare header, so an answer that merely
// ends in the word "thought" is untouched.
const TRAILING_THOUGHT_HEADER =
  /(?:<\|im_[a-z]+\|>\s*)+(?:[a-z]{0,6}=thought)?\s*$|^\s*(?:[a-z]{0,6}=thought)\s*$/i;

function stripTrailingThoughtHeader(text: string): string {
  return text.replace(TRAILING_THOUGHT_HEADER, "").trim();
}

export function extractFinalAnswer(rawText: string): string {
  const parts = rawText.split("<|im_end|>");
  const hasThoughtHeader = THOUGHT_HEADER_PATTERN.test(parts[0]!);

  if (!hasThoughtHeader) {
    // rc34 was observed to sometimes emit a reasoning trace with NO header at
    // all: two segments where the first is the trace and the second the
    // answer. Treating that as a direct answer leaks the whole trace into the
    // conversation. A genuine direct answer has ONE segment (plus any
    // hallucinated turns past the stop marker, which are discarded), so a
    // headerless two-segment response is a reasoning trace: take the last
    // segment as the answer.
    if (parts.length === 2 && parts[1]!.trim().length > 0) {
      return parts[1]!.trim();
    }

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
      `Reasoning-model response ended before completing its reasoning (got ${parts.length} of 3 expected segments); ` +
        `increase maxTokens. Raw: ${rawText.slice(0, 200)}`
    );
  }

  // Completed reasoning trace. The canonical shape is exactly three segments
  // (header / reasoning / answer), but rc34 was observed to emit a doubled
  // delimiter after the header, giving four segments with an empty second one:
  //   ["type=thought", "", "<reasoning>", "<answer>"]
  // Slicing from a fixed index then returns reasoning+answer joined, leaking
  // the trace. The answer is always the LAST non-empty segment, so take that.
  const lastNonEmpty = parts
    .map(part => stripTrailingThoughtHeader(part.trim()))
    .filter(part => part.length > 0)
    .at(-1);
  if (lastNonEmpty === undefined) {
    throw new Error(
      `Reasoning-model response contained no answer segment. Raw: ${rawText.slice(0, 200)}`
    );
  }
  return lastNonEmpty;
}

export function createMaiThinkingModel(config: MaiThinkingModelConfig): Model {
  const retryOptions: RetryOptions = {
    maxRetries: 5,
    initialDelayMs: 1000,
    maxDelayMs: 60000,
    backoffMultiplier: 2,
    jitterFactor: 0.2,
    onRetry: createLogRetryHandler(config.label ?? MAI_THINKING_SLUG),
    ...config.retry,
  };

  async function callOnce(
    request: ModelRequest,
    maxNewTokens: number
  ): Promise<string> {
    const text = renderChatMlPrompt(request.messages);
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
        `${config.label ?? MAI_THINKING_SLUG} POST ${config.url} failed: ${r.status} ${await r.text()}`
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

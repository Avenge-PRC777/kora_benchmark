/**
 * Retry utilities for handling transient API failures with exponential backoff.
 */

export interface RetryOptions {
  /** Maximum number of retry attempts. Default: 5 */
  maxRetries?: number;
  /** Initial delay in milliseconds. Default: 1000 */
  initialDelayMs?: number;
  /** Maximum delay in milliseconds. Default: 60000 */
  maxDelayMs?: number;
  /** Multiplier for exponential backoff. Default: 2 */
  backoffMultiplier?: number;
  /** Jitter factor (0-1) to randomize delays. Default: 0.2 */
  jitterFactor?: number;
  /** Called before each retry with attempt info */
  onRetry?: (attempt: number, error: Error, delayMs: number) => void;
}

const defaultOptions: Required<Omit<RetryOptions, "onRetry">> = {
  maxRetries: 5,
  initialDelayMs: 1000,
  maxDelayMs: 60000,
  backoffMultiplier: 2,
  jitterFactor: 0.2,
};

// A model that declines the task (common for gpt-5.2 on the sexual-content
// and self-harm risks) returns prose like "I'm sorry, but I cannot assist
// with that request." instead of JSON. That surfaces as a JSON parse error,
// which otherwise looks retryable — so the same request is re-sent five times
// with growing backoff (~35s) and refused identically each time, before the
// fallback chain finally advances to a model that will do the work.
//
// A refusal is a deterministic property of (model, prompt), not a transient
// fault: retrying it verbatim cannot succeed. Detect it and fail immediately
// so createFallbackModel fails over to the next model in the chain at once.
const REFUSAL_PATTERNS = [
  /\bi(?:'m| am) sorry\b/i,
  /\bi can(?:no|')t (?:assist|help|comply|continue|provide)\b/i,
  /\bi (?:cannot|can not|won't|will not) (?:assist|help|comply|continue|provide|create|generate)\b/i,
  /\bi'm (?:not able|unable) to\b/i,
  /\bunable to (?:assist|help|comply|provide)\b/i,
  /\bas an ai\b.{0,40}\bcan(?:no|')t\b/i,
];

/**
 * True when the error is a model declining the request rather than a
 * transient failure. Only meaningful for JSON-parse failures, where the
 * unparsed text is the model's prose refusal.
 */
export function isRefusalError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  // Only a parse failure carries refusal prose; a schema-validation error is
  // a real (retryable) structured response that just broke a constraint.
  const isParseFailure =
    error instanceof SyntaxError ||
    error.message.includes("is not valid JSON") ||
    error.message.toLowerCase().includes("failed to parse");
  if (!isParseFailure) return false;

  // JSON.parse quotes only ~10 characters of the offending text, so prefer
  // the full raw output attached at the throw site when it is available.
  const raw = (error as {rawOutput?: unknown}).rawOutput;
  const text = (typeof raw === "string" ? raw : error.message).trim();

  // A refusal is prose sent INSTEAD of JSON. Output that starts as a JSON
  // object is a genuine (if malformed) structured response — e.g. a raw
  // newline inside a string yields "Bad control character", and its content
  // may legitimately contain an apology ("...I am sorry I lied about lunch")
  // in a scenario narrative. Matching refusal patterns against that text
  // would misclassify a retryable truncation as a refusal and skip retries.
  if (text.startsWith("{") || text.startsWith("[")) return false;

  // Only the leading prose can be a refusal — a scenario that merely quotes
  // an apology deep in its body is not the model declining.
  return REFUSAL_PATTERNS.some(p => p.test(text.slice(0, 200)));
}

/**
 * Determines if an error is retryable based on its characteristics.
 */
function isRetryableError(error: unknown): boolean {
  // Checked before everything else: a refusal is deterministic, so retrying
  // the identical request just wastes backoff before the fallback chain
  // advances to a model that will answer.
  if (isRefusalError(error)) {
    return false;
  }

  if (!(error instanceof Error)) {
    return false;
  }

  // Node's fetch() wraps the real DNS/connection failure (ENOTFOUND,
  // ECONNREFUSED, etc.) in error.cause and only exposes a generic
  // "fetch failed" at the top level — check both so these are retried
  // instead of failing immediately with no retry at all.
  const cause = (error as {cause?: unknown}).cause;
  const causeMessage =
    cause instanceof Error ? cause.message.toLowerCase() : "";
  const causeCode =
    cause && typeof cause === "object" && "code" in cause
      ? String((cause as {code: unknown}).code).toLowerCase()
      : "";

  const message = error.message.toLowerCase() + " " + causeMessage;
  const name = error.name.toLowerCase();

  // DNS resolution failures
  if (
    message.includes("enotfound") ||
    message.includes("eai_again") ||
    causeCode.includes("enotfound") ||
    causeCode.includes("eai_again")
  ) {
    return true;
  }

  // Rate limiting
  if (message.includes("429") || message.includes("rate limit")) {
    return true;
  }

  // Server errors (5xx)
  if (
    message.includes("500") ||
    message.includes("502") ||
    message.includes("503") ||
    message.includes("504") ||
    message.includes("internal server error") ||
    message.includes("bad gateway") ||
    message.includes("service unavailable") ||
    message.includes("gateway timeout")
  ) {
    return true;
  }

  // Network/connection errors
  if (
    message.includes("econnreset") ||
    message.includes("econnrefused") ||
    message.includes("etimedout") ||
    message.includes("socket hang up") ||
    message.includes("network") ||
    message.includes("connection") ||
    message.includes("fetch failed") ||
    name.includes("fetch") ||
    name.includes("network")
  ) {
    return true;
  }

  // Timeout errors
  if (message.includes("timeout") || name.includes("timeout")) {
    return true;
  }

  // Overloaded API
  if (message.includes("overloaded") || message.includes("capacity")) {
    return true;
  }

  // Validation/parsing errors - model returned malformed output
  if (
    name.includes("valibot") ||
    message.includes("invalid") ||
    message.includes("expected") ||
    message.includes("parsing") ||
    message.includes("validation")
  ) {
    return true;
  }

  // AI SDK output parsing errors
  if (
    message.includes("failed to parse") ||
    message.includes("json") ||
    message.includes("output")
  ) {
    return true;
  }

  return false;
}

/**
 * Extracts retry-after delay from error if available.
 */
function getRetryAfterMs(error: unknown): number | undefined {
  if (!(error instanceof Error)) {
    return undefined;
  }

  // Check for retry-after in the error message or properties
  const errorAny = error as unknown as Record<string, unknown>;

  // Some APIs include headers in the error
  if (typeof errorAny.headers === "object" && errorAny.headers !== null) {
    const headers = errorAny.headers as Record<string, unknown>;
    const retryAfter = headers["retry-after"] ?? headers["Retry-After"];
    if (typeof retryAfter === "string") {
      const seconds = parseInt(retryAfter, 10);
      if (!isNaN(seconds)) {
        return seconds * 1000;
      }
    }
  }

  // Check for retryAfter property (some SDKs expose this directly)
  if (typeof errorAny.retryAfter === "number") {
    return errorAny.retryAfter * 1000;
  }

  return undefined;
}

/**
 * Calculates delay with exponential backoff and jitter.
 */
function calculateDelay(
  attempt: number,
  options: Required<Omit<RetryOptions, "onRetry">>,
  retryAfterMs?: number
): number {
  // Respect retry-after header if present
  if (retryAfterMs !== undefined) {
    return Math.min(retryAfterMs, options.maxDelayMs);
  }

  // Exponential backoff: initialDelay * (multiplier ^ attempt)
  const exponentialDelay =
    options.initialDelayMs * Math.pow(options.backoffMultiplier, attempt);

  // Apply jitter
  const jitter = 1 + (Math.random() * 2 - 1) * options.jitterFactor;
  const delayWithJitter = exponentialDelay * jitter;

  return Math.min(delayWithJitter, options.maxDelayMs);
}

/**
 * Sleeps for the specified duration.
 */
function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Executes an async function with retry logic and exponential backoff.
 *
 * @param fn - The async function to execute
 * @param options - Retry configuration options
 * @returns The result of the function
 * @throws The last error if all retries are exhausted
 */
export async function withRetry<T>(
  fn: () => Promise<T>,
  options?: RetryOptions
): Promise<T> {
  const opts = {...defaultOptions, ...options};
  let lastError: Error | undefined;

  for (let attempt = 0; attempt <= opts.maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      // Don't retry if this is not a retryable error
      if (!isRetryableError(error)) {
        throw lastError;
      }

      // Don't retry if we've exhausted all attempts
      if (attempt >= opts.maxRetries) {
        throw lastError;
      }

      const retryAfterMs = getRetryAfterMs(error);
      const delayMs = calculateDelay(attempt, opts, retryAfterMs);

      if (options?.onRetry) {
        options.onRetry(attempt + 1, lastError, delayMs);
      }

      await sleep(delayMs);
    }
  }

  // This should never be reached, but TypeScript needs it
  throw lastError ?? new Error("Retry failed with no error");
}

/**
 * Creates a default onRetry handler that logs to stderr.
 */
export function createLogRetryHandler(
  context?: string
): NonNullable<RetryOptions["onRetry"]> {
  return (attempt, error, delayMs) => {
    const prefix = context ? `[${context}] ` : "";
    const delaySeconds = (delayMs / 1000).toFixed(1);
    console.error(
      `${prefix}Retry ${attempt}: ${error.message}. Waiting ${delaySeconds}s...`
    );
  };
}

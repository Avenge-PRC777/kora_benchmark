import {createOpenAICompatible} from "@ai-sdk/openai-compatible";
import {ModelRequest, TypedModelRequest} from "@korabench/core";
import {toJsonSchema} from "@valibot/to-json-schema";
import {
  gateway,
  generateObject,
  generateText,
  jsonSchema,
  LanguageModel,
} from "ai";
import * as v from "valibot";
import {
  createLogRetryHandler,
  isRefusalError,
  RetryOptions,
  withRetry,
} from "../retry.js";
import {createFallbackModel, FallbackModel} from "./fallbackModel.js";
import {Model} from "./model.js";
import {resolveModelConfig} from "./modelConfig.js";

// Model provider used to serve gateway-routed slugs. Both the AI SDK Gateway
// and OpenRouter accept the same `provider/model` slug format (e.g.
// `openai/gpt-4o`), so `models.json` entries work unchanged either way.
// Precedence when both keys are set: AI_GATEWAY_API_KEY wins, since it's the
// documented default; set only OPENROUTER_API_KEY to use OpenRouter instead.
const openrouter = createOpenAICompatible({
  name: "openrouter",
  baseURL: "https://openrouter.ai/api/v1",
  apiKey: process.env.OPENROUTER_API_KEY,
});

function resolveLanguageModel(modelId: string): LanguageModel {
  if (process.env.AI_GATEWAY_API_KEY) return gateway(modelId);
  if (process.env.OPENROUTER_API_KEY) return openrouter(modelId);
  throw new Error(
    "No model provider configured. Set AI_GATEWAY_API_KEY or OPENROUTER_API_KEY."
  );
}

// OpenRouter's OpenAI-compatible endpoint requests `response_format:
// json_object` for generateObject, which upstream OpenAI/Azure reject unless
// the word "json" appears in the prompt — and even then, imposes constraints
// generateObject doesn't account for. Route structured responses through the
// text+extract fallback (below) for every model when running via OpenRouter,
// not just Google/Anthropic.
function usesTextExtractionFallback(modelId: string): boolean {
  if (!process.env.AI_GATEWAY_API_KEY && process.env.OPENROUTER_API_KEY) {
    return true;
  }
  return modelId.startsWith("google/") || modelId.startsWith("anthropic/");
}

export interface ModelOptions {
  retry?: RetryOptions;
}

const defaultRetryOptions: RetryOptions = {
  maxRetries: 5,
  initialDelayMs: 1000,
  maxDelayMs: 60000,
  backoffMultiplier: 2,
  jitterFactor: 0.2,
};

function buildRetryOptions(
  label: string,
  options?: ModelOptions
): Required<Pick<RetryOptions, "onRetry">> & RetryOptions {
  return {
    ...defaultRetryOptions,
    ...options?.retry,
    onRetry: options?.retry?.onRetry ?? createLogRetryHandler(label),
  };
}

// The Vercel AI Gateway corrupts structured-output responses for Anthropic
// (tool arguments dropped, returned as "{}") and for Google (thinking tags
// leak into text when structuredOutputs is off). Bypass generateObject for
// both providers and extract JSON from the plain-text response.
export function extractJson(text: string): string {
  const withoutThink = text.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
  const fenceMatch = withoutThink.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
  if (fenceMatch?.[1]) return fenceMatch[1].trim();

  // An UNTERMINATED fence (the response was cut off mid-object, so the closing
  // ``` never arrived) doesn't match above. Strip the opening fence so the
  // brace scan below sees the JSON rather than the leading backticks — without
  // this, `indexOf("{")` still finds the object start but the surrounding text
  // was returned verbatim whenever no closing brace existed either, and
  // JSON.parse failed on "```json" instead of on the truncation itself.
  const withoutOpeningFence = withoutThink.replace(/^```(?:json)?\s*/i, "");

  const start = withoutOpeningFence.indexOf("{");
  const end = withoutOpeningFence.lastIndexOf("}");
  if (start === -1 || end === -1 || end < start) return withoutOpeningFence;
  return withoutOpeningFence.slice(start, end + 1);
}

/**
 * Describe the request that `getTextResponse` would send for `modelSlug`,
 * without sending it. Mirrors the mapping done in `createGatewayModel` (system
 * message hoisted out of `messages`, defaults from `models.json` applied), so
 * `debug-turn` can print exactly what the provider receives.
 */
export function describeGatewayTextPayload(
  modelsJsonPath: string,
  modelSlug: string,
  request: ModelRequest
): Record<string, unknown> {
  const config = resolveModelConfig(modelsJsonPath, modelSlug);
  const provider = process.env.AI_GATEWAY_API_KEY
    ? "ai-gateway"
    : process.env.OPENROUTER_API_KEY
      ? "openrouter"
      : "unconfigured";

  return {
    provider,
    endpoint:
      provider === "openrouter"
        ? "https://openrouter.ai/api/v1/chat/completions"
        : provider === "ai-gateway"
          ? "(ai-sdk gateway)"
          : "(no provider key set)",
    slug: modelSlug,
    model: config.model,
    system: request.messages.find(m => m.role === "system")?.content,
    messages: request.messages
      .filter(m => m.role !== "system")
      .map(m => ({role: m.role, content: m.content})),
    maxOutputTokens: request.maxTokens ?? config.maxTokens,
    temperature: request.temperature ?? config.temperature,
    providerOptions: config.providerOptions,
  };
}

// A schema-validation failure (e.g. `Invalid length: Expected <=500 but
// received 530`) fails identically on a verbatim retry — the model has no way
// of knowing what was wrong. Feed the rejection back as an extra user turn so
// each attempt is a correction rather than a re-roll.
function validationFeedbackMessage(
  error: unknown,
  previousJson: string
): {role: "assistant" | "user"; content: string}[] {
  if (!(error instanceof v.ValiError)) return [];
  const issues = error.issues
    .map(issue => {
      const path =
        issue.path?.map((p: {key?: unknown}) => String(p.key)).join(".") ??
        "(root)";
      return `- ${path}: ${issue.message}`;
    })
    .join("\n");
  return [
    {role: "assistant", content: previousJson},
    {
      role: "user",
      content:
        "Your previous JSON was rejected by schema validation:\n" +
        issues +
        "\n\nRewrite the FULL JSON object, fixing exactly these problems and " +
        "keeping everything else the same. Length limits are hard caps measured " +
        "in characters — shorten the offending fields by trimming detail, not by " +
        "truncating mid-sentence.",
    },
  ];
}

/**
 * Attach the model's raw output to a parse/validation error so downstream
 * classification (notably `isRefusalError`) can inspect the actual text —
 * `JSON.parse` only quotes ~10 characters of it in its own message.
 * `onValidation` runs only for genuine schema failures, not refusals.
 */
function annotateRawOutput(
  error: unknown,
  rawText: string,
  onValidation: () => void
): unknown {
  if (error instanceof Error) {
    (error as {rawOutput?: string}).rawOutput = rawText;
  }
  // A refusal is prose, not a structured response — there is nothing to feed
  // back as a correction, and the request will fail over to the next model.
  if (!isRefusalError(error)) {
    onValidation();
  }
  return error;
}

export function createGatewayModelChain(
  modelsJsonPath: string,
  modelSlugs: readonly string[],
  options?: ModelOptions
): FallbackModel {
  if (modelSlugs.length === 0) {
    throw new Error("createGatewayModelChain: at least one slug required.");
  }
  return createFallbackModel(
    modelSlugs.map(slug => ({
      label: slug,
      model: createGatewayModel(modelsJsonPath, slug, options),
    }))
  );
}

export function createGatewayModel(
  modelsJsonPath: string,
  modelSlug: string,
  options?: ModelOptions
): Model {
  const config = resolveModelConfig(modelsJsonPath, modelSlug);
  const retryOptions = buildRetryOptions(config.model, options);

  return {
    async getTextResponse(request: ModelRequest): Promise<string> {
      const maxTokens = request.maxTokens ?? config.maxTokens;
      const temperature = request.temperature ?? config.temperature;

      const result = await withRetry(
        () =>
          generateText({
            model: resolveLanguageModel(config.model),
            system: request.messages.find(m => m.role === "system")?.content,
            messages: request.messages
              .filter(m => m.role !== "system")
              .map(m => ({
                role: m.role as "user" | "assistant",
                content: m.content,
              })),
            maxOutputTokens: maxTokens,
            temperature,
            providerOptions: config.providerOptions as
              | Record<string, Record<string, never>>
              | undefined,
            maxRetries: 0,
          }),
        retryOptions
      );

      return result.text;
    },

    async getStructuredResponse<T>(request: TypedModelRequest<T>): Promise<T> {
      const outputSchema = toJsonSchema(request.outputType);
      const maxTokens = request.maxTokens ?? config.maxTokens;
      const temperature = request.temperature ?? config.temperature;
      const systemMessage = request.messages.find(
        m => m.role === "system"
      )?.content;
      const userMessages = request.messages
        .filter(m => m.role !== "system")
        .map(m => ({
          role: m.role as "user" | "assistant",
          content: m.content,
        }));

      if (usesTextExtractionFallback(config.model)) {
        const schemaInstruction =
          "Respond with a single JSON object that strictly conforms to this JSON Schema. " +
          "Output JSON only — no prose, no code fences, no <think> tags.\n\n" +
          JSON.stringify(outputSchema);
        const combinedSystem = systemMessage
          ? `${systemMessage}\n\n${schemaInstruction}`
          : schemaInstruction;

        let correction: {role: "assistant" | "user"; content: string}[] = [];
        return withRetry(async () => {
          const result = await generateText({
            model: resolveLanguageModel(config.model),
            system: combinedSystem,
            messages: [...userMessages, ...correction],
            maxOutputTokens: maxTokens,
            temperature,
            providerOptions: config.providerOptions as any,
            maxRetries: 0,
          });

          const json = extractJson(result.text);
          try {
            return v.parse(request.outputType, JSON.parse(json));
          } catch (error) {
            // JSON.parse truncates the offending text to ~10 chars in its
            // message, which isn't enough to recognize a refusal. Attach the
            // full raw output so isRefusalError can inspect it.
            throw annotateRawOutput(error, result.text, () => {
              correction = validationFeedbackMessage(error, json);
            });
          }
        }, retryOptions);
      }

      let correction: {role: "assistant" | "user"; content: string}[] = [];
      return withRetry(async () => {
        const result = await generateObject({
          model: resolveLanguageModel(config.model),
          system: systemMessage,
          messages: [...userMessages, ...correction],
          schema: jsonSchema(outputSchema),
          maxOutputTokens: maxTokens,
          temperature,
          providerOptions: config.providerOptions as any,
          maxRetries: 0,
        });

        try {
          return v.parse(request.outputType, result.object);
        } catch (error) {
          correction = validationFeedbackMessage(
            error,
            JSON.stringify(result.object)
          );
          throw error;
        }
      }, retryOptions);
    },
  };
}

import {ModelRequest, TypedModelRequest} from "@korabench/core";
import {Model} from "./model.js";

interface LabeledModel {
  label: string;
  model: Model;
}

/**
 * A Model that also reports which member of its chain served the last call.
 * The chain label alone ("a|b") doesn't say whether the primary answered or a
 * fallback did, which matters when the fallback is a different judge and its
 * grades end up mixed into the results.
 */
export interface FallbackModel extends Model {
  /** Label of the model that served the most recent successful call. */
  lastUsedLabel(): string | undefined;
}

export function createFallbackModel(
  models: readonly LabeledModel[]
): FallbackModel {
  if (models.length === 0) {
    throw new Error("createFallbackModel: at least one model required.");
  }

  let lastUsed: string | undefined;

  const head = models[0]!;
  if (models.length === 1) {
    return {
      ...head.model,
      async getTextResponse(request) {
        const r = await head.model.getTextResponse(request);
        lastUsed = head.label;
        return r;
      },
      async getStructuredResponse(request) {
        const r = await head.model.getStructuredResponse(request);
        lastUsed = head.label;
        return r;
      },
      lastUsedLabel: () => lastUsed,
    };
  }

  async function tryChain<T>(
    method: "getTextResponse" | "getStructuredResponse",
    invoke: (m: Model) => Promise<T>
  ): Promise<T> {
    let lastError: unknown;
    for (let i = 0; i < models.length; i++) {
      const current = models[i]!;
      try {
        const result = await invoke(current.model);
        lastUsed = current.label;
        return result;
      } catch (error) {
        lastError = error;
        const message = error instanceof Error ? error.message : String(error);
        const next = models[i + 1];
        if (next) {
          console.error(
            `[fallback] ${method} on ${current.label} exhausted retries; failing over to ${next.label}: ${message.slice(0, 200)}`
          );
        } else {
          console.error(
            `[fallback] ${method} exhausted on final model ${current.label}: ${message.slice(0, 200)}`
          );
        }
      }
    }
    throw lastError;
  }

  return {
    getTextResponse(request: ModelRequest) {
      return tryChain("getTextResponse", m => m.getTextResponse(request));
    },
    getStructuredResponse<T>(request: TypedModelRequest<T>) {
      return tryChain("getStructuredResponse", m =>
        m.getStructuredResponse(request)
      );
    },
    lastUsedLabel: () => lastUsed,
  };
}

import {createMaiThinkingModel} from "./maiThinkingModel.js";
import {Model} from "./model.js";

const RC34_SLUG = "rc34";

export function isRc34Slug(slug: string): boolean {
  return slug === RC34_SLUG;
}

/**
 * Cluster -> gateway base URL, supplied by the operator.
 *
 * Internal gateway hostnames and deployment names are deliberately NOT
 * hardcoded here: they are environment-specific and not public. Configure
 * them via `RC34_CLUSTER_GATEWAYS` (a JSON object of cluster -> base URL) or
 * skip the mapping entirely and set `RC34_BASE_URL` directly. See
 * `.env.example`.
 *
 * A cluster's in-cluster address (e.g. a `*.svc.cluster.local` name printed
 * by the launcher) resolves only from inside that cluster; from a laptop you
 * need the externally reachable equivalent for the same cluster.
 */
export function rc34ClusterGateways(): Record<string, string> {
  const raw = process.env.RC34_CLUSTER_GATEWAYS;
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw) as Record<string, string>;
    return parsed;
  } catch (error) {
    throw new Error(
      `RC34_CLUSTER_GATEWAYS must be a JSON object of cluster -> base URL: ${error}`
    );
  }
}

export interface Rc34ModelConfig {
  /** Gateway base URL (no path); `/generate` is appended. */
  baseUrl: string;
  /** `x-deployment-name` routing header value. */
  deploymentName: string;
  maxTokens?: number;
  temperature?: number;
}

/** Full `/generate` URL for a gateway base URL. */
export function generateUrl(baseUrl: string): string {
  return `${baseUrl.replace(/\/+$/, "")}/generate`;
}

/**
 * RC34 is a reasoning model served over the raw `/generate` API in ChatML,
 * exactly like `maithinking` — it emits a `type=thought` trace before its
 * real answer and hallucinates extra turns without a stop condition. It has
 * no `tokenizer.chat_template`, so `/v1/chat/completions` fails outright with
 * a 400. Reuse the maithinking transport rather than duplicating the ChatML
 * rendering, thought-trace stripping, DNS caching, and token escalation.
 */
export function createRc34Model(config: Rc34ModelConfig): Model {
  return createMaiThinkingModel({
    url: generateUrl(config.baseUrl),
    deploymentName: config.deploymentName,
    maxTokens: config.maxTokens,
    temperature: config.temperature,
    label: RC34_SLUG,
  });
}

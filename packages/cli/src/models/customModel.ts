import {Scenario} from "@korabench/benchmark";
import {createMaiThinkingModel, isMaiThinkingSlug} from "./maiThinkingModel.js";
import {Model} from "./model.js";
import {
  createNativeRunnerModel,
  isNativeRunnerSlug,
} from "./nativeRunnerModel.js";
import {
  createRc34Model,
  isRc34Slug,
  rc34ClusterGateways,
} from "./rc34Model.js";
import {createWebRunnerModel, isWebRunnerSlug} from "./webRunnerModel.js";

const DEFAULT_WEB_RUNNER_URL = "http://localhost:7100";
const DEFAULT_NATIVE_RUNNER_URL = "http://localhost:7200";

/**
 * RC34 target configuration, from env with the documented defaults. Exported
 * so `debug-rc34` can print the resolved config without duplicating it.
 */
export function resolveRc34Config() {
  // No internal hostnames or deployment names are baked in — both are
  // environment-specific and not public. RC34_BASE_URL points straight at a
  // gateway; RC34_CLUSTER + RC34_CLUSTER_GATEWAYS resolve one by cluster name.
  const gateways = rc34ClusterGateways();
  const cluster = process.env.RC34_CLUSTER;
  if (cluster && !gateways[cluster]) {
    const known = Object.keys(gateways);
    throw new Error(
      `Unknown RC34_CLUSTER "${cluster}". ` +
        (known.length
          ? `Known: ${known.join(", ")}.`
          : `Set RC34_CLUSTER_GATEWAYS (JSON of cluster -> base URL), or use RC34_BASE_URL directly.`)
    );
  }

  const baseUrl =
    process.env.RC34_BASE_URL ?? (cluster ? gateways[cluster]! : undefined);
  if (!baseUrl) {
    throw new Error(
      "rc34: set RC34_BASE_URL (gateway base URL), or RC34_CLUSTER together with " +
        "RC34_CLUSTER_GATEWAYS. See .env.example."
    );
  }

  const deploymentName = process.env.RC34_DEPLOYMENT_NAME;
  if (!deploymentName) {
    throw new Error(
      "rc34: set RC34_DEPLOYMENT_NAME to the deployment to route to " +
        "(x-deployment-name). Gateway routing from outside the cluster must name " +
        "one deployment, not a deployment group."
    );
  }

  return {
    baseUrl,
    deploymentName,
    maxTokens: process.env.RC34_MAX_TOKENS
      ? parseInt(process.env.RC34_MAX_TOKENS, 10)
      : undefined,
  };
}

export async function createCustomModel(
  modelSlug: string,
  _scenario: Scenario
): Promise<Model> {
  if (isNativeRunnerSlug(modelSlug)) {
    const nativeRunnerUrl =
      process.env.NATIVE_RUNNER_URL ?? DEFAULT_NATIVE_RUNNER_URL;
    const apiKey = process.env.NATIVE_RUNNER_API_KEY;
    return createNativeRunnerModel({modelSlug, nativeRunnerUrl, apiKey});
  }
  if (isWebRunnerSlug(modelSlug)) {
    const webRunnerUrl = process.env.WEB_RUNNER_URL ?? DEFAULT_WEB_RUNNER_URL;
    const apiKey = process.env.WEB_RUNNER_API_KEY;
    return createWebRunnerModel({modelSlug, webRunnerUrl, apiKey});
  }
  if (isMaiThinkingSlug(modelSlug)) {
    const url = process.env.MAI_THINKING_URL;
    const deploymentName = process.env.MAI_THINKING_DEPLOYMENT_NAME;
    if (!url || !deploymentName) {
      throw new Error(
        "MAI_THINKING_URL and MAI_THINKING_DEPLOYMENT_NAME must both be set to use the maithinking target model."
      );
    }
    const maxTokens = process.env.MAI_THINKING_MAX_TOKENS
      ? parseInt(process.env.MAI_THINKING_MAX_TOKENS, 10)
      : undefined;
    return createMaiThinkingModel({url, deploymentName, maxTokens});
  }

  if (isRc34Slug(modelSlug)) {
    return createRc34Model(resolveRc34Config());
  }

  return {
    async getTextResponse() {
      throw new Error(
        `Custom model "${modelSlug}" is not implemented. ` +
          `Provide an implementation in customModel.ts.`
      );
    },

    async getStructuredResponse() {
      throw new Error(
        `Custom model "${modelSlug}" is not implemented. ` +
          `Provide an implementation in customModel.ts.`
      );
    },
  };
}

import {JudgeModel, Scenario, TestContext} from "@korabench/benchmark";
import * as R from "remeda";
import {createCustomModel} from "../../models/customModel.js";
import {createGatewayModel} from "../../models/gatewayModel.js";
import {isMaiThinkingSlug} from "../../models/maiThinkingModel.js";
import {Model} from "../../models/model.js";
import {isNativeRunnerSlug} from "../../models/nativeRunnerModel.js";
import {isRc34Slug} from "../../models/rc34Model.js";
import {isWebRunnerSlug} from "../../models/webRunnerModel.js";

export interface BuiltContext {
  context: TestContext;
  /** Tear down the target model (e.g., release the web-runner browser
   * session). Always safe to call; idempotent. */
  dispose: (outcome: "completed" | "errored") => Promise<void>;
}

export interface BuildContextOptions {
  /** Omit the system message from target-model requests. */
  omitSystemPrompt?: boolean;
}

export async function buildContext(
  judgeModels: Record<string, Model>,
  userModel: Model,
  targetModelSlug: string,
  targetGatewayModel: Model | undefined,
  scenario: Scenario,
  options: BuildContextOptions = {}
): Promise<BuiltContext> {
  const targetModel = await (async () => {
    if (targetGatewayModel) {
      return targetGatewayModel;
    }

    return createCustomModel(targetModelSlug, scenario);
  })();

  const context: TestContext = {
    getUserResponse: async request => ({
      output: await userModel.getTextResponse(request),
    }),
    getAssistantResponse: async request => ({
      output: await targetModel.getTextResponse(request),
    }),
    omitSystemPrompt: options.omitSystemPrompt === true,
    judgeModels: R.mapValues(
      judgeModels,
      (model: Model): JudgeModel => ({
        getResponse: async request => ({
          output: await model.getStructuredResponse(request),
        }),
        // Exposes which member of a "a|b" fallback chain actually answered, so
        // the recorded judgeModelSlug names the real grader rather than the
        // whole chain. Undefined for plain (non-chain) models.
        resolvedLabel: () =>
          (
            model as {lastUsedLabel?: () => string | undefined}
          ).lastUsedLabel?.(),
      })
    ),
  };

  return {
    context,
    async dispose(outcome) {
      // Only the targetModel is expected to hold disposable resources today
      // (e.g., the WebRunnerModel keeps a browser session). Gateway models
      // are stateless and have no `dispose`.
      if (targetModel.dispose) {
        await targetModel.dispose(outcome);
      }
    },
  };
}

export function resolveTargetGatewayModel(
  modelsJsonPath: string,
  targetModelSlug: string
): Model | undefined {
  if (
    targetModelSlug.startsWith("custom-") ||
    isWebRunnerSlug(targetModelSlug) ||
    isNativeRunnerSlug(targetModelSlug) ||
    isMaiThinkingSlug(targetModelSlug) ||
    isRc34Slug(targetModelSlug)
  ) {
    return undefined;
  }
  return createGatewayModel(modelsJsonPath, targetModelSlug);
}

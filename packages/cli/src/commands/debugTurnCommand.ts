import {AgeRange, conversationToNextMessagePrompt} from "@korabench/benchmark";
import {ModelMessage} from "@korabench/core";
import {Program} from "../cli.js";
import {
  enableTrace,
  printPayload,
  traceCall,
  traceStep,
} from "../debugTrace.js";
import {createCustomModel, resolveRc34Config} from "../models/customModel.js";
import {
  createGatewayModel,
  describeGatewayTextPayload,
} from "../models/gatewayModel.js";
import {renderChatMlPrompt} from "../models/maiThinkingModel.js";
import {generateUrl, isRc34Slug} from "../models/rc34Model.js";

const FILE = "debugTurnCommand.ts";

const TAGS: Record<string, ModelMessage["role"]> = {
  user: "user",
  child: "user",
  bot: "assistant",
  assistant: "assistant",
  model: "assistant",
};

/**
 * Parse `[<user>hi</user><bot>hey</bot><user>why?</user>]` into ModelMessages.
 * Surrounding square brackets are optional; whitespace between tags is ignored.
 */
export function parseConversationSpec(spec: string): readonly ModelMessage[] {
  const inner = spec.trim().replace(/^\[/, "").replace(/\]$/, "");
  const tagRegex = /<(\w+)>([\s\S]*?)<\/\1>/g;
  const messages = [...inner.matchAll(tagRegex)].map(match => {
    const tag = match[1]!.toLowerCase();
    const role = TAGS[tag];
    if (!role) {
      throw new Error(
        `Unknown conversation tag <${tag}>. Supported: ${Object.keys(TAGS)
          .map(t => `<${t}>`)
          .join(", ")}`
      );
    }
    return {role, content: match[2]!.trim()};
  });

  if (messages.length === 0) {
    throw new Error(
      `Could not parse any messages out of --conversation "${spec}". ` +
        `Expected e.g. [<user>hi</user><bot>hello</bot><user>why?</user>]`
    );
  }
  if (messages[messages.length - 1]!.role !== "user") {
    throw new Error(
      "The conversation must end with a <user> message — that's the turn the target model answers."
    );
  }
  return messages;
}

export interface DebugTurnOptions {
  /** When true, no system message is prepended (bare user/assistant turns). */
  noSystemPrompt: boolean;
  /** Age range whose child system prompt to use (ignored with noSystemPrompt). */
  ageRange?: AgeRange;
  /** Actually call the provider and print the response too. */
  send: boolean;
}

export async function debugTurnCommand(
  _program: Program,
  modelsJsonPath: string,
  targetModelSlug: string,
  conversationSpec: string,
  options: DebugTurnOptions
): Promise<void> {
  enableTrace();
  traceCall(FILE, "debugTurnCommand", `target=${targetModelSlug}`);

  const conversation = await traceStep(
    FILE,
    "parseConversationSpec",
    async () => parseConversationSpec(conversationSpec)
  );
  traceCall(
    FILE,
    "parseConversationSpec",
    `→ ${conversation.length} message(s)`
  );

  const messages: ModelMessage[] = await traceStep(
    FILE,
    "buildTargetMessages",
    async () => {
      if (options.noSystemPrompt) {
        traceCall(
          FILE,
          "buildTargetMessages",
          "--no-system-prompt → system omitted"
        );
        return [...conversation];
      }
      traceCall(
        "conversationToNextMessagePrompt.ts",
        "conversationToNextMessagePrompt",
        `ageRange=${options.ageRange ?? "undefined (default prompt)"}`
      );
      const prompt = conversationToNextMessagePrompt({
        ageRange: options.ageRange,
        modelMemory: undefined,
      });
      return [
        {role: "system" as const, content: prompt.input},
        ...conversation,
      ];
    }
  );

  const rc34 = isRc34Slug(targetModelSlug);

  const payload = await traceStep(FILE, "describePayload", async () => {
    if (rc34) {
      traceCall("customModel.ts", "resolveRc34Config");
      traceCall("maiThinkingModel.ts", "renderChatMlPrompt");
      const config = resolveRc34Config();
      return {
        target: "rc34",
        endpoint: generateUrl(config.baseUrl),
        headers: {
          "content-type": "application/json",
          "x-deployment-name": config.deploymentName,
        },
        body: {
          // RC34 has no chat template: the gateway takes a pre-rendered
          // ChatML string on /generate, not a messages array.
          text: renderChatMlPrompt(messages),
          sampling_params: {
            max_new_tokens: config.maxTokens ?? 16000,
            stop: ["<|im_start|>"],
            skip_special_tokens: false,
          },
        },
      };
    }

    traceCall("gatewayModel.ts", "describeGatewayTextPayload");
    traceCall("modelConfig.ts", "resolveModelConfig", targetModelSlug);
    return describeGatewayTextPayload(modelsJsonPath, targetModelSlug, {
      messages,
    });
  });

  printPayload(
    rc34
      ? "payload sent to RC34 for target model"
      : "payload sent to provider for target model",
    payload
  );

  if (!options.send) {
    console.log("(dry run — pass --send to actually call the target model)");
    return;
  }

  const model = await traceStep(FILE, "resolveTargetModel", async () => {
    if (rc34) {
      traceCall("customModel.ts", "createCustomModel", targetModelSlug);
      traceCall("rc34Model.ts", "createRc34Model");
      // createCustomModel takes a Scenario only for scenario-aware backends;
      // rc34 ignores it, so a placeholder keeps the debug path scenario-free.
      return createCustomModel(targetModelSlug, undefined as never);
    }
    traceCall("gatewayModel.ts", "createGatewayModel", targetModelSlug);
    return createGatewayModel(modelsJsonPath, targetModelSlug);
  });

  const text = await traceStep(
    rc34 ? "maiThinkingModel.ts" : "gatewayModel.ts",
    "getTextResponse",
    async () => {
      if (rc34) traceCall("maiThinkingModel.ts", "extractFinalAnswer");
      return model.getTextResponse({messages});
    }
  );
  console.log("\n===== target model response =====");
  console.log(text);
}

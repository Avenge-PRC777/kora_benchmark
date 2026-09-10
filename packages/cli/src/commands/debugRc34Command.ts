import {ModelMessage} from "@korabench/core";
import {Program} from "../cli.js";
import {
  enableTrace,
  printPayload,
  traceCall,
  traceStep,
} from "../debugTrace.js";
import {createCustomModel, resolveRc34Config} from "../models/customModel.js";
import {renderChatMlPrompt} from "../models/maiThinkingModel.js";
import {generateUrl} from "../models/rc34Model.js";

const FILE = "debugRc34Command.ts";

export interface DebugRc34Options {
  /** Print the resolved config + payload without calling RC34. */
  dryRun: boolean;
}

/**
 * Send a single prompt to the RC34 deployment and print its output, tracing
 * the `file::function` flow (config resolution → routing headers → model-id
 * discovery → chat-completions call).
 */
export async function debugRc34Command(
  _program: Program,
  prompt: string,
  options: DebugRc34Options
): Promise<void> {
  enableTrace();
  traceCall(FILE, "debugRc34Command", `prompt=${JSON.stringify(prompt)}`);

  const config = await traceStep(FILE, "resolveRc34Config", async () => {
    traceCall("customModel.ts", "resolveRc34Config");
    return resolveRc34Config();
  });

  const messages: ModelMessage[] = [{role: "user", content: prompt}];

  printPayload("RC34 request", {
    endpoint: generateUrl(config.baseUrl),
    headers: {
      "content-type": "application/json",
      "x-deployment-name": config.deploymentName,
    },
    body: {
      // RC34 has no chat template, so the gateway takes a pre-rendered ChatML
      // string on /generate rather than a messages array.
      text: renderChatMlPrompt(messages),
      sampling_params: {
        max_new_tokens: config.maxTokens ?? 16000,
        stop: ["<|im_start|>"],
        skip_special_tokens: false,
      },
    },
  });

  if (options.dryRun) {
    console.log("(dry run — omit --dry-run to actually call RC34)");
    return;
  }

  const model = await traceStep(FILE, "createTargetModel", async () => {
    traceCall("customModel.ts", "createCustomModel", "rc34");
    traceCall("rc34Model.ts", "createRc34Model");
    // rc34 ignores the Scenario argument; the debug path has no scenario.
    return createCustomModel("rc34", undefined as never);
  });

  const text = await traceStep(
    "maiThinkingModel.ts",
    "getTextResponse",
    async () => {
      traceCall("maiThinkingModel.ts", "renderPrompt");
      traceCall("maiThinkingModel.ts", "extractFinalAnswer");
      return model.getTextResponse({messages});
    }
  );

  console.log("\n===== RC34 output =====");
  console.log(text);
}

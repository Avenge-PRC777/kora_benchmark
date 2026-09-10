import {
  GenerateSeedsContext,
  GenerateSeedsOptions,
  kora,
  Risk,
  RiskCategory,
  Scenario,
  ScenarioSeed,
} from "@korabench/benchmark";
import {Program} from "../cli.js";
import {
  enableTrace,
  printPayload,
  traceCall,
  traceStep,
} from "../debugTrace.js";
import {createGatewayModelChain} from "../models/gatewayModel.js";

const FILE = "debugSeedsCommand.ts";

/**
 * Generate `count` seeds and expand each one just far enough to produce its
 * `firstUserMessage` — i.e. everything up to and including the first user
 * message — printing the `file::function` call flow along the way.
 */
export async function debugSeedsCommand(
  _program: Program,
  modelsJsonPath: string,
  seedModelSlugs: readonly string[],
  expandModelSlugs: readonly string[],
  userModelSlugs: readonly string[],
  count: number,
  options: GenerateSeedsOptions
): Promise<void> {
  enableTrace();
  traceCall(FILE, "debugSeedsCommand", `count=${count}`);

  // `totalSeeds` is PER RISK, so leaving the risk filter open would build
  // count x 25 tasks and (via generateScenarioSeeds' flatTransform(10, ...))
  // burn ~10 concurrent LLM calls before the loop below collects its `count`
  // seeds and stops. Pin a single risk by default so `--count 3` costs 3
  // calls, not 13. Pass --risk-ids explicitly to widen it.
  const riskIds = options.riskIds?.length
    ? options.riskIds
    : [RiskCategory.listAll()[0]!.risks[0]!.id];
  if (!options.riskIds?.length) {
    console.error(
      `[trace] no --risk-ids given; defaulting to a single risk (${riskIds[0]}) to keep the call count at --count. ` +
        `Pass --risk-ids to widen.`
    );
  }
  const seedOptions: GenerateSeedsOptions = {...options, riskIds};

  const seedModel = createGatewayModelChain(modelsJsonPath, seedModelSlugs);
  const expansionModel = createGatewayModelChain(
    modelsJsonPath,
    expandModelSlugs
  );
  const userModel = createGatewayModelChain(modelsJsonPath, userModelSlugs);

  const seedContext: GenerateSeedsContext = {
    getResponse: async request => {
      traceCall("gatewayModel.ts", "getStructuredResponse", "seed model");
      return {output: await seedModel.getStructuredResponse(request)};
    },
  };

  const seeds: ScenarioSeed[] = [];
  await traceStep(FILE, "collectSeeds", async () => {
    traceCall("kora.ts", "generateScenarioSeeds");
    traceCall("riskToScenarioSeedsPrompt.ts", "riskToScenarioSeedsPrompt");
    for await (const event of kora.generateScenarioSeeds(
      seedContext,
      seedOptions
    )) {
      for (const seed of event.items) {
        seeds.push(seed);
        if (seeds.length >= count) return;
      }
    }
  });

  if (seeds.length === 0) {
    throw new Error(
      "No seeds were generated — check the risk/motivation filters."
    );
  }

  const scenarios = await traceStep(FILE, "expandSeeds", async () => {
    traceCall("kora.ts", "expandScenario");
    traceCall("seedToScenarioPrompt.ts", "seedToScenarioPrompt");
    traceCall("scenarioToValidationPrompt.ts", "scenarioToValidationPrompt");
    traceCall("generateUserMessage.ts", "generateFirstUserMessage");
    traceCall(
      "scenarioToFirstUserMessagePrompt.ts",
      "scenarioToFirstUserMessagePrompt"
    );

    const expanded = await Promise.all(
      seeds.slice(0, count).map(seed =>
        kora.expandScenario(
          {
            getResponse: async request => ({
              output: await expansionModel.getStructuredResponse(request),
            }),
            getUserResponse: async request => ({
              output: await userModel.getTextResponse(request),
            }),
          },
          seed
        )
      )
    );
    return expanded.flat();
  });

  scenarios.forEach((scenario: Scenario, i: number) => {
    const riskCategory = RiskCategory.find(scenario.seed.riskCategoryId);
    const risk: Risk = RiskCategory.findRisk(
      riskCategory,
      scenario.seed.riskId
    );
    printPayload(`seed ${i + 1}/${scenarios.length} — ${risk.name}`, {
      seed: scenario.seed,
      shortTitle: scenario.shortTitle,
      childMaturity: scenario.childMaturity,
      childBackground: scenario.childBackground,
      narrative: scenario.narrative,
      evaluationCriteria: scenario.evaluationCriteria,
      modelMemory: scenario.modelMemory,
      firstUserMessage: scenario.firstUserMessage,
    });
  });
}

import {describe, expect, it} from "vitest";
import {createFallbackModel} from "../models/fallbackModel.js";

function stub(text: string) {
  return {
    getTextResponse: async () => text,
    getStructuredResponse: async () => ({v: text}) as never,
  };
}
function failing() {
  return {
    getTextResponse: async () => {
      throw new Error("nope");
    },
    getStructuredResponse: async () => {
      throw new Error("nope");
    },
  };
}

describe("fallback model lastUsedLabel", () => {
  it("is undefined before any call", () => {
    const m = createFallbackModel([
      {label: "a", model: stub("A")},
      {label: "b", model: stub("B")},
    ]);
    expect(m.lastUsedLabel()).toBeUndefined();
  });

  it("reports the primary when it succeeds", async () => {
    const m = createFallbackModel([
      {label: "a", model: stub("A")},
      {label: "b", model: stub("B")},
    ]);
    await m.getStructuredResponse({} as never);
    expect(m.lastUsedLabel()).toBe("a");
  });

  it("reports the FALLBACK when the primary fails", async () => {
    const m = createFallbackModel([
      {label: "a", model: failing()},
      {label: "b", model: stub("B")},
    ]);
    await m.getStructuredResponse({} as never);
    expect(m.lastUsedLabel()).toBe("b");
  });

  it("tracks per call, so a later primary success overwrites a fallback", async () => {
    let fail = true;
    const flaky = {
      getTextResponse: async () => {
        if (fail) throw new Error("nope");
        return "A";
      },
      getStructuredResponse: async () => {
        if (fail) throw new Error("nope");
        return {} as never;
      },
    };
    const m = createFallbackModel([
      {label: "a", model: flaky},
      {label: "b", model: stub("B")},
    ]);
    await m.getStructuredResponse({} as never);
    expect(m.lastUsedLabel()).toBe("b");
    fail = false;
    await m.getStructuredResponse({} as never);
    expect(m.lastUsedLabel()).toBe("a");
  });

  it("reports the label for a single-model chain too", async () => {
    const m = createFallbackModel([{label: "only", model: stub("X")}]);
    await m.getTextResponse({messages: []});
    expect(m.lastUsedLabel()).toBe("only");
  });
});

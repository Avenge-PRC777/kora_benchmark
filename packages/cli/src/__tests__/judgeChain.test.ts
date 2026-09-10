import {describe, expect, it} from "vitest";

/**
 * The judge slug grammar: "," separates distinct judges (median-scored),
 * "|" is a fallback chain WITHIN one judge. Parsing is done inline in the
 * commands; this pins the semantics so the two separators don't get conflated.
 */
function parseJudges(spec: string): string[][] {
  return spec.split(",").map(s => s.split("|").map(x => x.trim()));
}

describe("judge slug parsing", () => {
  it("treats a bare slug as a single-model chain", () => {
    expect(parseJudges("gpt-5.2:medium:limited")).toEqual([
      ["gpt-5.2:medium:limited"],
    ]);
  });

  it("treats | as a fallback chain within one judge", () => {
    expect(parseJudges("gpt-5.2:medium:limited|deepseek-v3.2")).toEqual([
      ["gpt-5.2:medium:limited", "deepseek-v3.2"],
    ]);
  });

  it("treats , as separate judges for median scoring", () => {
    expect(parseJudges("a,b,c")).toEqual([["a"], ["b"], ["c"]]);
  });

  it("supports a fallback chain per judge", () => {
    expect(parseJudges("a|a2,b|b2,c")).toEqual([
      ["a", "a2"],
      ["b", "b2"],
      ["c"],
    ]);
  });

  it("keeps the judge COUNT equal to the comma-separated entries", () => {
    // Median scoring requires an odd judge count; a fallback chain must not
    // change how many judges there are.
    expect(parseJudges("a|a2").length).toBe(1);
    expect(parseJudges("a|a2,b|b2,c|c2").length).toBe(3);
  });

  it("trims whitespace around chain entries", () => {
    expect(parseJudges("a | a2")).toEqual([["a", "a2"]]);
  });
});

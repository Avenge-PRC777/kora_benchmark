import {describe, expect, it} from "vitest";
import {extractJson} from "../models/gatewayModel.js";

describe("extractJson", () => {
  it("returns bare JSON unchanged", () => {
    expect(extractJson('{"a":1}')).toBe('{"a":1}');
  });

  it("unwraps a closed ```json fence", () => {
    expect(extractJson('```json\n{"a":1}\n```')).toBe('{"a":1}');
  });

  it("unwraps a closed bare fence", () => {
    expect(extractJson('```\n{"a":1}\n```')).toBe('{"a":1}');
  });

  it("strips an UNTERMINATED fence so the failure is the truncation, not the backticks", () => {
    // deepseek emits ```json and is cut off before the closing fence. Leaving
    // the opening fence in place made JSON.parse fail on '`' and crash the run.
    expect(extractJson('```json\n{\n  "shortTitle": "abc"')).toBe(
      '{\n  "shortTitle": "abc"'
    );
  });

  it("keeps prose around a JSON object", () => {
    expect(extractJson('Here you go:\n{"a":1}\nHope that helps.')).toBe(
      '{"a":1}'
    );
  });

  it("strips <think> blocks", () => {
    expect(extractJson('<think>reasoning</think>\n{"a":1}')).toBe('{"a":1}');
  });

  it("returns refusal prose untouched so it can be recognized as a refusal", () => {
    expect(
      extractJson("I'm sorry, but I cannot assist with that request.")
    ).toBe("I'm sorry, but I cannot assist with that request.");
  });
});

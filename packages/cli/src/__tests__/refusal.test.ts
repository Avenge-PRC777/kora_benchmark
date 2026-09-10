import {describe, expect, it} from "vitest";
import {isRefusalError} from "../retry.js";

/** Build the error shape gatewayModel throws: a JSON parse failure carrying
 * the model's full raw output on `rawOutput`. */
function parseErrorFor(rawText: string): Error {
  try {
    JSON.parse(rawText);
    throw new Error("expected a parse failure");
  } catch (error) {
    (error as {rawOutput?: string}).rawOutput = rawText;
    return error as Error;
  }
}

describe("isRefusalError", () => {
  it.each([
    "I'm sorry, but I cannot assist with that request.",
    "I am sorry, I can't help with this.",
    "I cannot assist with creating that content.",
    "I'm unable to help with this request.",
    "I won't generate that.",
  ])("treats a model refusal as non-retryable: %s", raw => {
    expect(isRefusalError(parseErrorFor(raw))).toBe(true);
  });

  it("does not flag a truncated/malformed JSON response", () => {
    expect(isRefusalError(parseErrorFor('{"shortTitle": "abc"'))).toBe(false);
  });

  it("does not flag malformed JSON whose CONTENT contains an apology", () => {
    // A raw newline inside a string yields "Bad control character". The
    // response is genuine (and retryable) even though a scenario narrative
    // legitimately quotes an apology — common on the self-harm risk.
    const raw =
      '{"narrative":"She told her friend, I am sorry I lied about eating lunch today.\nShe felt guilty."}';
    expect(isRefusalError(parseErrorFor(raw))).toBe(false);
  });

  it("does not flag a refusal-sounding phrase buried deep in prose", () => {
    const raw = "x".repeat(400) + " I'm sorry about that.";
    expect(isRefusalError(parseErrorFor(raw))).toBe(false);
  });

  it("does not flag prose that merely mentions being sorry inside valid JSON", () => {
    // Valid JSON never reaches the parse-failure branch at all.
    const error = new Error("Invalid length: Expected <=500 but received 530");
    expect(isRefusalError(error)).toBe(false);
  });

  it("does not flag a schema-validation error", () => {
    const error = new Error(
      'Invalid type: Expected ("white" | "black") but received "Asian"'
    );
    expect(isRefusalError(error)).toBe(false);
  });

  it("falls back to the error message when rawOutput is absent", () => {
    const error = new Error(
      `Unexpected token 'I', "I'm sorry,"... is not valid JSON`
    );
    expect(isRefusalError(error)).toBe(true);
  });

  it("ignores non-Error values", () => {
    expect(isRefusalError("I'm sorry")).toBe(false);
  });
});

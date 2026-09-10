import {describe, expect, it} from "vitest";
import {extractFinalAnswer} from "../models/maiThinkingModel.js";

describe("extractFinalAnswer", () => {
  it("returns a plain direct answer unchanged", () => {
    expect(extractFinalAnswer("Hi! How can I help you today?")).toBe(
      "Hi! How can I help you today?"
    );
  });

  it("strips a standard reasoning trace", () => {
    expect(
      extractFinalAnswer(
        "type=thought<|im_end|>lots of reasoning<|im_end|>The answer is 4."
      )
    ).toBe("The answer is 4.");
  });

  it("strips a trace whose header is preceded by a special token", () => {
    // rc34 emits "<|im_sep|>type=thought" (22 chars) — past the old 10-char
    // header window, which leaked the whole trace as the answer.
    expect(
      extractFinalAnswer(
        "<|im_sep|>type=thought<|im_end|>reasoning here<|im_end|>Hello!"
      )
    ).toBe("Hello!");
  });

  it("strips a trace with a garbled header", () => {
    expect(
      extractFinalAnswer("=thought<|im_end|>reasoning<|im_end|>Answer.")
    ).toBe("Answer.");
  });

  it("strips a HEADERLESS reasoning trace (two segments)", () => {
    // rc34 sometimes omits the header entirely; treating this as a direct
    // answer leaked the reasoning into the conversation.
    expect(
      extractFinalAnswer(
        "To solve the problem, let's proceed step by step.<|im_end|>Hi there!"
      )
    ).toBe("Hi there!");
  });

  it("keeps a direct answer that has a trailing stop delimiter", () => {
    expect(extractFinalAnswer("Hello there!<|im_end|>")).toBe("Hello there!");
  });

  it("discards hallucinated turns after a direct answer", () => {
    expect(
      extractFinalAnswer("Hello!<|im_end|>\n\n<|im_end|>fake extra turn")
    ).toBe("Hello!");
  });

  it("does not mistake an answer about thinking for a reasoning header", () => {
    expect(
      extractFinalAnswer("I thought about this and the answer is yes.")
    ).toBe("I thought about this and the answer is yes.");
  });

  it("strips a trace with a DOUBLED delimiter after the header", () => {
    // rc34 emits ["type=thought", "", reasoning, answer] — slicing from a
    // fixed index returned reasoning+answer joined and leaked the trace.
    expect(
      extractFinalAnswer(
        "type=thought<|im_end|><|im_end|>long reasoning here<|im_end|>Hi there!"
      )
    ).toBe("Hi there!");
  });

  it("strips a TRAILING thought header after the answer", () => {
    // rc34 finishes its answer then starts another thought block, which the
    // stop token truncates. Observed leaking into 11/594 recorded replies.
    expect(
      extractFinalAnswer(
        "type=thought<|im_end|>reasoning<|im_end|>I'm right here with you.<|im_sep|>type=thought"
      )
    ).toBe("I'm right here with you.");
  });

  it("strips a trailing special token with no header word", () => {
    expect(
      extractFinalAnswer("type=thought<|im_end|>r<|im_end|>Done.<|im_sep|>")
    ).toBe("Done.");
  });

  it("leaves an answer ending in the plain word 'thought' alone", () => {
    expect(
      extractFinalAnswer("type=thought<|im_end|>r<|im_end|>That is my thought")
    ).toBe("That is my thought");
  });

  it("throws when a headed trace is cut off before its answer", () => {
    expect(() =>
      extractFinalAnswer("type=thought<|im_end|>reasoning that never finished")
    ).toThrow(/ended before completing its reasoning/);
  });
});

import {afterEach, describe, expect, it} from "vitest";
import {renderChatMlPrompt} from "../models/maiThinkingModel.js";
import {
  generateUrl,
  isRc34Slug,
  rc34ClusterGateways,
} from "../models/rc34Model.js";

describe("isRc34Slug", () => {
  it("matches only the rc34 slug", () => {
    expect(isRc34Slug("rc34")).toBe(true);
    expect(isRc34Slug("maithinking")).toBe(false);
    expect(isRc34Slug("gpt-4o")).toBe(false);
    expect(isRc34Slug("rc34-extra")).toBe(false);
  });
});

describe("rc34ClusterGateways", () => {
  // Gateway hosts are environment-specific and deliberately not hardcoded;
  // the mapping comes from RC34_CLUSTER_GATEWAYS.
  afterEach(() => {
    delete process.env.RC34_CLUSTER_GATEWAYS;
  });

  it("is empty when unset, so nothing internal is baked in", () => {
    delete process.env.RC34_CLUSTER_GATEWAYS;
    expect(rc34ClusterGateways()).toEqual({});
  });

  it("parses a cluster -> base URL mapping from the environment", () => {
    process.env.RC34_CLUSTER_GATEWAYS = JSON.stringify({
      clusterA: "http://gw-a:8080",
      clusterB: "http://gw-b:8080",
    });
    expect(rc34ClusterGateways()).toEqual({
      clusterA: "http://gw-a:8080",
      clusterB: "http://gw-b:8080",
    });
  });

  it("throws a clear error on malformed JSON", () => {
    process.env.RC34_CLUSTER_GATEWAYS = "not json";
    expect(() => rc34ClusterGateways()).toThrow(/must be a JSON object/);
  });
});

describe("generateUrl", () => {
  it("appends /generate", () => {
    expect(generateUrl("http://gw:8080")).toBe("http://gw:8080/generate");
  });

  it("does not double the slash on a trailing-slash base URL", () => {
    expect(generateUrl("http://gw:8080/")).toBe("http://gw:8080/generate");
  });
});

describe("renderChatMlPrompt", () => {
  it("renders turns as ChatML primed for an assistant reply", () => {
    expect(
      renderChatMlPrompt([
        {role: "user", content: "hi"},
        {role: "assistant", content: "hey"},
        {role: "user", content: "why?"},
      ])
    ).toBe(
      "<|im_start|>user\nhi<|im_end|>\n" +
        "<|im_start|>assistant\nhey<|im_end|>\n" +
        "<|im_start|>user\nwhy?<|im_end|>\n" +
        "<|im_start|>assistant\n"
    );
  });

  it("renders a system message as its own ChatML turn", () => {
    expect(
      renderChatMlPrompt([
        {role: "system", content: "be safe"},
        {role: "user", content: "hi"},
      ])
    ).toBe(
      "<|im_start|>system\nbe safe<|im_end|>\n" +
        "<|im_start|>user\nhi<|im_end|>\n" +
        "<|im_start|>assistant\n"
    );
  });
});

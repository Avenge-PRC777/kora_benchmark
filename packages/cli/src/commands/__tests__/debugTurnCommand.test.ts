import {describe, expect, it} from "vitest";
import {parseConversationSpec} from "../debugTurnCommand.js";

describe("parseConversationSpec", () => {
  it("parses a bracketed user/bot/user conversation", () => {
    expect(
      parseConversationSpec(
        "[<user>hi</user><bot>bot reply here</bot><user>user question here</user>]"
      )
    ).toEqual([
      {role: "user", content: "hi"},
      {role: "assistant", content: "bot reply here"},
      {role: "user", content: "user question here"},
    ]);
  });

  it("accepts the spec without surrounding brackets and with whitespace", () => {
    expect(
      parseConversationSpec(
        "<user> hi </user>\n<assistant>yo</assistant>\n<user>ok</user>"
      )
    ).toEqual([
      {role: "user", content: "hi"},
      {role: "assistant", content: "yo"},
      {role: "user", content: "ok"},
    ]);
  });

  it("rejects unknown tags", () => {
    expect(() => parseConversationSpec("[<judge>hi</judge>]")).toThrow(
      /Unknown conversation tag <judge>/
    );
  });

  it("rejects a conversation that does not end on a user turn", () => {
    expect(() =>
      parseConversationSpec("[<user>hi</user><bot>bye</bot>]")
    ).toThrow(/must end with a <user> message/);
  });

  it("rejects an unparseable spec", () => {
    expect(() => parseConversationSpec("hello there")).toThrow(
      /Could not parse any messages/
    );
  });
});

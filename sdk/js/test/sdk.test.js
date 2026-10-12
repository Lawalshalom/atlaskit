// Runs against a live gateway: ATLASKIT_BASE_URL / ATLASKIT_API_KEY (the pytest suite starts one on a mock model).
import { test } from "node:test";
import assert from "node:assert/strict";
import { AtlasKit, AtlasKitError, LANGUAGES } from "../src/index.js";

const live = !!process.env.ATLASKIT_BASE_URL;
const atlas = new AtlasKit({ maxRetries: 0 });

test("exports languages", () => {
  assert.deepEqual(Object.keys(LANGUAGES), ["en", "yo", "ha", "ig"]);
});

test("builds chat bodies", () => {
  const b = AtlasKit._chatBody("hi", { language: "yo", system: "S", maxTokens: 9 });
  assert.equal(b.atlas_language, "yo");
  assert.equal(b.max_tokens, 9);
  assert.deepEqual(b.messages[0], { role: "system", content: "S" });
});

test("chat", { skip: !live }, async () => {
  const r = await atlas.chat("Sannu", { language: "ha" });
  assert.equal(r.text, "echo: Sannu");
});

test("stream", { skip: !live }, async () => {
  let out = "";
  for await (const d of atlas.stream("Kedu")) out += d;
  assert.equal(out.trim(), "echo: Kedu");
});

test("conversation", { skip: !live }, async () => {
  const c = atlas.conversation({ language: "ig" });
  await c.say("one");
  await c.say("two");
  assert.equal(c.messages.length, 4);
});

test("translate, detect, transcribe", { skip: !live }, async () => {
  assert.match(await atlas.translate("Hello", "yo"), /^echo:/);
  assert.equal((await atlas.detectLanguage("Kedu ka ị mere?")).language, "ig");
  const t = await atlas.transcribe(new Uint8Array([82, 73, 70, 70, 0, 0]), "ha", { filename: "a.wav" });
  assert.ok(t.text.length > 0);
});

test("errors carry status", { skip: !live }, async () => {
  const bad = new AtlasKit({ apiKey: "nope", maxRetries: 0 });
  await assert.rejects(bad.chat("x"), (e) => e instanceof AtlasKitError && e.status === 401);
});

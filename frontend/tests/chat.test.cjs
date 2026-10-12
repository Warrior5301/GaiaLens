// Run with: node --test tests/chat.test.cjs
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const ts = require("typescript");

function client(fetch) {
    const source = fs.readFileSync(require.resolve("../src/lib/chat.ts"), "utf8");
    const code = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText;
    const context = { exports: {}, fetch, FormData, File, Error };
    vm.runInNewContext(code, context);
    return context.exports.sendChat;
}

test("text-only chat uses the gateway and sends prior turns separately", async () => {
    const send = client(async (url, { body }) => {
        assert.equal(url, "/api/chat");
        assert.equal(body.get("query"), "Explain SAR");
        assert.equal(body.getAll("images").length, 0);
        assert.deepEqual(JSON.parse(body.get("history")), [{ role: "user", content: "Hello" }]);
        return Response.json({ response: "Radar imagery", images: [] });
    });
    assert.equal((await send("Explain SAR", [{ role: "user", content: "Hello", id: "ui-only" }])).content, "Radar imagery");
});

test("multi-image chat preserves upload order and returns tool overlays", async () => {
    const send = client(async (_, { body }) => {
        assert.deepEqual(body.getAll("images").map(file => file.name), ["before.png", "after.png"]);
        return Response.json({ response: "Changes found", images: [{ url: "/files/overlay.png" }] });
    });
    const result = await send("Compare", [], [new File(["a"], "before.png"), new File(["b"], "after.png")]);
    assert.equal(result.images[0].url, "/files/overlay.png");
});

test("gateway failures surface instead of being presented as model answers", async () => {
    const send = client(async () => Response.json({ error: "Backend unavailable" }, { status: 503 }));
    await assert.rejects(send("Hello", []), /Backend unavailable/);
});

const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

class FakeElement {
    constructor() {
        this.children = [];
        this.className = "";
        this.dataset = {};
        this.disabled = false;
        this.download = "";
        this.hidden = false;
        this.href = "";
        this.listeners = {};
        this.scrollHeight = 0;
        this.scrollTop = 0;
        this.textContent = "";
        this.value = "";
    }

    addEventListener(type, listener) { this.listeners[type] = listener; }
    append(...children) { this.children.push(...children); }
    appendChild(child) { this.children.push(child); }
    click() { this.clicked = true; }
    focus() {}
    removeAttribute(name) { if (this.attributes) delete this.attributes[name]; }
    remove() { this.removed = true; }
    replaceChildren(...children) { this.children = children; }
    setAttribute(name, value) {
        if (!this.attributes) this.attributes = {};
        this.attributes[name] = String(value);
    }
}

(async () => {
    const elements = new Map();
    for (const id of [
        "play-board", "game-status", "game-detail", "move-list", "move-count",
        "thinking-rail", "keyboard-move-form", "keyboard-move", "play-keyboard-move",
        "top-player-name", "bottom-player-name", "top-player-dot", "bottom-player-dot",
        "copy-fen", "copy-pgn", "download-pgn", "export-feedback", "fen-value",
    ]) {
        elements.set(id, new FakeElement());
    }

    const currentFen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2";
    const currentPgn = '[Event "Human vs LLM"]\n[White "You"]\n[Black "reasoner"]\n[Result "*"]\n\n1. e4 e5 *';
    const initialGame = {
        fen: currentFen,
        pgn: currentPgn,
        human_color: "white",
        model_id: "reasoner",
        reasoning_effort: "high",
        san_moves: ["e4", "e5"],
        status: "active",
        turn: "human",
    };
    const app = new FakeElement();
    app.dataset = {csrfToken: "csrf", moveUrl: "/move", startUrl: "/start"};
    elements.set("play-app", app);
    const initialNode = new FakeElement();
    initialNode.textContent = JSON.stringify(initialGame);
    elements.set("initial-game", initialNode);

    const createdLinks = [];
    const documentObject = {
        addEventListener() {},
        body: new FakeElement(),
        createElement(tag) {
            const element = new FakeElement();
            if (tag === "a") createdLinks.push(element);
            return element;
        },
        getElementById(id) { return elements.get(id) || null; },
    };
    const windowObject = {
        addEventListener() {},
        Chess: class {
            constructor(fen) { this.currentFen = fen; }
            fen() { return "optimistic-fen"; }
            move({from, to}) { return from === "e2" && to === "e4" ? {} : null; }
        },
        Chessboard(_id, options) {
            boardOptions = options;
            return {orientation() {}, position() {}, resize() {}, start() {}};
        },
        prompt() { return "queen"; },
    };
    let boardOptions;
    const copied = [];
    const objectUrls = [];
    const source = fs.readFileSync("web/static/js/play.js", "utf8");
    vm.runInNewContext(source, {
        Blob,
        console,
        document: documentObject,
        fetch() { return new Promise(() => {}); },
        FormData: class {},
        navigator: {clipboard: {writeText: async (value) => copied.push(value)}},
        URL: {
            createObjectURL(blob) {
                objectUrls.push(blob);
                return "blob:game-pgn";
            },
            revokeObjectURL() {},
        },
        window: windowObject,
    });

    assert.equal(elements.get("fen-value").textContent, currentFen);
    assert.equal(elements.get("copy-fen").disabled, false);
    assert.equal(elements.get("copy-pgn").disabled, false);
    assert.equal(elements.get("download-pgn").disabled, false);

    await elements.get("copy-fen").listeners.click();
    await elements.get("copy-pgn").listeners.click();
    assert.deepEqual(copied, [currentFen, currentPgn]);

    elements.get("download-pgn").listeners.click();
    assert.equal(createdLinks.length, 1);
    assert.equal(createdLinks[0].download, "chessbench-vs-reasoner-high.pgn");
    assert.equal(createdLinks[0].href, "blob:game-pgn");
    assert.equal(createdLinks[0].clicked, true);
    assert.equal(await objectUrls[0].text(), currentPgn + "\n");

    boardOptions.onDrop("e2", "e4", "wP");
    assert.equal(elements.get("export-feedback").textContent, "");
})().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});

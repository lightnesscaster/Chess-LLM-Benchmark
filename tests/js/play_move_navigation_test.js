const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

class FakeElement {
    constructor(tag = "div") {
        this.tag = tag;
        this.tagName = tag.toUpperCase();
        this.children = [];
        this.className = "";
        this.dataset = {};
        this.disabled = false;
        this.hidden = false;
        this.listeners = {};
        this.scrollHeight = 0;
        this.scrollTop = 0;
        this.textContent = "";
        this.value = "";
        this.attributes = {};
    }
    addEventListener(type, listener) { this.listeners[type] = listener; }
    append(...children) { this.children.push(...children); }
    appendChild(child) { this.children.push(child); }
    focus() {}
    removeAttribute(name) { delete this.attributes[name]; }
    replaceChildren(...children) { this.children = children; }
    setAttribute(name, value) { this.attributes[name] = String(value); }
}

const finalFen = "final-fen";
const initialGame = {
    fen: finalFen,
    pgn: "1. e4 e5 *",
    human_color: "white",
    model_id: "test-model",
    moves: ["e2e4", "e7e5"],
    san_moves: ["e4", "e5"],
    status: "active",
    turn: "human",
};

const elements = new Map();
for (const id of [
    "play-board", "game-status", "game-detail", "move-list", "move-count",
    "move-start", "move-previous", "move-next", "move-live", "move-position",
    "thinking-rail", "keyboard-move-form", "keyboard-move", "play-keyboard-move",
    "top-player-name", "bottom-player-name", "top-player-dot", "bottom-player-dot",
    "copy-fen", "copy-pgn", "download-pgn", "export-feedback", "fen-value",
]) elements.set(id, new FakeElement());
elements.set("keyboard-move", new FakeElement("input"));

const app = new FakeElement();
app.dataset = {csrfToken: "csrf", moveUrl: "/move", startUrl: "/start"};
elements.set("play-app", app);
const initialNode = new FakeElement();
initialNode.textContent = JSON.stringify(initialGame);
elements.set("initial-game", initialNode);

class FakeChess {
    constructor(fen) {
        this.position = fen || "start-fen";
        this.replayed = [];
    }
    fen() { return this.position; }
    move(move) {
        const uci = typeof move === "string" ? move : move.from + move.to;
        if (uci === "h2h3" && this.position === finalFen) {
            this.position = "optimistic-h2h3";
            return {};
        }
        if (uci === "e2e4" && this.replayed.length === 0) {
            this.replayed.push(uci);
            this.position = "after-e2e4";
            return {};
        }
        if (uci === "e7e5" && this.replayed.length === 1) {
            this.replayed.push(uci);
            this.position = finalFen;
            return {};
        }
        return null;
    }
}

const boardPositions = [];
const copied = [];
let boardOptions;
const windowObject = {
    addEventListener() {},
    Chess: FakeChess,
    Chessboard(_id, options) {
        boardOptions = options;
        return {
            orientation() {},
            position(fen) { boardPositions.push(fen); },
            resize() {},
            start() { boardPositions.push("start"); },
        };
    },
};
const documentObject = {
    listeners: {},
    addEventListener(type, listener) { this.listeners[type] = listener; },
    createElement(tag) { return new FakeElement(tag); },
    getElementById(id) { return elements.get(id) || null; },
};

(async () => {
    let rejectMoveRequest;
    const moveRequest = new Promise((_resolve, reject) => { rejectMoveRequest = reject; });
    vm.runInNewContext(fs.readFileSync("web/static/js/play.js", "utf8"), {
        Blob,
        console,
        document: documentObject,
        fetch() { return moveRequest; },
        FormData: class {},
        navigator: {clipboard: {writeText: async (value) => copied.push(value)}},
        URL: {createObjectURL() {}, revokeObjectURL() {}},
        window: windowObject,
    });

    assert.equal(boardPositions.at(-1), finalFen);
    assert.equal(elements.get("move-position").textContent, "Live · 2 / 2");

    const pressKey = (key, target = new FakeElement(), modifiers = {}) => {
        let prevented = false;
        documentObject.listeners.keydown({
            key,
            target,
            preventDefault() { prevented = true; },
            ...modifiers,
        });
        return prevented;
    };

    assert.equal(pressKey("ArrowLeft"), true);
    assert.equal(boardPositions.at(-1), "after-e2e4");
    assert.equal(elements.get("move-position").textContent, "1 / 2");
    assert.equal(pressKey("ArrowRight"), true);
    assert.equal(boardPositions.at(-1), finalFen);
    assert.equal(elements.get("move-position").textContent, "Live · 2 / 2");
    assert.equal(pressKey("ArrowUp"), true);
    assert.equal(boardPositions.at(-1), "start-fen");
    assert.equal(pressKey("ArrowDown"), true);
    assert.equal(boardPositions.at(-1), finalFen);

    assert.equal(pressKey("ArrowLeft", elements.get("keyboard-move")), false);
    assert.equal(boardPositions.at(-1), finalFen);
    assert.equal(pressKey("ArrowLeft", new FakeElement(), {metaKey: true}), false);
    assert.equal(boardPositions.at(-1), finalFen);

    const firstMove = elements.get("move-list").children[0].children[1];
    firstMove.listeners.click();
    assert.equal(boardPositions.at(-1), "after-e2e4");
    assert.equal(elements.get("move-position").textContent, "1 / 2");
    assert.equal(elements.get("fen-value").textContent, "after-e2e4");
    await elements.get("copy-fen").listeners.click();
    assert.deepEqual(copied, ["after-e2e4"]);
    assert.equal(boardOptions.onDragStart("e2", "wP"), false);

    elements.get("move-start").listeners.click();
    assert.equal(boardPositions.at(-1), "start-fen");
    elements.get("move-next").listeners.click();
    assert.equal(boardPositions.at(-1), "after-e2e4");
    elements.get("move-live").listeners.click();
    assert.equal(boardPositions.at(-1), finalFen);
    assert.equal(boardOptions.onDragStart("e2", "wP"), true);

    assert.equal(boardOptions.onDrop("h2", "h3", "wP"), undefined);
    elements.get("move-start").listeners.click();
    boardOptions.onSnapEnd();
    assert.equal(boardPositions.at(-1), "start-fen");
    rejectMoveRequest(new Error("provider unavailable"));
    await Promise.resolve();
    await Promise.resolve();
    assert.equal(boardPositions.at(-1), "start-fen");
    assert.equal(elements.get("move-position").textContent, "0 / 2");
})().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});

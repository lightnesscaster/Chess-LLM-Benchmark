const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

class FakeElement {
    constructor() {
        this.children = [];
        this.className = "";
        this.dataset = {};
        this.disabled = false;
        this.hidden = false;
        this.scrollHeight = 0;
        this.scrollTop = 0;
        this.textContent = "";
        this.value = "";
    }

    addEventListener() {}
    append(...children) { this.children.push(...children); }
    appendChild(child) { this.children.push(child); }
    focus() {}
    removeAttribute(name) { if (this.attributes) delete this.attributes[name]; }
    replaceChildren(...children) { this.children = children; }
    setAttribute(name, value) {
        if (!this.attributes) this.attributes = {};
        this.attributes[name] = String(value);
    }
}

const initialGame = {
    fen: "before-e2e4",
    human_color: "white",
    model_id: "test-model",
    san_moves: [],
    status: "active",
    turn: "human",
};
const elements = new Map();
for (const id of [
    "play-board",
    "game-status",
    "game-detail",
    "move-list",
    "move-count",
    "thinking-rail",
    "keyboard-move-form",
    "keyboard-move",
    "play-keyboard-move",
    "top-player-name",
    "bottom-player-name",
    "top-player-dot",
    "bottom-player-dot",
]) {
    elements.set(id, new FakeElement());
}
const app = new FakeElement();
app.dataset = {
    csrfToken: "test-csrf",
    moveUrl: "/api/admin/play/move",
    startUrl: "/api/admin/play/start",
};
elements.set("play-app", app);
const initialNode = new FakeElement();
initialNode.textContent = JSON.stringify(initialGame);
elements.set("initial-game", initialNode);

const boardPositions = [];
let boardOptions;
const windowObject = {
    addEventListener() {},
    Chess: class {
        constructor(fen) { this.currentFen = fen; }
        fen() { return this.currentFen; }
        move({from, to}) {
            if (from !== "e2" || to !== "e4") return null;
            this.currentFen = "after-e2e4";
            return {};
        }
    },
    Chessboard(_id, options) {
        boardOptions = options;
        return {
            orientation() {},
            position(fen) { boardPositions.push(fen); },
            resize() {},
            start() {},
        };
    },
    prompt() { return "queen"; },
};
const documentObject = {
    createElement() { return new FakeElement(); },
    getElementById(id) { return elements.get(id) || null; },
};

const pendingResponse = new Promise(() => {});
const source = fs.readFileSync("web/static/js/play.js", "utf8");
vm.runInNewContext(source, {
    console,
    document: documentObject,
    fetch() { return pendingResponse; },
    FormData: class {},
    window: windowObject,
});

assert.equal(boardOptions.onDrop("e2", "e4", "wP"), undefined);
boardOptions.onSnapEnd();
assert.equal(
    boardPositions.at(-1),
    "after-e2e4",
    "the local move should remain visible while the model request is pending",
);

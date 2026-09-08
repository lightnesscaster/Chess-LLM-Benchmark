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
        this.listeners = {};
    }

    addEventListener(type, listener) { this.listeners[type] = listener; }
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

const humanBlack = process.argv.includes("black");
const keyboard = process.argv.includes("keyboard");
const fail = process.argv.includes("failure");
const sourceSquare = humanBlack ? "e7" : "e2";
const targetSquare = humanBlack ? "e5" : "e4";
const san = humanBlack ? "e5" : "e4";
const initialGame = {
    fen: "before-e2e4",
    human_color: humanBlack ? "black" : "white",
    model_id: "test-model",
    san_moves: humanBlack ? ["e4"] : [],
    moves: humanBlack ? ["e2e4"] : [],
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
            if (from !== sourceSquare || to !== targetSquare) return null;
            this.currentFen = "after-e2e4";
            return {san};
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
    addEventListener() {},
    createElement() { return new FakeElement(); },
    getElementById(id) { return elements.get(id) || null; },
};

let resolveResponse, rejectResponse;
const pendingResponse = new Promise((resolve, reject) => {
    resolveResponse = resolve;
    rejectResponse = reject;
});
const source = fs.readFileSync("web/static/js/play.js", "utf8");
vm.runInNewContext(source, {
    console,
    document: documentObject,
    fetch() { return pendingResponse; },
    FormData: class {},
    window: windowObject,
});

if (keyboard) {
    elements.get("keyboard-move").value = sourceSquare + targetSquare;
    elements.get("keyboard-move-form").listeners.submit({preventDefault() {}});
} else {
    assert.equal(boardOptions.onDrop(sourceSquare, targetSquare, humanBlack ? "bP" : "wP"), undefined);
}
boardOptions.onSnapEnd();
assert.equal(
    boardPositions.at(-1),
    "after-e2e4",
    "the local move should remain visible while the model request is pending",
);
assert.equal(elements.get("move-count").textContent, humanBlack ? "2 plies" : "1 ply");
assert.equal(elements.get("move-list").children[0].children[humanBlack ? 2 : 1].textContent, san);
if (fail) rejectResponse(new Error("provider unavailable"));
else resolveResponse({ok: true, json: async () => ({game: {
    ...initialGame, fen: "confirmed", moves: [...initialGame.moves, sourceSquare + targetSquare, "g8f6"],
    san_moves: [...initialGame.san_moves, san, "Nf6"],
}})});
setImmediate(() => {
    if (fail) {
        assert.equal(boardPositions.at(-1), initialGame.fen);
        assert.equal(elements.get("move-count").textContent, humanBlack ? "1 ply" : "0 plies");
    } else {
        assert.equal(boardPositions.at(-1), "confirmed");
        assert.equal(elements.get("move-count").textContent, humanBlack ? "3 plies" : "2 plies");
    }
    assert.equal(elements.get("thinking-rail").hidden, true);
});

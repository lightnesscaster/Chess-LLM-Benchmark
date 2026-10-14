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
    showModal() { this.open = true; }
    close() { this.open = false; }
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
const promotion = process.argv.includes("promotion");
const cancel = process.argv.includes("cancel");
const sourceSquare = promotion ? (humanBlack ? "e2" : "e7") : (humanBlack ? "e7" : "e2");
const targetSquare = promotion ? (humanBlack ? "e1" : "e8") : (humanBlack ? "e5" : "e4");
const san = humanBlack ? "e5" : "e4";
const initialGame = {
    game_id: "original-game", recovery: {id: "approved-recovery"},
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
    "restore-game", "recovery-note",
    "promotion-dialog",
    "promotion-choices",
    "promotion-cancel",
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
    recoveryUrl: "/api/play/recover",
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
        move({from, to, promotion: chosen}) {
            if (from !== sourceSquare || to !== targetSquare) return null;
            this.currentFen = "after-e2e4";
            return {san, promotion: promotion ? chosen : undefined};
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
const requests = [];
const pendingResponse = new Promise((resolve, reject) => {
    resolveResponse = resolve;
    rejectResponse = reject;
});
const source = fs.readFileSync("web/static/js/play.js", "utf8");
vm.runInNewContext(source, {
    console,
    document: documentObject,
    fetch(url, options) { requests.push(JSON.parse(options.body)); return pendingResponse; },
    FormData: class {},
    window: windowObject,
});


const restore = elements.get("restore-game");
assert.equal(restore.hidden, false);
assert.equal(typeof restore.listeners.click, "function", "approved game must have a restore action");
restore.listeners.click();
restore.listeners.click();
assert.deepEqual(requests, [{game_id: "original-game", recovery_id: "approved-recovery",
    expected_fen: initialGame.fen, expected_ply: initialGame.moves.length}]);
assert.equal(restore.disabled, true);
assert.equal(elements.get("thinking-rail").hidden, true, "recovery must not claim the model is thinking");
if (fail) rejectResponse(new Error("Recovery unavailable"));
else resolveResponse({ok: true, json: async () => ({game: {
    ...initialGame, recovery: null, fen: "restored-fen", moves: ["e2e4", "e7e5"], san_moves: ["e4", "e5"],
}})});
setImmediate(() => {
    if (fail) {
        assert.equal(boardPositions.at(-1), initialGame.fen);
        assert.equal(restore.hidden, false);
        assert.equal(elements.get("game-detail").textContent, "Recovery unavailable");
    } else {
        assert.equal(boardPositions.at(-1), "restored-fen");
        assert.equal(restore.hidden, true);
        assert.equal(elements.get("move-count").textContent, "2 plies");
    }
    assert.equal(restore.disabled, false);
    assert.equal(elements.get("play-keyboard-move").disabled, false);
});

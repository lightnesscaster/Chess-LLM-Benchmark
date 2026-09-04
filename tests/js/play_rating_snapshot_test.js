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
    removeAttribute(name) { if (this.attributes) delete this.attributes[name]; }
    replaceChildren(...children) { this.children = children; }
    setAttribute(name, value) {
        if (!this.attributes) this.attributes = {};
        this.attributes[name] = String(value);
    }
}

function renderSnapshot(ratingPool, termination = "checkmate", illegalMoveCount = 0) {
    const profile = {
        username: "Some_Player",
        rating: 1847,
        rating_deviation: 73,
        provisional: false,
    };
    if (ratingPool) profile.rating_pool = ratingPool;
    const game = {
        fen: "start",
        pgn: "[Result \"1-0\"]",
        human_color: "white",
        model_id: "reasoner",
        san_moves: [],
        status: "finished",
        winner: "human",
        termination,
        llm_illegal_moves: illegalMoveCount,
        turn: "finished",
        human_profile: profile,
        rating_result: {
            recorded: true,
            model_rating: 1512,
            model_rating_deviation: 168,
        },
    };

    const elements = new Map();
    for (const id of [
        "play-board", "game-status", "game-detail", "move-list", "move-count",
        "thinking-rail", "keyboard-move-form", "keyboard-move", "play-keyboard-move",
        "top-player-name", "bottom-player-name", "top-player-dot", "bottom-player-dot",
    ]) elements.set(id, new FakeElement());

    const app = new FakeElement();
    app.dataset = {csrfToken: "csrf", moveUrl: "/move", startUrl: "/start"};
    elements.set("play-app", app);
    const initialNode = new FakeElement();
    initialNode.textContent = JSON.stringify(game);
    elements.set("initial-game", initialNode);

    const documentObject = {
        createElement() { return new FakeElement(); },
        getElementById(id) { return elements.get(id) || null; },
    };
    const windowObject = {
        addEventListener() {},
        Chessboard() { return {orientation() {}, position() {}, resize() {}, start() {}}; },
    };

    vm.runInNewContext(fs.readFileSync("web/static/js/play.js", "utf8"), {
        console,
        document: documentObject,
        fetch() { throw new Error("no request expected"); },
        FormData: class {},
        window: windowObject,
    });
    return elements;
}

const classicalElements = renderSnapshot("classical");
assert.equal(classicalElements.get("bottom-player-name").textContent, "Some_Player (you)");
assert.equal(classicalElements.get("game-status").textContent, "You won by checkmate.");
assert.match(classicalElements.get("game-detail").textContent, /Classical 1847, RD 73/);
assert.match(classicalElements.get("game-detail").textContent, /model rating 1512, RD 168/);

const legacyElements = renderSnapshot();
assert.match(legacyElements.get("game-detail").textContent, /Rapid 1847, RD 73/);

const forfeitElements = renderSnapshot("classical", "llm_forfeit_illegal_move", 2);
assert.equal(forfeitElements.get("game-status").textContent, "You won by forfeit.");
assert.match(
    forfeitElements.get("game-detail").textContent,
    /reasoner returned 2 invalid move responses/,
);

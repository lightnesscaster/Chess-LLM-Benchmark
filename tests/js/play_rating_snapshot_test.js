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
    replaceChildren(...children) { this.children = children; }
}

const game = {
    fen: "start",
    pgn: "[Result \"1-0\"]",
    human_color: "white",
    model_id: "reasoner",
    san_moves: [],
    status: "finished",
    winner: "human",
    termination: "checkmate",
    turn: "finished",
    human_profile: {
        username: "Some_Player",
        rating: 1847,
        rating_deviation: 73,
        provisional: false,
    },
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

assert.equal(elements.get("bottom-player-name").textContent, "Some_Player (you)");
assert.match(elements.get("game-detail").textContent, /Rapid 1847, RD 73/);
assert.match(elements.get("game-detail").textContent, /model rating 1512, RD 168/);

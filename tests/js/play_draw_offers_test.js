const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

class Element {
    constructor() {
        Object.assign(this, {children: [], dataset: {}, listeners: {}, attributes: {},
            disabled: false, hidden: false, checked: false, value: "", textContent: ""});
    }
    addEventListener(type, callback) { this.listeners[type] = callback; }
    append(...children) { this.children.push(...children); }
    appendChild(child) { this.children.push(child); }
    replaceChildren(...children) { this.children = children; }
    setAttribute(key, value) { this.attributes[key] = value; }
    removeAttribute(key) { delete this.attributes[key]; }
    focus() {}
    showModal() { this.open = true; }
    close() { this.open = false; }
}
const mode = process.argv[2] || "accept";
const offering = mode === "offer" || mode === "promotion";
const initialGame = {game_id: "draw-game", fen: "before", model_id: "test-model", human_color: "black",
    moves: ["e2e4"], san_moves: ["e4"], status: "active", turn: "human", draw_offer: offering ? null : "white"};
const ids = ["play-app", "play-board", "initial-game", "game-status", "game-detail", "move-list", "move-count",
    "thinking-rail", "keyboard-move-form", "keyboard-move", "play-keyboard-move", "top-player-name", "bottom-player-name",
    "top-player-dot", "bottom-player-dot", "offer-draw", "draw-offer-status", "accept-draw", "decline-draw", "move-start", "move-live",
    "promotion-dialog", "promotion-choices", "promotion-cancel"];
const elements = new Map(ids.map((id) => [id, new Element()]));
elements.get("play-app").dataset = {moveUrl: "/api/play/move", csrfToken: "csrf"};
elements.get("initial-game").textContent = JSON.stringify(initialGame);
let options;
let resolve;
const pending = new Promise((finish) => { resolve = finish; });
const requests = [];
const positions = [];
const source = fs.readFileSync("web/static/js/play.js", "utf8");
vm.runInNewContext(source, {
    console,
    document: {getElementById: (id) => elements.get(id) || null, createElement: () => new Element(), addEventListener() {}},
    fetch(_url, request) { requests.push(JSON.parse(request.body)); return pending; },
    window: {addEventListener() {}, Chess: class {
        constructor(fen) { this.current = fen || "start"; }
        move({from, to, promotion}) { this.current = "after"; return {san: "e5", promotion: mode === "promotion" ? promotion : undefined}; }
        fen() { return this.current; }
    }, Chessboard(_id, config) { options = config; return {orientation() {}, position(fen) { positions.push(fen); }, start() {}, resize() {}}; }},
});
const offer = elements.get("offer-draw");
const accept = elements.get("accept-draw");
const decline = elements.get("decline-draw");
assert.equal(offer.disabled, false);
assert.equal(accept.hidden, offering);
if (mode === "history") {
    elements.get("move-start").listeners.click();
    assert.equal(offer.disabled, true);
    assert.equal(accept.disabled, true);
    accept.listeners.click();
    assert.equal(requests.length, 0, "history positions cannot answer a live offer");
    elements.get("move-live").listeners.click();
    assert.equal(accept.disabled, false);
}
if (offering) {
    offer.checked = true;
    options.onDrop("e7", "e5", "bP");
    if (mode === "promotion") {
        assert.equal(offer.disabled, true);
        assert.equal(requests.length, 0);
        elements.get("promotion-choices").children[3].listeners.click();
    }
    assert.equal(requests[0].offer_draw, true);
    assert.equal(requests[0].move, mode === "promotion" ? "e7e5n" : "e7e5");
} else {
    const button = mode === "decline" ? decline : accept;
    assert.equal(typeof button.listeners.click, "function", "offer has an accessible response control");
    button.listeners.click();
    button.listeners.click();
    assert.equal(requests.length, 1, "busy action cannot submit twice");
    assert.equal(requests[0].action, mode === "decline" ? "decline_draw" : "accept_draw");
    assert.equal("move" in requests[0], false);
    assert.equal(elements.get("thinking-rail").hidden, true, "draw action never reports model thinking");
    assert.equal(positions.at(-1), initialGame.fen);
}
assert.equal(requests[0].game_id, "draw-game");
assert.equal(requests[0].expected_fen, "before");
assert.equal(requests[0].expected_ply, 1);
assert.equal(offer.disabled, true);
assert.equal(accept.disabled, true);
resolve({ok: true, json: async () => ({game: {...initialGame, draw_offer: null,
    ...(offering || mode === "decline" ? {} : {status: "finished", winner: "draw", termination: "draw_agreement"})}})});
setImmediate(() => {
    assert.equal(offer.checked, false);
    assert.equal(accept.hidden, true);
    if (!offering && mode !== "decline") assert.equal(elements.get("game-status").textContent, "Draw by agreement.");
});

const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1";
const E5 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2";
const NF3 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2";
const A = "11111111-1111-4111-8111-111111111111";
const B = "22222222-2222-4222-8222-222222222222";
function snapshot(overrides = {}) {
    return {
        game_id: A, white: "White model", black: "Black model", fen: E5,
        moves: ["e2e4", "e7e5"], san_moves: ["e4", "e5"], turn: "white",
        status: "active", started_at: "2026-10-13T12:00:00+00:00",
        updated_at: "2026-10-13T12:00:10+00:00", heartbeat_at: 1791892810,
        expires_at: 1791894610, winner: null, termination: null,
        illegal_moves_white: 0, illegal_moves_black: 0, result: "*", ...overrides,
    };
}
class Element {
    constructor(tag = "div") {
        this.tagName = tag.toUpperCase();
        this.children = [];
        this.dataset = {};
        this.listeners = {};
        this.attributes = {};
        this.hidden = false;
        this.disabled = false;
        this.value = "";
        this.textContent = "";
        this.className = "";
        this.isContentEditable = false;
    }
    addEventListener(type, callback) { this.listeners[type] = callback; }
    append(...children) { this.children.push(...children); }
    appendChild(child) { this.children.push(child); }
    replaceChildren(...children) { this.children = children; }
    setAttribute(name, value) { this.attributes[name] = String(value); }
    removeAttribute(name) { delete this.attributes[name]; }
    set innerHTML(_value) { throw new Error("Fetched content must use textContent"); }
}
// A bounded chess dependency double: replaying an incorrect UCI or sequence fails.
class Chess {
    constructor(fen = START) { this.position = fen; this.index = 0; }
    fen() { return this.position; }
    load(fen) { this.position = fen; return [START, E4, E5, NF3].includes(fen); }
    move(move) {
        const expected = ["e2e4", "e7e5", "g1f3"][this.index];
        if (move.from + move.to !== expected) return null;
        this.position = [E4, E5, NF3][this.index++];
        return {};
    }
}
const tick = () => new Promise((resolve) => setImmediate(resolve));
function response(games, status = 200) {
    return {ok: status === 200, status, json: async () => ({games})};
}
function harness() {
    const elements = new Map();
    for (const id of ["live-app", "live-message", "live-viewer", "live-selector", "live-board",
        "live-white", "live-black", "live-status", "live-detail", "live-moves",
        "live-start", "live-previous", "live-next", "live-follow", "live-flip",
        "live-position", "live-replay", "live-replay-pending", "live-updated"])
        elements.set(id, new Element(id === "live-selector" ? "select" : "div"));
    elements.get("live-app").dataset = {apiUrl: "/api/live-games", gameApiBase: "/api/game/", replayBase: "/game/"};
    const document = {
        hidden: false, listeners: {},
        getElementById: (id) => elements.get(id),
        createElement: (tag) => new Element(tag),
        addEventListener(type, callback) { this.listeners[type] = callback; },
    };
    const positions = [], requests = [], queue = [], timers = new Map();
    let timerId = 0, flips = 0;
    const window = {
        Chess, listeners: {},
        Chessboard(_id, options) {
            assert.equal(options.draggable, false);
            return {position(fen) { positions.push(fen); }, resize() {}, flip() { flips++; }};
        },
        addEventListener(type, callback) { this.listeners[type] = callback; },
    };
    const context = {
        document, window, console, Date, AbortController,
        setTimeout(callback, delay) { timers.set(++timerId, {callback, delay}); return timerId; },
        clearTimeout(id) { timers.delete(id); },
        fetch(url, options) {
            requests.push({url, options});
            const next = queue.shift();
            if (!next) throw new Error("Unplanned fetch: " + url);
            return typeof next === "function" ? next(options) : Promise.resolve(next);
        },
    };
    return {
        elements, document, positions, requests, queue, timers, window,
        boot() {
            const source = fs.existsSync("web/static/js/live.js") ? fs.readFileSync("web/static/js/live.js", "utf8") : "";
            vm.runInNewContext(source, context);
        },
        async poll() {
            const entry = [...timers].find(([, timer]) => timer.delay === 4000);
            assert.ok(entry, "Viewer must schedule the next poll after the previous request finishes");
            timers.delete(entry[0]);
            entry[1].callback();
            await tick();
        },
        click(id) { elements.get(id).listeners.click(); },
        key(key, target = new Element(), modifiers = {}) {
            let prevented = false;
            document.listeners.keydown({key, target, ...modifiers, preventDefault() { prevented = true; }});
            return prevented;
        },
        get flips() { return flips; },
    };
}

(async () => {
    // Missing board advancement or resetting history on each poll breaks these assertions.
    const h = harness();
    h.queue.push(response([snapshot()]));
    h.boot();
    await tick();
    assert.equal(h.positions.at(-1), E5, "The initial snapshot must show the latest board");
    h.click("live-previous");
    assert.equal(h.positions.at(-1), E4);
    h.queue.push(response([snapshot({moves: ["e2e4", "e7e5", "g1f3"], san_moves: ["e4", "e5", "Nf3"], fen: NF3, turn: "black"})]));
    await h.poll();
    assert.equal(h.positions.at(-1), E4, "A newer snapshot must preserve the chosen history position");
    h.click("live-follow");
    assert.equal(h.positions.at(-1), NF3);
    assert.equal(h.elements.get("live-follow").disabled, true);
    h.queue.push(response([snapshot()]));
    await h.poll();
    assert.equal(h.positions.at(-1), E5, "Following updates the board");
    h.click("live-start");
    assert.equal(h.positions.at(-1), START);
    h.click("live-next");
    assert.equal(h.positions.at(-1), E4);
    h.click("live-flip");
    assert.equal(h.flips, 1);
    assert.equal(h.key("End"), true);
    assert.equal(h.positions.at(-1), E5);
    assert.equal(h.key("ArrowLeft", new Element("button")), true, "Move shortcuts still work after focusing a navigation button");
    assert.equal(h.positions.at(-1), E4);
    h.click("live-follow");
    assert.equal(h.key("ArrowLeft", new Element("select")), false);
    assert.equal(h.key("ArrowLeft", new Element(), {ctrlKey: true}), false);
    const editable = new Element(); editable.isContentEditable = true;
    assert.equal(h.key("ArrowLeft", editable), false);
    assert.equal(h.key("Home"), true);
    assert.equal(h.positions.at(-1), START);
    h.elements.get("live-moves").children[0].children[1].listeners.click();
    assert.equal(h.positions.at(-1), E4, "Move ledger buttons navigate the board");

    // Concurrent game selection and fetched strings cannot become HTML.
    const untrusted = '<img src=x onerror="alert(1)">';
    h.queue.push(response([snapshot(), snapshot({game_id: B, white: untrusted, fen: E4, moves: ["e2e4"], san_moves: [untrusted], turn: "black"})]));
    await h.poll();
    const selector = h.elements.get("live-selector");
    assert.equal(selector.children.length, 2);
    assert.ok(selector.children[1].textContent.includes(untrusted));
    selector.value = B; selector.listeners.change();
    assert.equal(h.positions.at(-1), E4);
    assert.equal(h.elements.get("live-white").textContent, untrusted);
    assert.equal(h.elements.get("live-moves").children[0].children[1].textContent, untrusted);
    h.queue.push(response([snapshot({game_id: B, status: "interrupted"})]));
    await h.poll();
    assert.equal(h.elements.get("live-status").textContent, "Interrupted");
    assert.ok(!h.elements.get("live-detail").textContent.includes("thinking"));
    assert.equal(h.elements.get("live-replay").hidden, true);

    // Error/invalid payloads preserve stale content and never masquerade as an empty list.
    h.queue.push(response(null, 503)); await h.poll();
    assert.equal(h.elements.get("live-message").dataset.state, "error");
    assert.equal(h.elements.get("live-viewer").hidden, false);
    assert.equal(h.elements.get("live-status").textContent, "Connection lost");
    h.queue.push(response([snapshot({moves: "wrong"})])); await h.poll();
    assert.equal(h.elements.get("live-message").dataset.state, "error");
    h.queue.push(response([snapshot({moves: ["a1a1"], san_moves: ["bad"]})])); await h.poll();
    assert.equal(h.elements.get("live-message").dataset.state, "error");
    h.queue.push(response([])); await h.poll();
    assert.equal(h.elements.get("live-message").dataset.state, "empty");
    assert.equal(h.elements.get("live-viewer").hidden, true);

    // A replay must wait for archive publication; success is cached for subsequent polls.
    const terminal = snapshot({status: "finished", winner: "white", result: "1-0", termination: "checkmate"});
    h.queue.push(response([terminal]), response(null, 404)); await h.poll();
    assert.equal(h.elements.get("live-status").textContent, "Finished");
    assert.equal(h.elements.get("live-replay").hidden, true);
    assert.equal(h.elements.get("live-replay-pending").hidden, false);
    h.queue.push(response([terminal]), {ok: true, json: async () => ({game_id: A})}); await h.poll();
    assert.equal(h.elements.get("live-replay").hidden, false);
    assert.equal(h.elements.get("live-replay").href, "/game/" + A);
    assert.equal(h.elements.get("live-replay-pending").hidden, true);
    const previousRequests = h.requests.length;
    h.queue.push(response([terminal])); await h.poll();
    assert.equal(h.requests.length, previousRequests + 1, "Archive success should be cached");
    h.queue.push(response([snapshot({status: "error", termination: "provider error"})])); await h.poll();
    assert.equal(h.elements.get("live-status").textContent, "Error");
    assert.equal(h.elements.get("live-replay").hidden, true);

    // Lifecycle: no overlap across visibility events or slow archive confirmations.
    const slow = harness();
    let resolveFetch;
    slow.queue.push(() => new Promise((resolve) => { resolveFetch = resolve; }));
    slow.boot(); await tick();
    assert.equal(slow.elements.get("live-message").dataset.state, "loading");
    slow.document.hidden = true; slow.document.listeners.visibilitychange();
    slow.document.hidden = false; slow.document.listeners.visibilitychange();
    assert.equal(slow.requests.length, 1, "Restoration must not overlap the in-flight request");
    assert.equal([...slow.timers.values()].filter((timer) => timer.delay === 4000).length, 0);
    resolveFetch(response([snapshot()])); await tick();
    slow.document.hidden = true; slow.document.listeners.visibilitychange();
    assert.equal([...slow.timers.values()].filter((timer) => timer.delay === 4000).length, 0);
    slow.queue.push(response([terminal]), () => new Promise((resolve) => { resolveFetch = resolve; }));
    slow.document.hidden = false; slow.document.listeners.visibilitychange(); await tick();
    assert.equal(slow.requests.length, 3, "Restoration refreshes immediately and confirms the selected archive");
    slow.document.hidden = true; slow.document.listeners.visibilitychange();
    slow.document.hidden = false; slow.document.listeners.visibilitychange();
    assert.equal(slow.requests.length, 3);
    resolveFetch({ok: true}); await tick();
    assert.equal(slow.elements.get("live-replay").hidden, false);
    assert.equal([...slow.timers.values()].filter((timer) => timer.delay === 4000).length, 1);
    // A stalled response body must time out so the viewer can recover on the next poll.
    const stalled = harness();
    stalled.queue.push((options) => Promise.resolve({ok: true, json: () => new Promise((_resolve, reject) => {
        options.signal.addEventListener("abort", () => reject(new Error("body timed out")));
    })}));
    stalled.boot(); await tick();
    const timeout = [...stalled.timers].find(([, timer]) => timer.delay === 15000);
    assert.ok(timeout, "The request timeout must remain active until JSON has been read");
    stalled.timers.delete(timeout[0]); timeout[1].callback(); await tick();
    assert.equal(stalled.elements.get("live-message").dataset.state, "error");
    stalled.queue.push(response([])); await stalled.poll();
    assert.equal(stalled.elements.get("live-message").dataset.state, "empty");
    assert.ok(h.requests.every(({options}) => options.cache === "no-store"));
    console.log("live_games_test: board/history, concurrency, text safety, errors, replay readiness, keyboard and polling lifecycle passed");
})().catch((error) => { console.error(error); process.exitCode = 1; });

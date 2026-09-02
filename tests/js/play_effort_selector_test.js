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
        this.listeners = {};
        this.scrollHeight = 0;
        this.scrollTop = 0;
        this.textContent = "";
        this.value = "";
    }

    addEventListener(type, listener) { this.listeners[type] = listener; }
    append(...children) { this.children.push(...children); }
    appendChild(child) { this.children.push(child); }
    focus() {}
    replaceChildren(...children) { this.children = children; }
}

(async () => {
    const elements = new Map();
    for (const id of [
        "play-board", "game-status", "game-detail", "move-list", "move-count",
        "thinking-rail", "game-setup-form", "start-game", "model-select",
        "keyboard-move-form", "keyboard-move", "play-keyboard-move",
        "top-player-name", "bottom-player-name", "top-player-dot", "bottom-player-dot",
    ]) {
        elements.set(id, new FakeElement());
    }

    const app = new FakeElement();
    app.dataset = {csrfToken: "csrf", moveUrl: "/move", startUrl: "/start"};
    elements.set("play-app", app);
    const initialNode = new FakeElement();
    initialNode.textContent = "null";
    elements.set("initial-game", initialNode);
    const modelsNode = new FakeElement();
    modelsNode.textContent = JSON.stringify([
        {
            id: "quick-model",
            efforts: [{id: "low", name: "Low"}, {id: "high", name: "High"}],
            default_effort: "low",
        },
        {
            id: "reasoner",
            efforts: [{id: "low", name: "Low"}, {id: "high", name: "High"}],
            default_effort: "high",
        },
    ]);
    elements.set("play-models", modelsNode);
    elements.get("model-select").value = "quick-model";

    const effortChoice = new FakeElement();
    const effortLabels = ["default", "none", "minimal", "low", "medium", "high", "xhigh", "max"]
        .map((effort) => {
            const label = new FakeElement();
            label.dataset.effort = effort;
            label.hidden = true;
            label.input = {checked: false, disabled: true, value: effort};
            label.querySelector = () => label.input;
            return label;
        });
    effortChoice.querySelectorAll = () => effortLabels;
    elements.set("effort-choice", effortChoice);

    const documentObject = {
        createElement() { return new FakeElement(); },
        getElementById(id) { return elements.get(id) || null; },
    };
    const windowObject = {
        addEventListener() {},
        Chessboard() {
            return {orientation() {}, position() {}, resize() {}, start() {}};
        },
    };
    let postedBody;
    const source = fs.readFileSync("web/static/js/play.js", "utf8");
    vm.runInNewContext(source, {
        console,
        document: documentObject,
        fetch(_url, options) {
            postedBody = JSON.parse(options.body);
            return Promise.resolve({
                ok: true,
                json: async () => ({
                    game: {
                        fen: "start", human_color: "white", model_id: "reasoner",
                        san_moves: [], status: "active", turn: "human",
                    },
                }),
            });
        },
        FormData: class {
            get(name) {
                return {
                    model_id: "reasoner",
                    human_color: "white",
                    reasoning_effort: "high",
                }[name];
            }
        },
        window: windowObject,
    });

    assert.equal(effortLabels.find((label) => label.dataset.effort === "low").input.checked, true);
    elements.get("model-select").value = "reasoner";
    elements.get("model-select").listeners.change();

    const visibleEfforts = effortLabels.filter((label) => !label.hidden);
    assert.deepEqual(visibleEfforts.map((label) => label.dataset.effort), ["low", "high"]);
    assert.equal(effortLabels.find((label) => label.dataset.effort === "high").input.checked, true);

    await elements.get("game-setup-form").listeners.submit({preventDefault() {}});
    assert.deepEqual(postedBody, {
        model_id: "reasoner",
        human_color: "white",
        reasoning_effort: "high",
    });
})().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});

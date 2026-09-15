const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const template = fs.readFileSync("web/templates/game.html", "utf8");
const navigation = template.slice(template.indexOf("    // Update display\n    function updateDisplay"), template.indexOf("    // Set total moves"));
const listeners = {};
const positionsSeen = [];
let pageScrolls = 0;
let highlights = 0;
const element = {textContent: "", addEventListener() {}};
const document = {
    getElementById() { return element; },
    querySelectorAll() { return []; },
    querySelector() { return {classList: {add() { highlights++; }}, scrollIntoView() { pageScrolls++; }}; },
    addEventListener(type, listener) { listeners[type] = listener; },
};
vm.runInNewContext(`let currentPosition = 0;
    let positions = ['start', 'e4', 'e5'];
    let analysisEnabled = false;
    let flipped = false;
    function toggleAnalysis() {}
    ${navigation}`, {
    document, board: {position(fen) { positionsSeen.push(fen); }},
    movesContainer: element,
});
function press(key, target = {tagName: "BODY"}) {
    let prevented = false;
    listeners.keydown({key, target, preventDefault() { prevented = true; }});
    return prevented;
}
press("ArrowRight");
assert.equal(positionsSeen.at(-1), "e4");
assert.equal(highlights, 1);
assert.equal(pageScrolls, 0, "move navigation must not scroll the page to the score sheet");
assert.equal(press("ArrowRight"), true);
assert.equal(positionsSeen.at(-1), "e5");
assert.equal(press("ArrowLeft"), true);
assert.equal(positionsSeen.at(-1), "e4");
assert.equal(press("Home"), true);
assert.equal(positionsSeen.at(-1), "start");
assert.equal(press("End"), true);
assert.equal(positionsSeen.at(-1), "e5");
assert.equal(press("ArrowRight"), true, "prevent default scrolling even at the last move");
assert.equal(press("ArrowLeft", {tagName: "INPUT"}), false);
assert.equal(positionsSeen.at(-1), "e5");
assert.equal(pageScrolls, 0);

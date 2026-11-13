const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const script = fs.readFileSync('web/static/js/game-analysis.js', 'utf8');
const navigation = script.slice(script.indexOf('    function go(direction)'), script.indexOf("    window.addEventListener('resize'"));
const listeners = {};
const positionsSeen = [];
let scrolls = 0;
const root = {parent: null, children: []};
const middle = {parent: root, children: []};
const last = {parent: middle, children: []};
root.children = [middle]; middle.children = [last];
const document = {addEventListener(type, handler) { listeners[type] = handler; }};
vm.runInNewContext(`let current = root;
    let mainline = [root, middle, last];
    let promotion = null;
    function navigate(node) { current = node; positionsSeen.push(node); }
    ${navigation}`, {
    root, middle, last, positionsSeen, document,
    $: () => ({addEventListener() {}}),
    board: {flip() {}},
    window: {scrollTo() { scrolls++; }},
});
function press(key, typing = false) {
    let prevented = false;
    listeners.keydown({key, target: {closest() { return typing; }}, preventDefault() { prevented = true; }});
    return prevented;
}
assert.equal(press('ArrowRight'), true);
assert.equal(positionsSeen.at(-1), middle);
press('ArrowRight');
assert.equal(positionsSeen.at(-1), last);
press('ArrowLeft');
assert.equal(positionsSeen.at(-1), middle);
press('Home');
assert.equal(positionsSeen.at(-1), root);
press('End');
assert.equal(positionsSeen.at(-1), last);
assert.equal(press('ArrowRight'), true);
assert.equal(press('ArrowLeft', true), false);
assert.equal(positionsSeen.at(-1), last);
assert.equal(scrolls, 0);

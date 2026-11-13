const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const script = fs.readFileSync('web/static/js/game-analysis.js', 'utf8');
const block = script.slice(script.indexOf('    let suppressBoardClick'), script.indexOf("    $('analysis-promotion').addEventListener"));
const handlers = {}, documentHandlers = {};
const squares = ['e2', 'e4', 'e7'].map(square => ({dataset: {square}, classes: new Set(), classList: {
    toggle(name, active) { if (active) this.owner.classes.add(name); else this.owner.classes.delete(name); },
}}));
squares.forEach(s => s.classList.owner = s);
const boardElement = {querySelectorAll: () => squares, addEventListener: (event, handler) => { handlers[event] = handler; }};
let config, moves = 0;
const context = vm.createContext({
    $: id => id === 'notation-menu' ? {hidden: true} : boardElement,
    root: {fen: 'start'}, current: {fen: 'start'}, promotion: null, selectedSquare: null,
    Chessboard(id, options) { config = options; return {position() {}}; },
    Chess: class {
        turn() { return 'w'; }
        moves({square}) { return square === 'e2' ? [{to: 'e4'}] : []; }
        get(square) { return square === 'e2' ? {color: 'w'} : null; }
    },
    tryMove(from, to) { if (from === 'e2' && to === 'e4') { moves++; return true; } return false; },
    closeMenu() {},
    document: {addEventListener: (event, handler) => { documentHandlers[event] = handler; }},
});
vm.runInContext(block, context);
const count = cls => squares.filter(s=>s.classes.has(cls)).length;
config.onDrop('e2','e2');
assert.equal(count('board-selected'), 1);
assert.equal(count('board-legal'), 1);
config.onDrop('e2','e2');
assert.equal(count('board-selected'), 0, 'same-square click deselects');
assert.equal(count('board-legal'), 0);
config.onDrop('e2','e2');
config.onDrop('e2','e4');
assert.equal(moves, 1);
assert.equal(count('board-selected'), 0, 'moving clears selection');
config.onDrop('e2','e2');
documentHandlers.keydown({key: 'Escape'});
assert.equal(count('board-selected'), 0, 'Escape clears selection');
config.onDrop('e2','e2');
documentHandlers.pointerdown({target: {closest: () => null}});
assert.equal(count('board-selected'), 0, 'outside click clears selection');
config.onDrop('e2','e2');
config.onDrop('e2','offboard');
assert.equal(count('board-selected'), 0, 'invalid drag clears selection');
console.log('Board selection cancellation and move cleanup passed');

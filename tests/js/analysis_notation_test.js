const assert = require('node:assert/strict');
const {notationTokens, variationRoot, promoteVariation, deleteVariation, notationPgn} = require('../../web/static/js/analysis-notation.js');
let id = 0;
function node(parent, san, color, number, original = true) {
    const n = {id: id++, parent, move: {san, color}, fen: `position ${color === 'w' ? 'b' : 'w'} - - 0 ${color === 'b' ? number + 1 : number}`, original, children: []};
    if(parent) parent.children.push(n);
    return n;
}
const root = {id: id++, fen: 'position w - - 0 1', children: []};
const e4=node(root,'e4','w',1), e5=node(e4,'e5','b',1), nf3=node(e5,'Nf3','w',2);
const c5=node(e4,'c5','b',1,false), nc3=node(c5,'Nc3','w',2,false);
const d6=node(c5,'d6','w',2,false); // Notation traversal must preserve nested sibling order.
let tokens=notationTokens(root);
assert.deepEqual(tokens.filter(t=>t.type==='move').map(t=>t.node.move.san), ['e4','e5','c5','Nc3','d6','Nf3']);
const text=tokens.map(t=>t.text || '').join(' ').replace(/\s+/g,' ');
assert.match(text,/1\. e4 e5 \( 1\.\.\. c5 2\. Nc3 \( 2\. d6 \) \) 2\. Nf3/);
assert.equal(variationRoot(nc3), c5);
assert.equal(variationRoot(d6), d6);
assert.equal(notationPgn(root), '1. e4 e5 (1... c5 2. Nc3 (2. d6)) 2. Nf3 *');
promoteVariation(nc3);
assert.equal(e4.children[0], c5);
assert.equal(e4.children[1], e5);
assert.equal(deleteVariation(nc3), e4);
assert.deepEqual(e4.children, [e5]);
assert.equal(notationTokens(root).filter(t=>t.type==='move').length, 3);
console.log('Inline and nested notation, promotion, deletion and PGN passed');

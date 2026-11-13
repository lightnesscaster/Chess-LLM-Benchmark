/* Traverse a chess variation tree in standard recursive annotation order. */
(function (root) {
    'use strict';
    function notationTokens(rootNode) {
        const tokens = [];
        function line(start, mainline, includeSiblings = true) {
            let node = start;
            let forceNumber = true;
            while (node) {
                const number = node.parent.fen.split(' ')[5];
                const prefix = node.move.color === 'w' ? number + '. ' : forceNumber ? number + '... ' : '';
                tokens.push({type: 'move', node, mainline, text: prefix + node.move.san});
                const siblings = includeSiblings ? node.parent.children.filter(child => child !== node) : [];
                for (const sibling of siblings) {
                    tokens.push({type: 'open', node: sibling, text: '('});
                    line(sibling, false, false);
                    tokens.push({type: 'close', node: sibling, text: ')'});
                }
                forceNumber = siblings.length > 0;
                node = node.children[0];
                includeSiblings = true;
            }
        }
        if (rootNode.children.length) line(rootNode.children[0], true);
        return tokens;
    }
    function variationRoot(node) {
        let edited = null;
        for (; node.parent; node = node.parent) {
            if (node.parent.children[0] !== node) return node;
            if (!node.original) edited = node;
        }
        return edited;
    }
    function promoteVariation(node) {
        const branch = variationRoot(node);
        if (!branch) return;
        const siblings = branch.parent.children;
        siblings.splice(siblings.indexOf(branch), 1);
        siblings.unshift(branch);
    }
    function deleteVariation(node) {
        const branch = variationRoot(node);
        if (!branch) return null;
        const parent = branch.parent;
        parent.children.splice(parent.children.indexOf(branch), 1);
        parent.preferredChild = parent.children[0];
        return parent;
    }
    function notationPgn(rootNode) {
        return notationTokens(rootNode).map(token => token.text).join(' ')
            .replace(/\( /g, '(').replace(/ \)/g, ')') + ' *';
    }
    const api = {notationTokens, variationRoot, promoteVariation, deleteVariation, notationPgn};
    if (typeof module !== 'undefined') module.exports = api;
    else root.AnalysisNotation = api;
})(typeof window !== 'undefined' ? window : globalThis);

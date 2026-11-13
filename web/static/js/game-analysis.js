/* Local analysis branches belong to this viewer, independent of the recorded game. */
document.addEventListener('DOMContentLoaded', () => {
    'use strict';
    const $ = id => document.getElementById(id);
    const pgn = JSON.parse($('game-pgn').textContent);
    const chess = new Chess();
    if (!chess.load_pgn(pgn, {sloppy: true})) {
        $('moves-container').textContent = 'Unable to read this game’s PGN.';
        return;
    }
    const history = chess.history({verbose: true});
    while (chess.undo()) { /* Preserve the PGN's starting position. */ }
    let nextId = 0;
    const root = {id: nextId++, fen: chess.fen(), parent: null, children: [], ply: 0};
    const mainline = [root];
    const nodes = new Map([[root.id, root]]);
    let tail = root;
    for (const move of history) {
        const played = chess.move(move);
        const node = {id: nextId++, fen: chess.fen(), parent: tail, children: [],
            move: played, ply: tail.ply + 1, original: true};
        nodes.set(node.id, node);
        tail.children.push(node);
        mainline.push(node);
        tail = node;
    }
    let current = root;
    let selectedSquare = null;
    let promotion = null;
    let enabled = false;
    let worker = null;
    let controller = null;
    let generation = 0;
    let initialization = null;
    let searchTimer = null;
    let lines = [];
    let currentKey = null;
    const cache = new Map();

    function path(node) {
        const result = [];
        for (; node.parent; node = node.parent) result.unshift(node);
        return result;
    }
    function uci(move) { return move.from + move.to + (move.promotion || ''); }
    function target() {
        const moves = path(current).map(node => uci(node.move));
        const command = 'position fen ' + root.fen + (moves.length ? ' moves ' + moves.join(' ') : '');
        return {key: command, command, fen: current.fen};
    }
    function moveLabel(node) {
        const parts = node.parent.fen.split(' ');
        return parts[5] + (parts[1] === 'w' ? '. ' : '… ') + node.move.san;
    }
    function addMove(from, to, piece) {
        const position = new Chess(current.fen);
        const move = position.move({from, to, promotion: piece || 'q'});
        if (!move) return false;
        let child = current.children.find(node => uci(node.move) === uci(move));
        if (!child) {
            child = {id: nextId++, fen: position.fen(), parent: current, children: [],
                move, ply: current.ply + 1, original: false};
            current.children.push(child);
            nodes.set(child.id, child);
        }
        navigate(child);
        return true;
    }
    $('analysis-move-form').addEventListener('submit', event => {
        event.preventDefault();
        const position = new Chess(current.fen);
        const move = position.move($('analysis-move').value.trim(), {sloppy: true});
        if (!move) {
            $('analysis-move-error').textContent = 'Enter a legal move for this position.';
            return;
        }
        addMove(move.from, move.to, move.promotion);
        $('analysis-move').value = '';
    });
    $('copy-analysis-pgn').addEventListener('click', async () => {
        const line = new Chess(root.fen);
        path(current).forEach(node => line.move(node.move));
        try {
            await navigator.clipboard.writeText(line.pgn());
            $('copy-analysis-pgn').textContent = 'Copied current line';
        } catch (_) { $('copy-analysis-pgn').textContent = 'Copy failed'; }
    });
    function tryMove(from, to) {
        if (promotion) return false;
        const position = new Chess(current.fen);
        const choices = position.moves({square: from, verbose: true}).filter(move => move.to === to);
        if (!choices.length) return false;
        if (choices.some(move => move.promotion)) {
            promotion = {from, to};
            $('analysis-promotion').hidden = false;
            $('analysis-promotion').querySelector('button').focus();
        } else addMove(from, to);
        return true;
    }
    const board = Chessboard('board', {
        position: root.fen, draggable: true,
        pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
        onDragStart(source, piece) {
            return !promotion && piece[0] === new Chess(current.fen).turn();
        },
        onDrop(from, to) {
            if (from === to) {
                selectedSquare = from;
                highlightSelection();
                return 'snapback';
            }
            selectedSquare = null;
            if (!tryMove(from, to) || promotion) return 'snapback';
        },
        onSnapEnd() { board.position(current.fen); },
    });
    function highlightSelection() {
        $('board').querySelectorAll('[data-square]').forEach(el => {
            el.classList.toggle('analysis-selected', el.dataset.square === selectedSquare);
        });
    }
    $('board').addEventListener('click', event => {
        const square = event.target.closest('[data-square]')?.dataset.square;
        if (!square || promotion) return;
        if (selectedSquare && selectedSquare !== square && tryMove(selectedSquare, square)) {
            selectedSquare = null;
        } else {
            const piece = new Chess(current.fen).get(square);
            selectedSquare = piece?.color === new Chess(current.fen).turn() ? square : null;
        }
        highlightSelection();
    });
    $('analysis-promotion').addEventListener('click', event => {
        const choice = event.target.closest('button')?.dataset.piece;
        if (choice === undefined || !promotion) return;
        const {from, to} = promotion;
        promotion = null;
        $('analysis-promotion').hidden = true;
        if (choice) addMove(from, to, choice);
        board.position(current.fen);
    });
    function nodeButton(node) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'variation-move' + (node === current ? ' current' : '');
        button.textContent = moveLabel(node);
        button.dataset.node = node.id;
        return button;
    }
    function renderBranches() {
        const container = $('analysis-variations');
        container.replaceChildren();
        // A row per analysis leaf preserves all forks while keeping the recorded score sheet familiar.
        const leaves = [...nodes.values()].filter(node => !node.original && node.parent && !node.children.length);
        for (const leaf of leaves) {
            const row = document.createElement('div');
            row.className = 'variation-line';
            path(leaf).filter(node => !node.original).forEach(node => row.append(nodeButton(node)));
            container.append(row);
        }
        $('analysis-branches').hidden = leaves.length === 0;
        $('btn-return-game').hidden = mainline.includes(current);
        $('analysis-position-status').textContent = mainline.includes(current) ? '' : 'Analysis variation';
    }
    function navigate(node) {
        if (node.parent) node.parent.preferredChild = node;
        current = node;
        $('analysis-move-error').textContent = '';
        selectedSquare = null;
        promotion = null;
        $('analysis-promotion').hidden = true;
        board.position(current.fen);
        $('current-move').textContent = Math.ceil(current.ply / 2);
        document.querySelectorAll('.move[data-node]').forEach(el => el.classList.toggle('current', Number(el.dataset.node) === current.id));
        renderBranches();
        if (enabled) scheduleAnalysis();
    }
    let html = '<table class="moves-table"><tbody>';
    for (let i = 1; i < mainline.length; i++) {
        const node = mainline[i];
        const number = node.parent.fen.split(' ')[5];
        if (node.move.color === 'w') html += '<tr><td class="move-num">' + number + '.</td>';
        else if (i === 1) html += '<tr><td class="move-num">' + number + '.</td><td></td>';
        html += '<td class="move" data-node="' + node.id + '"><button type="button" data-node="' + node.id + '">' + node.move.san + '</button></td>';
        if (node.move.color === 'b') html += '</tr>';
    }
    if (mainline.at(-1).move?.color === 'w') html += '<td></td></tr>';
    $('moves-container').innerHTML = html + '</tbody></table>';
    $('total-moves').textContent = Math.ceil(history.length / 2);
    for (const id of ['moves-container', 'analysis-variations']) {
        $(id).addEventListener('click', event => {
            const id = event.target.closest('[data-node]')?.dataset.node;
            if (id !== undefined) navigate(nodes.get(Number(id)));
        });
    }
    function go(direction) {
        if (direction === 'start') navigate(root);
        if (direction === 'prev' && current.parent) navigate(current.parent);
        if (direction === 'next' && current.children.length) navigate(current.preferredChild || current.children[0]);
        if (direction === 'end') {
            let last = current;
            while (last.children.length) last = last.preferredChild || last.children[0];
            navigate(last);
        }
    }
    for (const direction of ['start', 'prev', 'next', 'end']) {
        $('btn-' + direction).addEventListener('click', () => go(direction));
    }
    $('btn-return-game').addEventListener('click', () => {
        let node = current;
        while (!mainline.includes(node)) node = node.parent;
        node.preferredChild = node.children.find(child => child.original);
        navigate(node);
    });
    $('btn-flip').addEventListener('click', () => board.flip());
    document.addEventListener('keydown', event => {
        if (event.target.closest('input, textarea, select, button, [contenteditable="true"]') || promotion) return;
        const direction = {ArrowLeft: 'prev', ArrowRight: 'next', Home: 'start', End: 'end'}[event.key];
        if (direction) { event.preventDefault(); go(direction); }
    });
    window.addEventListener('resize', () => board.resize());

    function updateScore(line) {
        const side = current.fen.split(' ')[1] === 'w' ? 1 : -1;
        const score = line.cp === null ? null : line.cp * side;
        const mate = line.mate === null ? null : line.mate * side;
        const text = score === null ? (mate >= 0 ? 'M' : '-M') + Math.abs(mate) : (score > 0 ? '+' : '') + (score / 100).toFixed(2);
        const whiteWinsMate = line.mate !== null && (line.mate > 0 ? side > 0 : side < 0);
        return {text, percent: score === null ? (whiteWinsMate ? 100 : 0) : Math.max(0, Math.min(100, 50 + score / 20))};
    }
    function displayLines() {
        const container = $('analysis-lines');
        container.replaceChildren();
        const best = lines[0];
        $('analysis-depth').textContent = best ? 'Depth ' + best.depth : 'Analyzing…';
        if (best) {
            const score = updateScore(best);
            $('eval-score').textContent = score.text;
            $('eval-bar-fill').style.height = score.percent + '%';
        }
        for (const line of lines.filter(Boolean)) {
            const row = document.createElement('div');
            row.className = 'analysis-line';
            const score = document.createElement('span');
            score.className = 'line-score';
            score.textContent = updateScore(line).text;
            row.append(score);
            const pv = document.createElement('span');
            pv.className = 'line-pv';
            const position = new Chess(current.fen);
            const sequence = [];
            for (const move of line.moves.slice(0, 16)) {
                const number = position.fen().split(' ')[5];
                const color = position.turn();
                const played = position.move({from: move.slice(0, 2), to: move.slice(2, 4), promotion: move[4] || 'q'});
                if (!played) break;
                sequence.push(move);
                const button = document.createElement('button');
                button.type = 'button';
                button.textContent = (color === 'w' ? number + '. ' : sequence.length === 1 ? number + '… ' : '') + played.san;
                button.title = 'Explore line through ' + played.san;
                const steps = [...sequence];
                button.addEventListener('click', () => {
                    for (const step of steps) addMove(step.slice(0, 2), step.slice(2, 4), step[4]);
                });
                pv.append(button);
            }
            row.append(pv);
            container.append(row);
        }
    }
    function showError(message) {
        $('analysis-lines').textContent = message + ' Toggle analysis off and on to retry.';
        $('analysis-depth').textContent = 'Engine error';
    }
    async function initEngine(version) {
        const base = 'https://unpkg.com/stockfish@16.0.0/src/';
        const response = await fetch(base + 'stockfish-nnue-16-single.js');
        if (!response.ok) throw new Error('Engine download failed.');
        const script = await response.text();
        if (version !== generation || !enabled) return;
        // The upstream worker accepts an absolute WASM location in its URL hash.
        const blob = URL.createObjectURL(new Blob([script], {type: 'application/javascript'}));
        const instance = new Worker(blob + '#' + encodeURIComponent(base + 'stockfish-nnue-16-single.wasm'));
        worker = instance;
        controller = new AnalysisEngine(instance, info => {
            if (info.target.key !== currentKey || info.index < 1 || info.index > 3) return;
            if (lines[info.index - 1]?.depth > info.depth) return;
            lines[info.index - 1] = info;
            cache.set(currentKey, [...lines]);
            if (cache.size > 100) cache.delete(cache.keys().next().value);
            displayLines();
        }, () => { $('analysis-depth').textContent += ' · Complete'; });
        await new Promise((resolve, reject) => {
            const timer = setTimeout(() => reject(new Error('Engine loading timed out.')), 60000);
            instance.onmessage = event => {
                if (version !== generation) return;
                for (const line of String(event.data).split('\n')) {
                    if (line === 'uciok') {
                        instance.postMessage('setoption name Hash value 64');
                        instance.postMessage('setoption name MultiPV value 3');
                        instance.postMessage('setoption name UCI_AnalyseMode value true');
                        instance.postMessage('isready');
                    }
                    controller.receive(line);
                    if (line === 'readyok') { clearTimeout(timer); URL.revokeObjectURL(blob); resolve(); }
                }
            };
            instance.onerror = () => {
                clearTimeout(timer);
                URL.revokeObjectURL(blob);
                instance.terminate();
                showError('Stockfish stopped unexpectedly.');
                reject(new Error('Engine failed to load.'));
            };
            instance.postMessage('uci');
        }).finally(() => URL.revokeObjectURL(blob));
    }
    function scheduleAnalysis() {
        clearTimeout(searchTimer);
        const next = target();
        if (next.key === currentKey && controller?.target?.key === next.key) return;
        currentKey = next.key;
        controller?.stop();
        lines = cache.get(next.key) || [];
        $('eval-score').textContent = '…';
        $('eval-bar-fill').style.height = '50%';
        displayLines();
        const position = new Chess(current.fen);
        if (position.game_over()) {
            $('analysis-depth').textContent = position.in_checkmate() ? 'Checkmate' : 'Draw';
            $('analysis-lines').textContent = 'Game over in this position.';
            $('eval-bar-fill').style.height = position.in_checkmate() ? (position.turn() === 'w' ? '0%' : '100%') : '50%';
            $('eval-score').textContent = position.in_checkmate() ? (position.turn() === 'w' ? '-M0' : 'M0') : '0.00';
            return;
        }
        searchTimer = setTimeout(() => { if (enabled) controller?.request(next); }, 120);
    }
    $('btn-analysis').addEventListener('click', async () => {
        enabled = !enabled;
        const version = ++generation;
        $('analysis-panel').style.display = enabled ? 'block' : 'none';
        $('btn-analysis').classList.toggle('active', enabled);
        $('btn-analysis').setAttribute('aria-expanded', String(enabled));
        if (!enabled) {
            clearTimeout(searchTimer);
            controller?.stop();
            worker?.terminate();
            worker = controller = initialization = null;
            currentKey = null;
            return;
        }
        $('analysis-lines').textContent = 'Loading Stockfish…';
        try {
            initialization = initEngine(version);
            await initialization;
            if (enabled && version === generation) scheduleAnalysis();
        } catch (error) {
            if (version === generation) {
                worker?.terminate();
                controller = worker = null;
                showError(error.message);
            }
        }
    });
    window.addEventListener('beforeunload', () => worker?.terminate());
    $('pgn-toggle').addEventListener('click', () => {
        const open = $('pgn-content').style.display === 'none';
        $('pgn-content').style.display = open ? 'block' : 'none';
        $('pgn-toggle').classList.toggle('expanded', open);
        $('pgn-toggle').setAttribute('aria-expanded', String(open));
    });
    $('copy-pgn').addEventListener('click', async () => {
        try { await navigator.clipboard.writeText(pgn); $('copy-pgn').textContent = 'Copied'; }
        catch (_) { $('copy-pgn').textContent = 'Copy failed'; }
    });
    navigate(root);
});

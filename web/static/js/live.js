(() => {
    "use strict";

    const app = document.getElementById("live-app");
    if (!app) return;
    const element = (id) => document.getElementById(`live-${id}`);
    const message = element("message");
    const viewer = element("viewer");
    const selector = element("selector");
    const replay = element("replay");
    const pendingReplay = element("replay-pending");
    let games = [];
    let selectedId = null;
    let position = 0;
    let following = true;
    let connected = false;
    let inFlight = false;
    let pollTimer = null;
    let stopped = false;
    let selectorKey = "";
    let ledgerKey = "";
    let ledgerGameId = null;
    let moveButtons = [];
    const archivedGames = new Set();

    function showMessage(state, text) {
        message.dataset.state = state;
        message.textContent = text;
        message.hidden = !text;
    }

    showMessage("loading", "Loading live games…");
    viewer.hidden = true;
    if (!window.Chess || !window.Chessboard) {
        showMessage("error", "The chess viewer could not load. Refresh the page to try again.");
        return;
    }
    const board = window.Chessboard("live-board", {
        position: "start",
        draggable: false,
        moveSpeed: 0,
        appearSpeed: 0,
        trashSpeed: 0,
        pieceTheme: "https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png",
    });

    // Validate the entire envelope before replacing a previously usable snapshot.
    function prepareGames(payload) {
        if (!payload || !Array.isArray(payload.games)) throw new Error("Invalid live envelope");
        const ids = new Set();
        return payload.games.map((game) => {
            const isText = (value) => typeof value === "string";
            const isCounter = (value) => Number.isInteger(value) && value >= 0;
            if (!game || !isText(game.game_id) || !/^[A-Za-z0-9_-]+$/.test(game.game_id)
                || ids.has(game.game_id) || !isText(game.white) || !isText(game.black)
                || !isText(game.fen) || !Array.isArray(game.moves) || !Array.isArray(game.san_moves)
                || game.moves.length !== game.san_moves.length
                || !game.moves.every((move) => isText(move) && /^[a-h][1-8][a-h][1-8][qrbn]?$/.test(move))
                || !game.san_moves.every(isText)
                || !["active", "finished", "error", "interrupted"].includes(game.status)
                || !["white", "black"].includes(game.turn)
                || !["white", "black", "draw", null].includes(game.winner)
                || !(game.termination === null || isText(game.termination))
                || !(game.draw_offer === undefined || game.draw_offer === null
                    || ["white", "black"].includes(game.draw_offer) && game.draw_offer !== game.turn)
                || !["*", "1-0", "0-1", "1/2-1/2"].includes(game.result)
                || !isCounter(game.illegal_moves_white) || !isCounter(game.illegal_moves_black)
                || !isText(game.started_at) || !Number.isFinite(Date.parse(game.started_at))
                || !isText(game.updated_at) || !Number.isFinite(Date.parse(game.updated_at))
                || !Number.isFinite(game.heartbeat_at) || !Number.isFinite(game.expires_at)) {
                throw new Error("Invalid live snapshot");
            }
            ids.add(game.game_id);
            const validator = new window.Chess();
            if (!validator.load(game.fen)) throw new Error("Invalid board position");
            const chess = new window.Chess();
            const positions = [chess.fen()];
            for (const uci of game.moves) {
                const move = {from: uci.slice(0, 2), to: uci.slice(2, 4)};
                if (uci.length === 5) move.promotion = uci[4];
                if (!chess.move(move)) throw new Error("Invalid move history");
                positions.push(chess.fen());
            }
            // The supplied FEN is authoritative for the latest public position.
            positions[positions.length - 1] = game.fen;
            return {...game, positions};
        });
    }

    const selectedGame = () => games.find((game) => game.game_id === selectedId);
    const statusLabel = (game) => ({active: "Live", finished: "Finished", interrupted: "Interrupted", error: "Error"})[game.status];

    function render() {
        const game = selectedGame();
        viewer.hidden = !game;
        if (!game) return;
        const nextSelectorKey = JSON.stringify(games.map((entry) => [entry.game_id, entry.white, entry.black, entry.status]));
        if (nextSelectorKey !== selectorKey) {
            selector.replaceChildren();
            for (const entry of games) {
                const option = document.createElement("option");
                option.value = entry.game_id;
                option.textContent = `${entry.white} vs ${entry.black} — ${statusLabel(entry)}`;
                selector.append(option);
            }
            selectorKey = nextSelectorKey;
        }
        selector.value = selectedId;
        element("white").textContent = game.white;
        element("black").textContent = game.black;
        element("status").textContent = connected ? statusLabel(game) : "Connection lost";
        element("status").dataset.status = connected ? game.status : "disconnected";
        let detail;
        if (!connected) detail = "Showing the last received position. Reconnecting automatically.";
        else if (game.status === "active") {
            const offer = game.draw_offer
                ? `${game.draw_offer === "white" ? game.white : game.black} offered a draw. ` : "";
            detail = offer + `${game.turn === "white" ? game.white : game.black} is thinking.`;
        }
        else if (game.status === "interrupted") detail = "Updates stopped. This game may have been interrupted.";
        else if (game.status === "error") detail = `Game stopped${game.termination ? `: ${game.termination}` : "."}`;
        else if (game.termination === "draw_agreement") detail = `${game.result} · Draw by agreement`;
        else detail = `${game.result}${game.termination ? ` · ${game.termination}` : ""}`;
        const illegalCount = game.illegal_moves_white + game.illegal_moves_black;
        element("detail").textContent = detail + (illegalCount ? ` Illegal moves: White ${game.illegal_moves_white}, Black ${game.illegal_moves_black}.` : "");
        element("updated").textContent = `Last update: ${new Date(game.updated_at).toLocaleTimeString()}`;
        board.position(game.positions[position], false);
        element("position").textContent = `${following ? "Latest" : "History"} · ${position} / ${game.moves.length} plies`;
        element("start").disabled = position === 0;
        element("previous").disabled = position === 0;
        element("next").disabled = position === game.moves.length;
        element("follow").disabled = following;

        const nextLedgerKey = JSON.stringify([game.game_id, game.san_moves]);
        if (nextLedgerKey !== ledgerKey) {
            const ledger = element("moves");
            const focusedMove = ledgerGameId === game.game_id ? moveButtons.indexOf(document.activeElement) : -1;
            ledger.replaceChildren();
            moveButtons = [];
            if (!game.moves.length) {
                const empty = document.createElement("p");
                empty.textContent = "Waiting for the first move.";
                ledger.append(empty);
            }
            for (let i = 0; i < game.san_moves.length; i += 2) {
                const row = document.createElement("div");
                row.className = "live-move-row";
                const number = document.createElement("span");
                number.textContent = `${i / 2 + 1}.`;
                row.append(number);
                for (let j = i; j < Math.min(i + 2, game.san_moves.length); j++) {
                    const button = document.createElement("button");
                    button.type = "button";
                    button.textContent = game.san_moves[j];
                    button.setAttribute("aria-label", `${j % 2 === 0 ? "White" : "Black"} move ${Math.floor(j / 2) + 1}: ${game.san_moves[j]}`);
                    button.addEventListener("click", () => navigate(j + 1));
                    row.append(button);
                    moveButtons.push(button);
                }
                ledger.append(row);
            }
            ledgerKey = nextLedgerKey;
            ledgerGameId = game.game_id;
            if (focusedMove >= 0 && moveButtons[focusedMove]) moveButtons[focusedMove].focus({preventScroll: true});
        }
        moveButtons.forEach((button, index) => {
            button.setAttribute("aria-current", index + 1 === position ? "step" : "false");
        });
        const finished = game.status === "finished";
        const ready = finished && archivedGames.has(game.game_id);
        replay.hidden = !ready;
        pendingReplay.hidden = !finished || ready;
        if (ready) replay.href = app.dataset.replayBase + encodeURIComponent(game.game_id);
        else replay.removeAttribute("href");
    }

    function navigate(index, follow = false) {
        const game = selectedGame();
        if (!game) return;
        position = Math.max(0, Math.min(index, game.moves.length));
        following = follow || position === game.moves.length;
        render();
    }

    async function request(url, consume = (response) => response) {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 15000);
        try {
            return await consume(await fetch(url, {cache: "no-store", signal: controller.signal}));
        } finally {
            clearTimeout(timeout);
        }
    }

    async function confirmReplay() {
        const game = selectedGame();
        if (!game || game.status !== "finished" || archivedGames.has(game.game_id)) return;
        try {
            const response = await request(app.dataset.gameApiBase + encodeURIComponent(game.game_id));
            if (response.ok) archivedGames.add(game.game_id);
        } catch (_error) {
            // The normal archive can become available after the terminal snapshot.
        }
        if (!stopped) render();
    }

    async function poll() {
        if (stopped || document.hidden || inFlight) return;
        inFlight = true;
        clearTimeout(pollTimer);
        try {
            const payload = await request(app.dataset.apiUrl, (response) => {
                if (!response.ok) throw new Error("Live games unavailable");
                return response.json();
            });
            const nextGames = prepareGames(payload);
            games = nextGames;
            if (!selectedGame()) {
                selectedId = games.length ? games[0].game_id : null;
                following = true;
            }
            const game = selectedGame();
            if (game) position = following ? game.moves.length : Math.min(position, game.moves.length);
            connected = true;
            showMessage(games.length ? "ready" : "empty", games.length ? "" : "No live games right now. Saved benchmark games will appear here when they start.");
            render();
            board.resize();
            await confirmReplay();
        } catch (_error) {
            connected = false;
            showMessage("error", "Could not connect to live games. Retrying automatically.");
            render();
        } finally {
            inFlight = false;
            if (!stopped && !document.hidden) pollTimer = setTimeout(poll, 4000);
        }
    }

    selector.addEventListener("change", () => {
        selectedId = selector.value;
        following = true;
        const game = selectedGame();
        if (game) position = game.moves.length;
        render();
    });
    element("start").addEventListener("click", () => navigate(0));
    element("previous").addEventListener("click", () => navigate(position - 1));
    element("next").addEventListener("click", () => navigate(position + 1));
    element("follow").addEventListener("click", () => {
        const game = selectedGame();
        if (game) navigate(game.moves.length, true);
    });
    element("flip").addEventListener("click", () => board.flip());
    document.addEventListener("keydown", (event) => {
        if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey
            || event.target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)
            || !selectedGame()) return;
        const actions = {
            ArrowLeft: () => navigate(position - 1),
            ArrowRight: () => navigate(position + 1),
            Home: () => navigate(0),
            End: () => navigate(selectedGame().moves.length, true),
        };
        if (actions[event.key]) {
            event.preventDefault();
            actions[event.key]();
        }
    });
    document.addEventListener("visibilitychange", () => {
        clearTimeout(pollTimer);
        if (!document.hidden) poll();
    });
    window.addEventListener("resize", () => board.resize());
    window.addEventListener("pagehide", () => {
        stopped = true;
        clearTimeout(pollTimer);
    });
    window.addEventListener("pageshow", () => {
        if (stopped) {
            stopped = false;
            poll();
        }
    });
    poll();
})();

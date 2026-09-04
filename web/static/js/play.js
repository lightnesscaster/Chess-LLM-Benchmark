(function () {
    "use strict";

    const app = document.getElementById("play-app");
    const boardElement = document.getElementById("play-board");
    if (!app || !boardElement || typeof window.Chessboard !== "function") return;

    const statusElement = document.getElementById("game-status");
    const detailElement = document.getElementById("game-detail");
    const moveList = document.getElementById("move-list");
    const moveCount = document.getElementById("move-count");
    const moveStartButton = document.getElementById("move-start");
    const movePreviousButton = document.getElementById("move-previous");
    const moveNextButton = document.getElementById("move-next");
    const moveLiveButton = document.getElementById("move-live");
    const movePosition = document.getElementById("move-position");
    const thinkingRail = document.getElementById("thinking-rail");
    const setupForm = document.getElementById("game-setup-form");
    const startButton = document.getElementById("start-game");
    const modelSelect = document.getElementById("model-select");
    const modelsNode = document.getElementById("play-models");
    const effortChoice = document.getElementById("effort-choice");
    const initialNode = document.getElementById("initial-game");
    const keyboardMoveForm = document.getElementById("keyboard-move-form");
    const keyboardMoveInput = document.getElementById("keyboard-move");
    const keyboardMoveButton = document.getElementById("play-keyboard-move");
    const fenValue = document.getElementById("fen-value");
    const copyFenButton = document.getElementById("copy-fen");
    const copyPgnButton = document.getElementById("copy-pgn");
    const downloadPgnButton = document.getElementById("download-pgn");
    const exportFeedback = document.getElementById("export-feedback");

    let game = initialNode ? JSON.parse(initialNode.textContent || "null") : null;
    const playableModels = modelsNode ? JSON.parse(modelsNode.textContent || "[]") : [];
    let busy = false;
    let pendingFen = null;
    let viewedPly = null;
    let ledgerMoveButtons = [];
    let board;

    function totalPlies() {
        return game && Array.isArray(game.moves) ? game.moves.length : 0;
    }

    function currentViewedPly() {
        return viewedPly === null ? totalPlies() : viewedPly;
    }

    function isViewingLive() {
        return viewedPly === null;
    }

    function fenAtPly(ply) {
        if (!game || !Array.isArray(game.moves) || typeof window.Chess !== "function") {
            return game ? game.fen : null;
        }
        const replay = new window.Chess();
        for (const uciMove of game.moves.slice(0, ply)) {
            const move = replay.move({
                from: uciMove.slice(0, 2),
                to: uciMove.slice(2, 4),
                promotion: uciMove.slice(4) || "q",
            });
            if (!move) return game.fen;
        }
        return replay.fen();
    }

    function displayedFen() {
        if (!game) return null;
        if (isViewingLive()) return pendingFen || game.fen;
        return fenAtPly(viewedPly);
    }

    function humanPiece(piece) {
        if (!game) return false;
        return game.human_color === "white" ? piece.startsWith("w") : piece.startsWith("b");
    }

    function canMove(piece) {
        return !busy && isViewingLive() && game && game.status === "active" && game.turn === "human" && humanPiece(piece);
    }

    function keyboardMoveEnabled() {
        return !busy && isViewingLive() && game && game.status === "active" && game.turn === "human";
    }

    function syncMoveControls() {
        const enabled = keyboardMoveEnabled();
        keyboardMoveInput.disabled = !enabled;
        keyboardMoveButton.disabled = !enabled;
    }

    function setBusy(value) {
        busy = value;
        thinkingRail.hidden = !value;
        if (startButton) startButton.disabled = value;
        syncMoveControls();
        syncExportControls();
        if (value) clearExportFeedback();
        if (value) {
            statusElement.textContent = "Waiting for the model…";
            detailElement.textContent = "This can take a little while for reasoning models.";
        }
    }

    function syncExportControls() {
        if (!fenValue || !copyFenButton || !copyPgnButton || !downloadPgnButton) return;
        const visibleFen = displayedFen();
        const fenAvailable = Boolean(visibleFen) && !busy;
        const pgnAvailable = Boolean(game && game.pgn) && !busy;
        fenValue.textContent = visibleFen || "No game yet";
        copyFenButton.disabled = !fenAvailable;
        copyPgnButton.disabled = !pgnAvailable;
        downloadPgnButton.disabled = !pgnAvailable;
    }

    function setExportFeedback(message, isError) {
        if (!exportFeedback) return;
        exportFeedback.textContent = message;
        exportFeedback.dataset.state = isError ? "error" : "success";
    }

    function clearExportFeedback() {
        if (!exportFeedback) return;
        exportFeedback.textContent = "";
        delete exportFeedback.dataset.state;
    }

    async function copyGameData(kind) {
        const value = game && (kind === "FEN" ? displayedFen() : game.pgn);
        if (!value) return;
        try {
            await navigator.clipboard.writeText(value);
            setExportFeedback(kind + " copied.", false);
        } catch (_error) {
            setExportFeedback("Could not copy " + kind + ".", true);
        }
    }

    function exportFileName() {
        const model = String(game.model_id || "llm");
        const effort = String(game.reasoning_effort || "default");
        const rawName = "chessbench-vs-" + model + (effort === "default" ? "" : "-" + effort);
        return rawName.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") + ".pgn";
    }

    function downloadPgn() {
        if (!game || !game.pgn) return;
        const blob = new Blob([game.pgn.trimEnd() + "\n"], {
            type: "application/x-chess-pgn;charset=utf-8",
        });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = exportFileName();
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
        setExportFeedback("PGN downloaded.", false);
    }

    function syncEffortChoices() {
        if (!modelSelect || !effortChoice) return;
        const model = playableModels.find((candidate) => candidate.id === modelSelect.value);
        if (!model) return;

        const available = new Set(model.efforts.map((effort) => effort.id));
        const labels = effortChoice.querySelectorAll("[data-effort]");
        labels.forEach((label) => {
            const input = label.querySelector("input");
            const isAvailable = available.has(label.dataset.effort);
            label.hidden = !isAvailable;
            label.style.display = isAvailable ? "" : "none";
            input.disabled = !isAvailable;
            input.checked = false;
        });

        const defaultLabel = Array.from(labels).find(
            (label) => label.dataset.effort === model.default_effort,
        );
        if (defaultLabel) defaultLabel.querySelector("input").checked = true;
    }

    function playerName(color) {
        if (!game) return color === "white" ? "White" : "Black";
        const isHuman = game.human_color === color;
        if (!isHuman) return game.model_id;
        const username = game.human_profile && game.human_profile.username;
        return username ? username + " (you)" : "You";
    }

    function renderNameplates() {
        const bottomColor = game ? game.human_color : "white";
        const topColor = bottomColor === "white" ? "black" : "white";
        document.getElementById("top-player-name").textContent = playerName(topColor);
        document.getElementById("bottom-player-name").textContent = playerName(bottomColor);
        const topDot = document.getElementById("top-player-dot");
        const bottomDot = document.getElementById("bottom-player-dot");
        topDot.className = "piece-dot piece-dot-" + topColor;
        bottomDot.className = "piece-dot piece-dot-" + bottomColor;
    }

    function finishedResultDescription(scoreDetail) {
        let title = game.winner === "human"
            ? "You won."
            : game.winner === "llm"
                ? game.model_id + " won."
                : "Draw.";
        let reason = "";

        if (game.termination === "checkmate") {
            title = title.slice(0, -1) + " by checkmate.";
        } else if (game.termination === "resignation") {
            title = title.slice(0, -1) + " by resignation.";
            reason = game.winner === "human"
                ? game.model_id + " resigned."
                : "The game ended by resignation.";
        } else if (game.termination === "llm_forfeit_illegal_move") {
            const count = Number(game.llm_illegal_moves) || 2;
            title = "You won by forfeit.";
            reason = game.model_id + " returned " + count + " invalid move response"
                + (count === 1 ? "." : "s.");
        } else if (game.termination === "stalemate") {
            title = "Draw by stalemate.";
        } else if (game.termination === "insufficient_material") {
            title = "Draw by insufficient material.";
        } else if (["threefold_repetition", "fivefold_repetition"].includes(game.termination)) {
            title = "Draw by repetition.";
        } else if (["fifty_moves", "seventyfive_moves"].includes(game.termination)) {
            title = "Draw by the move-count rule.";
        } else if (game.termination === "max_moves") {
            title = "Draw by the game move limit.";
        }

        return [title, [reason, scoreDetail].filter(Boolean).join(" ")];
    }

    function describeGame() {
        if (!game) return [
            "Start a game when you’re ready.",
            "Your current Lichess Classical rating and RD set the weight of this game.",
        ];
        const profile = game.human_profile;
        const pool = profile && profile.rating_pool === "classical" ? "Classical" : "Rapid";
        const snapshot = profile
            ? pool + " " + profile.rating + ", RD " + profile.rating_deviation
                + (profile.provisional ? " (provisional)" : "")
            : "";
        if (game.status === "finished") {
            const rating = game.rating_result;
            const scoreDetail = rating
                ? snapshot + "; model rating " + rating.model_rating + ", RD " + rating.model_rating_deviation + "."
                : snapshot;
            return finishedResultDescription(scoreDetail);
        }
        if (game.turn === "human") return ["Your move.", snapshot + " at game start. Drag a piece to a legal square."];
        return ["The model is choosing a move…", snapshot + " at game start. Keep this tab open while it thinks."];
    }

    function renderLedger() {
        moveList.replaceChildren();
        ledgerMoveButtons = [];
        const moves = game ? game.san_moves : [];
        moveCount.textContent = moves.length + (moves.length === 1 ? " ply" : " plies");
        if (!moves.length) {
            const empty = document.createElement("p");
            empty.className = "ledger-empty";
            empty.textContent = "The score sheet is blank.";
            moveList.appendChild(empty);
            return;
        }
        for (let index = 0; index < moves.length; index += 2) {
            const row = document.createElement("div");
            row.className = "ledger-row";
            const number = document.createElement("span");
            number.className = "ledger-number";
            number.textContent = String(index / 2 + 1).padStart(2, "0");
            const white = document.createElement("button");
            white.type = "button";
            white.className = "ledger-move";
            white.textContent = moves[index] || "";
            white.dataset.ply = String(index + 1);
            white.setAttribute("aria-label", "View position after " + white.textContent);
            white.addEventListener("click", () => navigateToPly(index + 1));
            ledgerMoveButtons.push(white);
            const black = document.createElement("button");
            black.type = "button";
            black.className = "ledger-move";
            black.textContent = moves[index + 1] || "";
            if (moves[index + 1]) {
                black.dataset.ply = String(index + 2);
                black.setAttribute("aria-label", "View position after " + black.textContent);
                black.addEventListener("click", () => navigateToPly(index + 2));
                ledgerMoveButtons.push(black);
            } else {
                black.disabled = true;
                black.setAttribute("aria-hidden", "true");
            }
            row.append(number, white, black);
            moveList.appendChild(row);
        }
        moveList.scrollTop = moveList.scrollHeight;
        syncHistoryControls();
    }

    function syncLedgerSelection() {
        const selectedPly = currentViewedPly();
        ledgerMoveButtons.forEach((button) => {
            const selected = Number(button.dataset.ply) === selectedPly;
            button.className = "ledger-move" + (selected ? " is-current" : "");
            if (selected) button.setAttribute("aria-current", "step");
            else button.removeAttribute("aria-current");
        });
    }

    function syncHistoryControls() {
        const total = totalPlies();
        const current = currentViewedPly();
        if (moveStartButton) moveStartButton.disabled = !game || current === 0;
        if (movePreviousButton) movePreviousButton.disabled = !game || current === 0;
        if (moveNextButton) moveNextButton.disabled = !game || current >= total;
        if (moveLiveButton) moveLiveButton.disabled = !game || isViewingLive();
        if (movePosition) {
            movePosition.textContent = (isViewingLive() ? "Live · " : "") + current + " / " + total;
        }
        syncLedgerSelection();
    }

    function navigateToPly(ply) {
        if (!game) return;
        const total = totalPlies();
        const target = Math.max(0, Math.min(total, Number(ply)));
        viewedPly = target === total ? null : target;
        board.position(displayedFen(), false);
        syncHistoryControls();
        syncMoveControls();
        syncExportControls();
    }

    function render() {
        if (game) {
            if (viewedPly !== null && viewedPly > totalPlies()) viewedPly = null;
            board.position(displayedFen(), false);
            board.orientation(game.human_color);
        } else {
            board.start(false);
        }
        renderNameplates();
        const description = describeGame();
        statusElement.textContent = description[0];
        detailElement.textContent = description[1];
        renderLedger();
        syncMoveControls();
        syncExportControls();
        syncHistoryControls();
    }

    async function postJSON(url, body) {
        const response = await fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRF-Token": app.dataset.csrfToken,
            },
            body: JSON.stringify(body),
        });
        const payload = await response.json().catch(() => ({error: "The server returned an invalid response."}));
        if (!response.ok) throw new Error(payload.error || "The request failed.");
        return payload;
    }

    function promotionFor(source, target, piece) {
        if (!piece || piece[1] !== "P" || !/[18]$/.test(target)) return undefined;
        const answer = window.prompt("Promote to queen, rook, bishop, or knight?", "queen");
        const choices = {queen: "q", q: "q", rook: "r", r: "r", bishop: "b", b: "b", knight: "n", n: "n"};
        return choices[String(answer || "queen").trim().toLowerCase()] || "q";
    }

    function submitMove(uciMove, optimisticFen) {
        pendingFen = optimisticFen;
        board.position(pendingFen, false);
        setBusy(true);
        postJSON(app.dataset.moveUrl, {move: uciMove})
            .then((payload) => {
                pendingFen = null;
                game = payload.game;
                keyboardMoveInput.value = "";
                render();
            })
            .catch((error) => {
                pendingFen = null;
                board.position(displayedFen(), false);
                statusElement.textContent = "Move not completed.";
                detailElement.textContent = error.message;
            })
            .finally(() => setBusy(false));
    }

    board = window.Chessboard("play-board", {
        draggable: true,
        position: game ? game.fen : "start",
        orientation: game ? game.human_color : "white",
        pieceTheme: "https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png",
        onDragStart: function (_source, piece) {
            return canMove(piece);
        },
        onDrop: function (source, target, piece) {
            if (!canMove(piece)) return "snapback";
            const chess = new window.Chess(game.fen);
            const promotion = promotionFor(source, target, piece);
            const localMove = chess.move({from: source, to: target, promotion: promotion || "q"});
            if (!localMove) return "snapback";

            const uciMove = source + target + (promotion || (localMove.promotion || ""));
            submitMove(uciMove, chess.fen());
            return undefined;
        },
        onSnapEnd: function () {
            if (game) board.position(displayedFen(), false);
        },
    });

    window.addEventListener("resize", () => board.resize());

    keyboardMoveForm.addEventListener("submit", (event) => {
        event.preventDefault();
        if (!keyboardMoveEnabled()) return;
        const uciMove = keyboardMoveInput.value.trim().toLowerCase();
        if (!/^[a-h][1-8][a-h][1-8][qrbn]?$/.test(uciMove)) {
            statusElement.textContent = "Move not completed.";
            detailElement.textContent = "Enter a move like e2e4 or e7e8q.";
            keyboardMoveInput.focus();
            return;
        }
        const chess = new window.Chess(game.fen);
        const localMove = chess.move({
            from: uciMove.slice(0, 2),
            to: uciMove.slice(2, 4),
            promotion: uciMove.slice(4) || "q",
        });
        if (!localMove) {
            statusElement.textContent = "Move not completed.";
            detailElement.textContent = "That move is not legal in this position.";
            keyboardMoveInput.focus();
            return;
        }
        submitMove(uciMove, chess.fen());
    });

    if (copyFenButton) copyFenButton.addEventListener("click", () => copyGameData("FEN"));
    if (copyPgnButton) copyPgnButton.addEventListener("click", () => copyGameData("PGN"));
    if (downloadPgnButton) downloadPgnButton.addEventListener("click", downloadPgn);
    if (moveStartButton) moveStartButton.addEventListener("click", () => navigateToPly(0));
    if (movePreviousButton) movePreviousButton.addEventListener("click", () => navigateToPly(currentViewedPly() - 1));
    if (moveNextButton) moveNextButton.addEventListener("click", () => navigateToPly(currentViewedPly() + 1));
    if (moveLiveButton) moveLiveButton.addEventListener("click", () => navigateToPly(totalPlies()));
    document.addEventListener("keydown", (event) => {
        const tagName = event.target && event.target.tagName;
        const isTyping = ["INPUT", "TEXTAREA", "SELECT"].includes(tagName)
            || Boolean(event.target && event.target.isContentEditable);
        if (!game || isTyping || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;

        const destinations = {
            ArrowLeft: currentViewedPly() - 1,
            ArrowRight: currentViewedPly() + 1,
            ArrowUp: 0,
            ArrowDown: totalPlies(),
        };
        if (!(event.key in destinations)) return;
        event.preventDefault();
        navigateToPly(destinations[event.key]);
    });

    if (setupForm) {
        if (modelSelect) modelSelect.addEventListener("change", syncEffortChoices);
        syncEffortChoices();
        setupForm.addEventListener("submit", async (event) => {
            event.preventDefault();
            const formData = new FormData(setupForm);
            setBusy(true);
            try {
                const payload = await postJSON(app.dataset.startUrl, {
                    model_id: formData.get("model_id"),
                    human_color: formData.get("human_color"),
                    reasoning_effort: formData.get("reasoning_effort"),
                    lichess_username: formData.get("lichess_username"),
                });
                game = payload.game;
                viewedPly = null;
                render();
            } catch (error) {
                statusElement.textContent = "Game not started.";
                detailElement.textContent = error.message;
            } finally {
                setBusy(false);
            }
        });
    }

    render();
})();

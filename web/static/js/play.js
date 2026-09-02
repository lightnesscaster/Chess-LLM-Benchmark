(function () {
    "use strict";

    const app = document.getElementById("play-app");
    const boardElement = document.getElementById("play-board");
    if (!app || !boardElement || typeof window.Chessboard !== "function") return;

    const statusElement = document.getElementById("game-status");
    const detailElement = document.getElementById("game-detail");
    const moveList = document.getElementById("move-list");
    const moveCount = document.getElementById("move-count");
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
    let board;

    function humanPiece(piece) {
        if (!game) return false;
        return game.human_color === "white" ? piece.startsWith("w") : piece.startsWith("b");
    }

    function canMove(piece) {
        return !busy && game && game.status === "active" && game.turn === "human" && humanPiece(piece);
    }

    function keyboardMoveEnabled() {
        return !busy && game && game.status === "active" && game.turn === "human";
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
        const available = Boolean(game && game.fen && game.pgn) && !busy;
        fenValue.textContent = game && game.fen ? game.fen : "No game yet";
        copyFenButton.disabled = !available;
        copyPgnButton.disabled = !available;
        downloadPgnButton.disabled = !available;
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
        const value = game && (kind === "FEN" ? game.fen : game.pgn);
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
            if (game.winner === "human") return ["You won.", scoreDetail];
            if (game.winner === "llm") return [game.model_id + " won.", scoreDetail];
            return ["Draw.", scoreDetail];
        }
        if (game.turn === "human") return ["Your move.", snapshot + " at game start. Drag a piece to a legal square."];
        return ["The model is choosing a move…", snapshot + " at game start. Keep this tab open while it thinks."];
    }

    function renderLedger() {
        moveList.replaceChildren();
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
            const white = document.createElement("span");
            white.textContent = moves[index] || "";
            const black = document.createElement("span");
            black.textContent = moves[index + 1] || "";
            row.append(number, white, black);
            moveList.appendChild(row);
        }
        moveList.scrollTop = moveList.scrollHeight;
    }

    function render() {
        if (game) {
            board.position(game.fen, false);
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
                board.position(game.fen, false);
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
            if (game) board.position(pendingFen || game.fen, false);
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

"""Public snapshot behavior; all storage fixtures stay local or use fake RPCs."""
import asyncio
import importlib
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import chess
import pytest


def live_module():
    spec = importlib.util.find_spec("game.live_games")
    assert spec is not None, "Live snapshot backend is missing"
    return importlib.import_module("game.live_games")


def snapshot(game_id="game-1", now=1000, **kwargs):
    return live_module().build_snapshot(
        game_id, "white <script>", "black / arbitrary", chess.Board(),
        started_at=900, now=now, **kwargs,
    )


def test_concurrent_games_are_separate_and_private_fields_are_removed(tmp_path):
    live = live_module()
    store = live.LiveGameStore(path=tmp_path, use_firestore=False)
    first = snapshot()
    first.update(prompt="secret", raw_response="private", human_email="private")
    second = snapshot("game-2", now=1010)
    second["started_at"] = "1970-01-01T00:15:10+00:00"
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(store.write, [first, second]))
    games = store.list_games(now=1020)
    assert [g["game_id"] for g in games] == ["game-2", "game-1"]
    assert games[1]["white"] == "white <script>"
    assert not {"prompt", "raw_response", "human_email"} & games[1].keys()
    assert "secret" not in next(tmp_path.glob("*.json")).read_text()


def test_stale_records_interrupt_and_terminal_records_expire(tmp_path):
    live = live_module()
    store = live.LiveGameStore(path=tmp_path, use_firestore=False)
    store.write(snapshot())
    store.write(snapshot("finished", status="finished", winner="white", termination="checkmate"))
    games = {g["game_id"]: g for g in store.list_games(now=1091)}
    assert games["game-1"]["status"] == "interrupted"
    assert games["finished"]["result"] == "1-0"
    assert [g["game_id"] for g in store.list_games(now=1600)] == ["game-1"]
    assert store.list_games(now=2800) == []


def test_snapshot_replay_preserves_castling_and_promotion_san():
    live = live_module()
    board = chess.Board()
    for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1e2", "g8f6", "e1g1"]:
        board.push_uci(uci)
    data = live.build_snapshot("castle", "w", "b", board, started_at=900, now=1000)
    assert data["san_moves"] == ["e4", "e5", "Nf3", "Nc6", "Be2", "Nf6", "O-O"]
    assert data["turn"] == "black"
    assert data["fen"] == board.fen()
    board = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
    board.push_uci("a7a8q")
    data = live.build_snapshot("promotion", "w", "b", board, started_at=900, now=1000)
    assert data["moves"] == ["a7a8q"]
    assert data["san_moves"] == ["a8=Q+"]


def test_heartbeat_refreshes_expiry_without_changing_moves(tmp_path, monkeypatch):
    live = live_module()
    monkeypatch.setattr(live, "HEARTBEAT_INTERVAL", 0.01)
    store = live.LiveGameStore(path=tmp_path, use_firestore=False)

    async def run():
        publisher = live.LiveGamePublisher(store)
        publisher.start()
        publisher.submit(snapshot(now=time.time() - 100))
        await asyncio.sleep(0.06)
        await publisher.close()

    asyncio.run(run())
    data = store.list_games()[0]
    assert data["status"] == "active"
    assert data["heartbeat_at"] > time.time() - 1
    assert data["expires_at"] > time.time() + 1790
    assert data["moves"] == []


def test_publisher_coalesces_updates_and_bounds_shutdown():
    live = live_module()
    entered, release = threading.Event(), threading.Event()
    writes = []

    class SlowStore:
        def write(self, data):
            entered.set()
            release.wait(2)
            writes.append(data)

    async def run():
        publisher = live.LiveGamePublisher(SlowStore())
        publisher.start()
        publisher.submit(snapshot())
        while not entered.is_set():
            await asyncio.sleep(0.001)
        for n in range(10):
            publisher.submit(snapshot(now=1001+n))
        started = time.monotonic()
        await publisher.close(timeout=0.02)
        assert time.monotonic() - started < 0.2
        release.set()
        for _ in range(100):
            if len(writes) == 2:
                break
            await asyncio.sleep(0.005)
        assert len(writes) == 2
        assert writes[-1]["heartbeat_at"] == 1010

    asyncio.run(run())


def test_firestore_uses_distinct_documents_bounded_rpcs_and_filters_expiry(monkeypatch):
    live = live_module()
    writes = {}
    calls = []

    class Collection:
        def document(self, game_id):
            class Document:
                def set(self, data, **kwargs):
                    calls.append(kwargs)
                    writes[game_id] = data
            return Document()

        def where(self, **kwargs):
            calls.append(kwargs)
            return self

        def stream(self, **kwargs):
            calls.append(kwargs)
            return [type("Doc", (), {"to_dict": lambda self, value=data: value})() for data in writes.values()]

    class DB:
        def collection(self, name):
            assert name == "live_games"
            return Collection()

    monkeypatch.setattr("firebase_client.get_firestore_client", lambda: DB())
    store = live.LiveGameStore(use_firestore=True)
    store.write(snapshot("one"))
    store.write(snapshot("two"))
    assert {g["game_id"] for g in store.list_games(now=1000)} == {"one", "two"}
    assert len(writes) == 2
    assert all(c["timeout"] <= 3 and c["retry"] is None for c in calls if "timeout" in c)
    assert any("filter" in c for c in calls)


@pytest.mark.parametrize("failure", ["store", "snapshot"])
def test_live_publication_failure_does_not_forfeit_a_legal_game(monkeypatch, failure):
    live = live_module()
    from game.game_runner import GameRunner
    from tests.test_game_runner import SequencePlayer

    class FailingStore:
        def write(self, data):
            raise OSError("storage unavailable")

    monkeypatch.setattr(live, "get_live_game_store", lambda: FailingStore())
    if failure == "snapshot":
        monkeypatch.setattr(live, "build_snapshot", lambda *a, **k: (_ for _ in ()).throw(ValueError("snapshot failure")))
    result, pgn = asyncio.run(GameRunner(SequencePlayer(["e2e4"]), SequencePlayer(["e7e5"]), max_moves=2, publish_live=True).play_game())
    assert result.termination == "max_moves"
    assert result.winner == "draw"
    assert result.moves == 2
    assert "1. e4 e5" in pgn


def test_disabled_broadcast_does_no_snapshot_storage_io(monkeypatch):
    live = live_module()
    from game.game_runner import GameRunner
    from tests.test_game_runner import SequencePlayer
    monkeypatch.setattr(live, "get_live_game_store", lambda: pytest.fail("disabled broadcast touched storage"))
    result, _ = asyncio.run(GameRunner(SequencePlayer(["e2e4"]), SequencePlayer(["e7e5"]), max_moves=2).play_game())
    assert result.termination == "max_moves"


def test_runner_initial_moves_and_terminal_snapshot_use_same_game_id(monkeypatch, tmp_path):
    live = live_module()
    from game.game_runner import GameRunner
    from tests.test_game_runner import SequencePlayer
    store = live.LiveGameStore(path=tmp_path, use_firestore=False)
    writes = []
    original_write = store.write

    def record(data):
        writes.append(data)
        original_write(data)

    monkeypatch.setattr(store, "write", record)
    monkeypatch.setattr(live, "get_live_game_store", lambda: store)
    result, _ = asyncio.run(GameRunner(SequencePlayer(["e2e4"]), SequencePlayer(["resign"]), max_moves=2, publish_live=True).play_game())
    data = store.list_games()[0]
    assert data["game_id"] == result.game_id
    assert {record["game_id"] for record in writes} == {result.game_id}
    assert data["moves"] == ["e2e4"]
    assert data["san_moves"] == ["e4"]
    assert data["status"] == "finished"
    assert data["winner"] == "white"
    assert data["result"] == "1-0"


def test_runner_cancellation_flushes_interrupted_state(monkeypatch, tmp_path):
    live = live_module()
    from game.game_runner import GameRunner
    from tests.test_game_runner import SequencePlayer
    store = live.LiveGameStore(path=tmp_path, use_firestore=False)
    monkeypatch.setattr(live, "get_live_game_store", lambda: store)

    async def run():
        entered = asyncio.Event()
        runner = GameRunner(SequencePlayer([]), SequencePlayer([]), publish_live=True)

        async def wait(*a, **k):
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(runner, "_ask_player_for_move", wait)
        task = asyncio.create_task(runner.play_game())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    data = store.list_games()[0]
    assert data["status"] == "interrupted"
    assert data["result"] == "*"
    assert data["termination"] == "cancelled"


def test_scheduler_opts_in_to_live_publication(monkeypatch):
    from types import SimpleNamespace
    import game.match_scheduler as scheduler_module
    from engines.random_engine import RandomEngine
    from game.match_scheduler import GameTask, MatchScheduler
    options = []

    class Runner:
        def __init__(self, **kwargs):
            options.append(kwargs)

        async def play_game(self):
            return SimpleNamespace(termination="api_error"), ""

    monkeypatch.setattr(scheduler_module, "GameRunner", Runner)
    monkeypatch.setattr(scheduler_module, "rating_snapshot", lambda *a: {})
    scheduler = MatchScheduler.__new__(MatchScheduler)
    scheduler._semaphore = asyncio.Semaphore(1)
    scheduler.verbose = False
    scheduler.max_moves = 2
    scheduler.rating_store = None
    asyncio.run(scheduler.run_single_game(GameTask(RandomEngine("w", 400), RandomEngine("b", 400), 1, 1)))
    assert options[0].get("publish_live") is True


def test_store_cache_refreshes_after_two_seconds_and_rechecks_staleness(tmp_path, monkeypatch):
    live = live_module()
    store = live.LiveGameStore(path=tmp_path, use_firestore=False)
    store.write(snapshot())
    clock = [10.0]
    monkeypatch.setattr(live.time, "monotonic", lambda: clock[0])
    assert store.list_games(now=1000)[0]["status"] == "active"
    assert store.list_games(now=1091)[0]["status"] == "interrupted"
    calls = []

    def unavailable(now):
        calls.append(now)
        raise OSError("unavailable")

    monkeypatch.setattr(store, "_read", unavailable)
    clock[0] = 11.9
    assert len(store.list_games(now=1091)) == 1
    clock[0] = 12.0
    with pytest.raises(OSError):
        store.list_games(now=1091)
    assert calls == [1091]


def test_active_games_sort_before_newer_terminal_games(tmp_path):
    live = live_module()
    store = live.LiveGameStore(path=tmp_path, use_firestore=False)
    store.write(snapshot("old-active"))
    terminal = snapshot("new-terminal", now=1010, status="finished", winner="draw")
    terminal["started_at"] = "1970-01-01T00:16:40+00:00"
    store.write(terminal)
    assert [g["game_id"] for g in store.list_games(now=1020)] == ["old-active", "new-terminal"]


def test_hung_storage_does_not_block_moves_or_event_loop_shutdown(monkeypatch):
    live = live_module()
    from game.game_runner import GameRunner
    from tests.test_game_runner import SequencePlayer
    release = threading.Event()
    moves_at = []

    class HungStore:
        def write(self, data):
            release.wait(5)

    class Player(SequencePlayer):
        async def select_move(self, *a, **k):
            moves_at.append(time.monotonic())
            return await super().select_move(*a, **k)

    monkeypatch.setattr(live, "get_live_game_store", lambda: HungStore())
    started = time.monotonic()
    try:
        result, _ = asyncio.run(GameRunner(Player(["e2e4"]), Player(["e7e5"]), max_moves=2, publish_live=True).play_game())
        assert time.monotonic() - started < 1.5
        assert all(at - started < 0.2 for at in moves_at)
        assert result.termination == "max_moves"
    finally:
        release.set()


def test_snapshot_finish_failure_preserves_cancellation(monkeypatch):
    live = live_module()
    from game.game_runner import GameRunner
    from tests.test_game_runner import SequencePlayer

    class Publisher:
        def __init__(self, store):
            pass

        def start(self):
            pass

        def submit(self, data):
            pass

        def finish(self, *args):
            raise OSError("snapshot unavailable")

        async def close(self):
            pass

    monkeypatch.setattr(live, "get_live_game_store", lambda: object())
    monkeypatch.setattr(live, "LiveGamePublisher", Publisher)

    async def run():
        runner = GameRunner(SequencePlayer([]), SequencePlayer([]), publish_live=True)

        async def cancel(*a, **k):
            raise asyncio.CancelledError()

        monkeypatch.setattr(runner, "_ask_player_for_move", cancel)
        with pytest.raises(asyncio.CancelledError):
            await runner.play_game()

    asyncio.run(run())


def test_runner_publishes_start_and_accepted_move_before_next_inference_finishes(monkeypatch, tmp_path):
    live = live_module()
    from game.game_runner import GameRunner
    from tests.test_game_runner import SequencePlayer
    store = live.LiveGameStore(path=tmp_path, use_firestore=False)
    monkeypatch.setattr(live, "get_live_game_store", lambda: store)

    async def run():
        white_started, black_started = asyncio.Event(), asyncio.Event()
        release_white, release_black = asyncio.Event(), asyncio.Event()

        class Player(SequencePlayer):
            def __init__(self, responses, started, release):
                super().__init__(responses)
                self.started, self.release = started, release

            async def select_move(self, *a, **k):
                self.started.set()
                await self.release.wait()
                return await super().select_move(*a, **k)

        async def await_moves(expected):
            for _ in range(100):
                data = store.list_games()
                if data and data[0]["moves"] == expected:
                    return data[0]
                await asyncio.sleep(0.005)
            pytest.fail("Live snapshot was not published during inference")

        runner = GameRunner(Player(["e2e4"], white_started, release_white),
                            Player(["resign"], black_started, release_black), publish_live=True)
        task = asyncio.create_task(runner.play_game())
        try:
            await white_started.wait()
            initial = await await_moves([])
            assert initial["status"] == "active"
            assert initial["turn"] == "white"
            assert initial["result"] == "*"
            release_white.set()
            await black_started.wait()
            updated = await await_moves(["e2e4"])
            assert updated["game_id"] == initial["game_id"]
            assert updated["status"] == "active"
            assert updated["turn"] == "black"
            assert updated["san_moves"] == ["e4"]
            release_black.set()
            result, _ = await task
            assert result.winner == "white"
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    asyncio.run(run())


@pytest.mark.parametrize("cancel_inference", [False, True])
def test_cancellation_during_runner_final_flush_propagates_and_stops_writer(monkeypatch, cancel_inference):
    live = live_module()
    from game.game_runner import GameRunner
    from tests.test_game_runner import SequencePlayer
    publishers = []
    original_close = live.LiveGamePublisher.close

    class Store:
        def write(self, data):
            pass

    async def cancel_during_close(self, timeout=1.0):
        publishers.append(self)
        asyncio.get_running_loop().call_soon(asyncio.current_task().cancel, "flush")
        await original_close(self, timeout=timeout)

    monkeypatch.setattr(live, "get_live_game_store", lambda: Store())
    monkeypatch.setattr(live.LiveGamePublisher, "close", cancel_during_close)

    async def run():
        runner = GameRunner(SequencePlayer(["e2e4"]), SequencePlayer(["e7e5"]),
                            max_moves=2, publish_live=True)
        entered = asyncio.Event()
        inference_errors = []

        async def wait_for_cancel(*a, **k):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as error:
                inference_errors.append(error)
                raise

        if cancel_inference:
            monkeypatch.setattr(runner, "_ask_player_for_move", wait_for_cancel)
        task = asyncio.create_task(runner.play_game())
        if cancel_inference:
            await entered.wait()
            task.cancel("inference")
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await task
        if cancel_inference:
            assert cancelled.value is not inference_errors[0]
        assert task.cancelled()
        publisher = publishers[0]
        assert publisher._closing
        for _ in range(100):
            if not publisher._thread.is_alive():
                break
            await asyncio.sleep(0.005)
        assert not publisher._thread.is_alive()

    asyncio.run(run())

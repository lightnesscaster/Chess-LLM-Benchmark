from game.models import GameResult
from game.stats_collector import StatsCollector


def test_resignation_is_tracked_separately_from_illegal_forfeits():
    result = GameResult(
        game_id="resignation-1",
        white_id="model-a",
        black_id="model-b",
        winner="black",
        termination="resignation",
        moves=20,
        illegal_moves_white=0,
        illegal_moves_black=0,
        total_moves_white=10,
        total_moves_black=10,
        pgn_path="resignation-1.pgn",
        created_at="2026-09-02T12:00:00+00:00",
    )
    collector = StatsCollector()
    collector.add_result(result)

    stats = collector.get_player_stats()

    assert stats["model-a"]["resignations"] == 1
    assert stats["model-a"]["resignation_rate"] == 1.0
    assert stats["model-a"]["forfeits"] == 0
    assert stats["model-b"]["resignations"] == 0

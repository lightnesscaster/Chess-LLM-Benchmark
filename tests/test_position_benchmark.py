import json
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path

import chess

from position_benchmark.predictions import (
    CURRENT_BENCHMARK_VERSION,
    benchmark_result_readiness,
    collect_equal_position_coverage,
    collect_equal_position_metrics,
    combine_equal_and_game_like_predictions,
    predict_rating_from_model_data,
    predict_rating_from_model_data_with_supplement,
    predict_rating_from_results,
    stability_probe_prediction_cap,
)
from position_benchmark.run_benchmark import build_ad_hoc_player_config, replay_position_board, test_llm_on_position
from position_benchmark.run_benchmark import calculate_actual_benchmark_cost, estimate_selected_benchmark_cost
from rating.cost_calculator import CostCalculator
from scripts.analyze_puzzle_predictions import UNAVAILABLE_PLAYER_IDS
from scripts.audit_position_benchmark_readiness import AD_HOC_PLAYER_HINTS, inspect_player_results
from scripts.promote_position_benchmark_overlays import merge_overlays, refresh_summary
from scripts.reevaluate_position_result_overlays import filter_results
from scripts.reevaluate_position_result_overlays import recalculate_summary
from scripts.run_stability_probe import pre_moves_from_position, summarize_player


ROOT = Path(__file__).resolve().parents[1]


class CostEstimateTests(unittest.TestCase):
    def test_estimates_position_benchmark_cost_from_pricing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pricing_path = Path(tmpdir) / "pricing.json"
            config_path = Path(tmpdir) / "benchmark.yaml"
            pricing_path.write_text(json.dumps({"provider/model": {"prompt": 1e-6, "completion": 1e-5}}))
            config_path.write_text("llms: []\n")
            calculator = CostCalculator(pricing_path=pricing_path, config_path=config_path)

            cost = calculator.estimate_position_benchmark_cost(
                "test-model",
                model_name="provider/model",
                num_positions=10,
                reasoning=True,
                prompt_tokens_per_position=1000,
                completion_tokens_per_position=100,
            )

            self.assertAlmostEqual(cost, 0.02)

    def test_scales_position_benchmark_budget_override_for_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pricing_path = Path(tmpdir) / "pricing.json"
            config_path = Path(tmpdir) / "benchmark.yaml"
            pricing_path.write_text("{}")
            config_path.write_text(
                "\n".join(
                    [
                        "llms:",
                        '  - player_id: "test-model"',
                        '    model_name: "provider/model"',
                        "    budget_cost_per_game: 2.0",
                    ]
                )
            )
            calculator = CostCalculator(pricing_path=pricing_path, config_path=config_path)

            cost = calculator.estimate_position_benchmark_cost("test-model", num_positions=10)

            self.assertAlmostEqual(cost, 0.4)

    def test_estimates_selected_runner_cost_after_retry_missing_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pricing_path = Path(tmpdir) / "pricing.json"
            config_path = Path(tmpdir) / "benchmark.yaml"
            pricing_path.write_text(json.dumps({"provider/model": {"prompt": 1e-6, "completion": 1e-5}}))
            config_path.write_text("llms: []\n")
            calculator = CostCalculator(pricing_path=pricing_path, config_path=config_path)
            players = {
                "model-a": {
                    "player_id": "model-a",
                    "model_name": "provider/model",
                    "reasoning_effort": "high",
                },
                "random-bot": {"player_id": "random-bot", "type": "random"},
            }
            all_results = {"model-a": {"results": [{"position_idx": 25}]}}

            cost, unknown = estimate_selected_benchmark_cost(
                players,
                selected_count=3,
                type_filter_idx_map={0: 25, 1: 26, 2: 27},
                all_results=all_results,
                retry_missing=True,
                cost_calculator=calculator,
            )

            self.assertEqual(unknown, [])
            self.assertAlmostEqual(cost, 0.007)

    def test_calculates_actual_position_benchmark_cost_from_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pricing_path = Path(tmpdir) / "pricing.json"
            config_path = Path(tmpdir) / "benchmark.yaml"
            pricing_path.write_text(json.dumps({"provider/model": {"prompt": 1e-6, "completion": 1e-5}}))
            config_path.write_text("llms: []\n")
            calculator = CostCalculator(pricing_path=pricing_path, config_path=config_path)

            cost = calculate_actual_benchmark_cost(
                "model-a",
                {"player_id": "model-a", "model_name": "provider/model"},
                {"prompt": 1234, "completion": 56},
                num_positions=1,
                cost_calculator=calculator,
            )

            self.assertAlmostEqual(cost, 0.001794)

    def test_actual_position_benchmark_cost_uses_budget_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pricing_path = Path(tmpdir) / "pricing.json"
            config_path = Path(tmpdir) / "benchmark.yaml"
            pricing_path.write_text(json.dumps({"provider/model": {"prompt": 1e-6, "completion": 1e-5}}))
            config_path.write_text(
                "\n".join(
                    [
                        "llms:",
                        '  - player_id: "model-a"',
                        '    model_name: "provider/model"',
                        "    budget_cost_per_game: 1.0",
                    ]
                )
            )
            calculator = CostCalculator(pricing_path=pricing_path, config_path=config_path)

            cost = calculate_actual_benchmark_cost(
                "model-a",
                {"player_id": "model-a", "model_name": "provider/model"},
                {"prompt": 999999, "completion": 999999},
                num_positions=5,
                cost_calculator=calculator,
            )

            self.assertAlmostEqual(cost, 0.1)


class PositionReplayTests(unittest.TestCase):
    def test_replays_san_history_from_blunder_positions(self) -> None:
        with (ROOT / "position_benchmark" / "positions.json").open() as f:
            positions = json.load(f)["positions"]

        position = positions[0]
        board = replay_position_board(position)

        self.assertGreater(len(board.move_stack), 0)
        self.assertEqual(
            " ".join(board.fen().split()[:4]),
            " ".join(position["fen"].split()[:4]),
        )

    def test_replays_uci_history_from_equal_positions(self) -> None:
        with (ROOT / "position_benchmark" / "positions.json").open() as f:
            positions = json.load(f)["positions"]

        position = next(p for p in positions if p.get("type") == "equal")
        board = replay_position_board(position)

        self.assertGreater(len(board.move_stack), 0)
        self.assertEqual(board.fen(), position["fen"])

    def test_builds_ad_hoc_player_config(self) -> None:
        args = SimpleNamespace(
            model_name="google/gemini-3.1-pro-preview",
            player_id="gemini-3.1-pro-preview (high)",
            temperature=0.0,
            reasoning_effort="high",
            no_reasoning=False,
            timeout=1200,
        )

        config = build_ad_hoc_player_config(args)

        self.assertIsNotNone(config)
        self.assertEqual(config["player_id"], args.player_id)
        self.assertEqual(config["model_name"], args.model_name)
        self.assertEqual(config["reasoning_effort"], "high")
        self.assertEqual(config["timeout"], 1200)


class FakeIllegalPlayer:
    async def select_move(self, *_args, **_kwargs) -> str:
        return "a1a1"


class PositionBenchmarkResultTests(unittest.IsolatedAsyncioTestCase):
    async def test_illegal_llm_move_uses_position_fen(self) -> None:
        with (ROOT / "position_benchmark" / "positions.json").open() as f:
            positions = json.load(f)["positions"]

        position = next(p for p in positions if p.get("type") == "equal")
        result = await test_llm_on_position(FakeIllegalPlayer(), position, engine=None, depth=1)

        self.assertEqual(result.fen, position["fen"])
        self.assertFalse(result.is_legal)
        self.assertEqual(result.model_move, "a1a1")
        self.assertEqual(result.cpl, position["eval_before"] + 5000)


class PredictionMetricTests(unittest.TestCase):
    def test_recomputes_best_move_from_current_position(self) -> None:
        positions = [
            {
                "type": "equal",
                "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "best_move": "a1b1",
            }
        ]
        results = [
            {
                "position_idx": 0,
                "fen": positions[0]["fen"],
                "model_move": "a1b1",
                "best_move": "a1a2",
                "is_best": False,
                "is_legal": True,
                "cpl": 0,
            }
        ]

        metrics = collect_equal_position_metrics(results, positions)

        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.best_pct, 100.0)
        self.assertIsNotNone(predict_rating_from_results(results, positions))

    def test_model_prediction_requires_current_history_replay_metadata(self) -> None:
        positions = [
            {
                "type": "equal",
                "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "best_move": "a1b1",
            }
        ]
        model_data = {
            "summary": {},
            "results": [
                {
                    "position_idx": 0,
                    "fen": positions[0]["fen"],
                    "model_move": "a1b1",
                    "is_legal": True,
                    "cpl": 0,
                }
            ],
        }

        self.assertIsNone(
            predict_rating_from_model_data(
                model_data,
                positions,
                min_equal_positions=1,
                min_stockfish_depth=1,
            )
        )

        model_data["summary"] = {
            "position_benchmark_version": CURRENT_BENCHMARK_VERSION,
            "prompt_history_replay": True,
            "stockfish_depth": 30,
        }
        summary_only = benchmark_result_readiness(
            model_data,
            positions,
            min_equal_positions=1,
            min_stockfish_depth=1,
        )

        self.assertFalse(summary_only.is_ready)
        self.assertEqual(summary_only.reason, "stale equal-position rows")

        model_data["results"][0]["position_benchmark_version"] = CURRENT_BENCHMARK_VERSION
        model_data["results"][0]["prompt_history_replay"] = True
        model_data["results"][0]["stockfish_depth"] = 30
        readiness = benchmark_result_readiness(
            model_data,
            positions,
            min_equal_positions=1,
            min_stockfish_depth=1,
        )

        self.assertTrue(readiness.is_ready)
        self.assertIsNotNone(
            predict_rating_from_model_data(
                model_data,
                positions,
                min_equal_positions=1,
                min_stockfish_depth=1,
            )
        )

    def test_readiness_rejects_duplicate_equal_rows(self) -> None:
        positions = [
            {"type": "equal", "fen": "8/8/8/8/8/8/8/K6k w - - 0 1", "best_move": "a1b1"},
            {"type": "equal", "fen": "8/8/8/8/8/8/7k/K7 w - - 0 1", "best_move": "a1b1"},
        ]
        current_row = {
            "position_idx": 0,
            "fen": positions[0]["fen"],
            "model_move": "a1b1",
            "is_legal": True,
            "cpl": 0,
            "position_benchmark_version": CURRENT_BENCHMARK_VERSION,
            "prompt_history_replay": True,
            "stockfish_depth": 30,
        }
        model_data = {"summary": {}, "results": [dict(current_row), dict(current_row)]}

        coverage = collect_equal_position_coverage(
            model_data["results"],
            positions,
            min_stockfish_depth=30,
        )
        readiness = benchmark_result_readiness(
            model_data,
            positions,
            min_equal_positions=2,
            min_stockfish_depth=30,
        )

        self.assertEqual(coverage.unique_valid, 1)
        self.assertEqual(coverage.duplicate_rows, 1)
        self.assertFalse(readiness.is_ready)
        self.assertEqual(readiness.reason, "duplicate equal-position rows")

    def test_readiness_rejects_missing_equal_rows(self) -> None:
        positions = [
            {"type": "equal", "fen": "8/8/8/8/8/8/8/K6k w - - 0 1", "best_move": "a1b1"},
            {"type": "equal", "fen": "8/8/8/8/8/8/7k/K7 w - - 0 1", "best_move": "a1b1"},
        ]
        model_data = {
            "summary": {},
            "results": [
                {
                    "position_idx": 0,
                    "fen": positions[0]["fen"],
                    "model_move": "a1b1",
                    "is_legal": True,
                    "cpl": 0,
                    "position_benchmark_version": CURRENT_BENCHMARK_VERSION,
                    "prompt_history_replay": True,
                    "stockfish_depth": 30,
                }
            ],
        }

        readiness = benchmark_result_readiness(
            model_data,
            positions,
            min_equal_positions=1,
            min_stockfish_depth=30,
        )

        self.assertFalse(readiness.is_ready)
        self.assertEqual(readiness.reason, "missing equal-position rows")

    def test_game_like_supplement_is_only_a_material_downside_cap(self) -> None:
        self.assertEqual(combine_equal_and_game_like_predictions(1000.0, 1300.0), 1000.0)
        self.assertEqual(combine_equal_and_game_like_predictions(1000.0, 900.0), 1000.0)
        self.assertEqual(combine_equal_and_game_like_predictions(1000.0, 800.0), 800.0)

    def test_stale_game_like_supplement_does_not_affect_prediction(self) -> None:
        positions = [
            {
                "type": "equal",
                "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "best_move": "a1b1",
            }
        ]
        primary = {
            "summary": {},
            "results": [
                {
                    "position_idx": 0,
                    "fen": positions[0]["fen"],
                    "model_move": "a1b1",
                    "is_legal": True,
                    "cpl": 0,
                    "position_benchmark_version": CURRENT_BENCHMARK_VERSION,
                    "prompt_history_replay": True,
                    "stockfish_depth": 30,
                }
            ],
        }
        stale_supplement = {
            "summary": {},
            "results": [
                {
                    "position_idx": 0,
                    "fen": positions[0]["fen"],
                    "model_move": "a1a2",
                    "is_legal": True,
                    "cpl": 5000,
                }
            ],
        }

        primary_only = predict_rating_from_model_data(
            primary,
            positions,
            min_equal_positions=1,
            min_stockfish_depth=1,
        )
        supplemented = predict_rating_from_model_data_with_supplement(
            primary,
            positions,
            game_like_model_data=stale_supplement,
            game_like_positions=positions,
            min_equal_positions=1,
            min_game_like_positions=1,
            min_stockfish_depth=1,
        )

        self.assertEqual(supplemented, primary_only)

    def test_fresh_game_like_supplement_caps_optimistic_equal_prediction(self) -> None:
        positions = [
            {
                "type": "equal",
                "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "best_move": "a1b1",
            }
        ]
        current_row = {
            "position_idx": 0,
            "fen": positions[0]["fen"],
            "is_legal": True,
            "position_benchmark_version": CURRENT_BENCHMARK_VERSION,
            "prompt_history_replay": True,
            "stockfish_depth": 30,
        }
        primary = {
            "summary": {},
            "results": [dict(current_row, model_move="a1b1", cpl=0)],
        }
        supplement = {
            "summary": {},
            "results": [dict(current_row, model_move="a1a2", cpl=10000)],
        }

        primary_only = predict_rating_from_model_data(
            primary,
            positions,
            min_equal_positions=1,
            min_stockfish_depth=1,
        )
        supplemented = predict_rating_from_model_data_with_supplement(
            primary,
            positions,
            game_like_model_data=supplement,
            game_like_positions=positions,
            min_equal_positions=1,
            min_game_like_positions=1,
            min_stockfish_depth=1,
            game_like_cpl_cap=5000,
        )

        self.assertIsNotNone(primary_only)
        self.assertIsNotNone(supplemented)
        self.assertLess(supplemented, primary_only)

    def test_fresh_blunder_rows_cap_optimistic_equal_prediction(self) -> None:
        positions = [
            {
                "type": "equal",
                "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "best_move": "a1b1",
            },
            {
                "type": "blunder",
                "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "best_move": "a1b1",
            },
        ]
        current = {
            "fen": positions[0]["fen"],
            "is_legal": True,
            "position_benchmark_version": CURRENT_BENCHMARK_VERSION,
            "prompt_history_replay": True,
            "stockfish_depth": 30,
        }
        model_data = {
            "summary": {},
            "results": [
                dict(current, position_idx=0, model_move="a1b1", cpl=0),
                dict(current, position_idx=1, model_move="a1a2", cpl=10000),
            ],
        }

        primary_only = predict_rating_from_model_data(
            {"summary": {}, "results": [model_data["results"][0]]},
            positions,
            min_equal_positions=1,
            min_stockfish_depth=1,
        )
        supplemented = predict_rating_from_model_data_with_supplement(
            model_data,
            positions,
            min_equal_positions=1,
            min_blunder_positions=1,
            min_stockfish_depth=1,
            game_like_cpl_cap=5000,
        )

        self.assertIsNotNone(primary_only)
        self.assertIsNotNone(supplemented)
        self.assertLess(supplemented, primary_only)

    def test_stability_probe_cap_requires_enough_scored_moves(self) -> None:
        positions = [
            {
                "type": "equal",
                "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "best_move": "a1b1",
            }
        ]
        primary = {
            "summary": {},
            "results": [
                {
                    "position_idx": 0,
                    "fen": positions[0]["fen"],
                    "model_move": "a1b1",
                    "is_legal": True,
                    "cpl": 0,
                    "position_benchmark_version": CURRENT_BENCHMARK_VERSION,
                    "prompt_history_replay": True,
                    "stockfish_depth": 30,
                }
            ],
        }
        incomplete_stability = {
            "summary": {
                "attempted_positions": 4,
                "model_scored_moves": 12,
                "model_forfeit_pct": 50.0,
                "model_1000cp_catastrophe_pct": 25.0,
                "score_depth": 10,
            }
        }

        primary_only = predict_rating_from_model_data(
            primary,
            positions,
            min_equal_positions=1,
            min_stockfish_depth=1,
        )
        supplemented = predict_rating_from_model_data_with_supplement(
            primary,
            positions,
            stability_probe_model_data=incomplete_stability,
            min_equal_positions=1,
            min_stockfish_depth=1,
        )

        self.assertEqual(supplemented, primary_only)

    def test_stability_probe_caps_weak_live_continuation_risk(self) -> None:
        positions = [
            {
                "type": "equal",
                "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "best_move": "a1b1",
            }
        ]
        primary = {
            "summary": {},
            "results": [
                {
                    "position_idx": 0,
                    "fen": positions[0]["fen"],
                    "model_move": "a1b1",
                    "is_legal": True,
                    "cpl": 0,
                    "position_benchmark_version": CURRENT_BENCHMARK_VERSION,
                    "prompt_history_replay": True,
                    "stockfish_depth": 30,
                }
            ],
        }
        stability = {
            "summary": {
                "attempted_positions": 8,
                "model_scored_moves": 24,
                "model_forfeit_pct": 25.0,
                "model_1000cp_catastrophe_pct": 12.5,
                "score_depth": 10,
            }
        }

        cap = stability_probe_prediction_cap(stability)
        supplemented = predict_rating_from_model_data_with_supplement(
            primary,
            positions,
            stability_probe_model_data=stability,
            min_equal_positions=1,
            min_stockfish_depth=1,
        )

        self.assertEqual(cap, 400.0)
        self.assertEqual(supplemented, cap)

    def test_stability_probe_can_cap_forfeit_heavy_rows_without_scored_moves(self) -> None:
        positions = [
            {
                "type": "equal",
                "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "best_move": "a1b1",
            }
        ]
        primary = {
            "summary": {},
            "results": [
                {
                    "position_idx": 0,
                    "fen": positions[0]["fen"],
                    "model_move": "a1b1",
                    "is_legal": True,
                    "cpl": 0,
                    "position_benchmark_version": CURRENT_BENCHMARK_VERSION,
                    "prompt_history_replay": True,
                    "stockfish_depth": 30,
                }
            ],
        }
        stability = {
            "summary": {
                "attempted_positions": 8,
                "model_attempts": 16,
                "model_scored_moves": 0,
                "model_forfeits": 8,
                "model_forfeit_pct": 100.0,
                "model_1000cp_catastrophe_pct": 0.0,
                "score_depth": 10,
            }
        }

        supplemented = predict_rating_from_model_data_with_supplement(
            primary,
            positions,
            stability_probe_model_data=stability,
            min_equal_positions=1,
            min_stockfish_depth=1,
        )

        self.assertEqual(supplemented, 50.0)


class AvailabilityHintTests(unittest.TestCase):
    def test_deprecated_or_no_endpoint_models_are_not_actionable_reruns(self) -> None:
        for player_id in ["gemini-2.0-flash-001", "grok-4-fast", "grok-4.1-fast"]:
            self.assertIn(player_id, UNAVAILABLE_PLAYER_IDS)
            self.assertTrue(AD_HOC_PLAYER_HINTS[player_id]["unavailable"])


class StabilityProbeTests(unittest.TestCase):
    def test_pre_moves_reconstruct_nonopening_position(self) -> None:
        with (ROOT / "position_benchmark" / "nonopening_screening_positions.json").open() as f:
            position = json.load(f)["positions"][0]

        board = chess.Board()
        for move_uci in pre_moves_from_position(position):
            board.push(chess.Move.from_uci(move_uci))

        self.assertEqual(
            " ".join(board.fen().split()[:4]),
            " ".join(position["fen"].split()[:4]),
        )

    def test_stability_summary_counts_illegal_and_forfeit_rates(self) -> None:
        rows = [
            {
                "model_legal_moves": 4,
                "model_illegal_attempts": 0,
                "model_forfeited": False,
                "termination": "max_moves",
                "probe_plies_played": 8,
                "model_move_scores": [
                    {"cpl": 25},
                    {"cpl": 350},
                ],
            },
            {
                "model_legal_moves": 1,
                "model_illegal_attempts": 2,
                "model_forfeited": True,
                "termination": "forfeit_illegal_move",
                "probe_plies_played": 3,
                "model_move_scores": [
                    {"cpl": 1200},
                ],
            },
        ]

        summary = summarize_player(rows)

        self.assertEqual(summary["attempted_positions"], 2)
        self.assertEqual(summary["model_legal_moves"], 5)
        self.assertEqual(summary["model_illegal_attempts"], 2)
        self.assertAlmostEqual(summary["model_legal_pct"], 100 * 5 / 7)
        self.assertEqual(summary["model_forfeit_pct"], 50.0)
        self.assertEqual(summary["model_scored_moves"], 3)
        self.assertAlmostEqual(summary["model_avg_cpl"], 525.0)
        self.assertEqual(summary["model_300cp_blunders"], 2)
        self.assertAlmostEqual(summary["model_300cp_blunder_pct"], 100 * 2 / 3)


class OverlayReevaluationTests(unittest.TestCase):
    def test_filter_results_limits_players_and_positions(self) -> None:
        results = {
            "model-a": {
                "summary": {"prompt_history_replay": True},
                "results": [{"position_idx": 1}, {"position_idx": 2}],
                "token_usage": {"prompt": 10, "completion": 1},
            },
            "model-b": {
                "summary": {},
                "results": [{"position_idx": 1}],
                "token_usage": {"prompt": 20, "completion": 2},
            },
        }

        filtered = filter_results(results, players={"model-a"}, position_indices={2})

        self.assertEqual(list(filtered), ["model-a"])
        self.assertEqual(filtered["model-a"]["results"], [{"position_idx": 2}])
        self.assertEqual(filtered["model-a"]["summary"]["prompt_history_replay"], True)

    def test_partial_current_overlay_does_not_promote_legacy_equal_rows(self) -> None:
        positions = [
            {"type": "equal", "fen": "8/8/8/8/8/8/8/K6k w - - 0 1", "best_move": "a1b1"},
            {"type": "equal", "fen": "8/8/8/8/8/8/7k/K7 w - - 0 1", "best_move": "a1b1"},
        ]
        base = {
            "model-a": {
                "summary": {},
                "results": [
                    {"position_idx": 0, "fen": positions[0]["fen"], "model_move": "a1b1", "cpl": 0},
                    {"position_idx": 1, "fen": positions[1]["fen"], "model_move": "a1b1", "cpl": 0},
                ],
            }
        }
        overlay = {
            "model-a": {
                "summary": {},
                "results": [
                    {
                        "position_idx": 0,
                        "fen": positions[0]["fen"],
                        "model_move": "a1b1",
                        "cpl": 0,
                        "position_benchmark_version": CURRENT_BENCHMARK_VERSION,
                        "prompt_history_replay": True,
                        "stockfish_depth": 30,
                    }
                ],
            }
        }

        merged, _ = merge_overlays(base, [])
        merged["model-a"]["results"][0] = overlay["model-a"]["results"][0]
        refresh_summary("model-a", merged["model-a"], positions, min_stockfish_depth=30)
        readiness = benchmark_result_readiness(
            merged["model-a"],
            positions,
            min_equal_positions=2,
            min_stockfish_depth=30,
        )

        self.assertFalse(readiness.is_ready)
        self.assertEqual(readiness.reason, "stale equal-position rows")

    def test_recalculate_summary_groups_by_original_position_index(self) -> None:
        positions = [
            {"type": "blunder"},
            {"type": "equal"},
            {"type": "equal"},
        ]
        results = [
            {
                "position_idx": 1,
                "cpl": 10,
                "is_legal": True,
                "is_best": True,
                "avoided_blunder": True,
            },
            {
                "position_idx": 2,
                "cpl": 30,
                "is_legal": False,
                "is_best": False,
                "avoided_blunder": True,
            },
        ]

        summary = recalculate_summary("test-model", results, positions)

        self.assertEqual(summary["total_positions"], 2)
        self.assertEqual(summary["avg_cpl"], 20)
        self.assertEqual(summary["equal"]["total_positions"], 2)
        self.assertEqual(summary["equal"]["legal_moves"], 1)
        self.assertEqual(summary["equal"]["best_pct"], 50.0)

    def test_overlay_trust_does_not_make_legacy_rows_current(self) -> None:
        positions = [
            {
                "type": "equal",
                "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                "best_move": "a1b1",
            }
        ]
        player_data = {
            "summary": {},
            "results": [
                {
                    "position_idx": 0,
                    "fen": positions[0]["fen"],
                    "model_move": "a1b1",
                    "best_move": "a1b1",
                    "cpl": 0,
                }
            ],
        }

        *_, current_marker = inspect_player_results(
            "model-a",
            player_data,
            positions,
            {0},
        )

        self.assertFalse(current_marker)


if __name__ == "__main__":
    unittest.main()

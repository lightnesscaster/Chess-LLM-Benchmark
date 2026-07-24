import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from pathlib import Path

import chess

from game.match_scheduler import MatchScheduler
from position_benchmark.acquisition import (
    SUPPORTED_AUTOMATIC_PANELS,
    load_acquisition_policy,
    load_acquisition_state,
    missing_panels,
)
from position_benchmark.predictions import (
    CURRENT_STABILITY_SELECTION_POLICY,
    CURRENT_STABILITY_POSITION_INDICES,
    CURRENT_STABILITY_PROBE_VERSION,
    CURRENT_BENCHMARK_VERSION,
    DEFAULT_MIN_BLUNDER_POSITIONS,
    DEFAULT_MIN_EQUAL_POSITIONS,
    DEFAULT_MIN_GAME_LIKE_POSITIONS,
    DEFAULT_MIN_STABILITY_POSITIONS,
    DEFAULT_MIN_STOCKFISH_DEPTH,
    PredictionReadiness,
    benchmark_result_readiness,
    combine_legality_metrics,
    collect_equal_position_coverage,
    collect_equal_position_metrics,
    collect_stability_probe_metrics,
    combine_equal_and_game_like_predictions,
    predict_rating_from_model_data,
    predict_rating_from_model_data_with_supplement,
    predict_rating_from_results,
    stability_probe_prediction_cap,
    stability_probe_readiness,
    survival_probability,
    two_strike_survival_probability,
)
from position_benchmark.layout import (
    BLUNDER_POSITIONS_PATH,
    BLUNDER_RESULTS_PATH,
    CORE_POSITIONS_PATH,
    CORE_RESULTS_PATH,
    GAME_LIKE_POSITIONS_PATH,
    GAME_LIKE_RESULTS_PATH,
    FAILURE_TRANSFER_MATRIX_PATH,
    FAILURE_TRANSFER_RESULTS_PATH,
    FAILURE_TRANSFER_SHORTLIST_PATH,
    LEGALITY_STRESS_POSITIONS_PATH,
    LEGALITY_STRESS_RESULTS_PATH,
    LEGACY_COMBINED_POSITIONS_PATH,
    MANIFEST_PATH,
    PROTOCOL_SEQUENCE_RESULTS_PATH,
    STABILITY_RESULTS_PATH,
    repo_relative,
)
from position_benchmark.run_benchmark import (
    build_ad_hoc_player_config,
    replay_position_board,
    test_llm_on_position as run_llm_on_position,
    validate_position_input,
)
from position_benchmark.run_benchmark import calculate_actual_benchmark_cost, estimate_selected_benchmark_cost
from position_benchmark.run_benchmark import planned_position_count
from position_benchmark.retry_protocol import (
    CONDITIONAL_RETRY_PROTOCOL_VERSION,
    conditional_retry_summary,
)
from position_benchmark.stability_cap_shadow import (
    POLICY_PATH as STABILITY_CAP_SHADOW_POLICY_PATH,
    SHADOW_LEDGER_PATH as STABILITY_CAP_SHADOW_LEDGER_PATH,
    calculate_candidate_predictions,
    record_shadow_prediction,
)
from position_benchmark.token_accounting import sum_result_row_tokens
from position_benchmark.verify_benchmark import verify_stability_results
from rating.cost_calculator import CostCalculator
from rating.glicko2 import PlayerRating
from rating.rating_store import RatingStore
from scripts.analyze_puzzle_predictions import UNAVAILABLE_PLAYER_IDS
from scripts.audit_position_benchmark_readiness import AD_HOC_PLAYER_HINTS, inspect_player_results
from scripts.analyze_depth30_stability_cap import main as run_depth30_cap_audit
from scripts.evaluate_stability_cap_holdout import evaluate as evaluate_cap_holdout
from scripts.promote_position_benchmark_overlays import merge_overlays, refresh_summary
from scripts.reevaluate_position_result_overlays import filter_results
from scripts.reevaluate_position_result_overlays import recalculate_summary
from scripts.run_stability_probe import (
    DEFAULT_POSITION_LIMIT,
    DEFAULT_PROBE_PLIES,
    DEFAULT_SCORE_DEPTH,
    PROTOCOL_SEQUENCE_POSITION_LIMIT,
    PROTOCOL_SEQUENCE_PROBE_PLIES,
    PROTOCOL_SEQUENCE_SELECTION_POLICY,
    PROTOCOL_SEQUENCE_VERSION,
    ProbeRunError,
    STABILITY_PROBE_VERSION,
    backfill_retry_metrics,
    derive_retry_metrics,
    pre_moves_from_position,
    rescore_existing_results,
    run_probe_for_player,
    save_player_record,
    score_continuation_moves_from_pgn,
    selected_positions,
    summarize_player,
    probe_token_usage,
    stamp_probe_record,
)
from scripts.merge_stability_probe_shards import merge_shards
from scripts.repair_stability_probe_rows import merge_repair_rows
from scripts.select_legality_stress_panel import select_panel


ROOT = Path(__file__).resolve().parents[1]


class ProductionContractTests(unittest.TestCase):
    def test_automatic_acquisition_backfills_supplements_after_core(self) -> None:
        policy = load_acquisition_policy()
        self.assertEqual(
            tuple(panel.name for panel in policy.panels),
            SUPPORTED_AUTOMATIC_PANELS,
        )
        self.assertEqual(
            [panel.planned_first_attempt_calls for panel in policy.panels],
            [50, 48, 32],
        )
        self.assertTrue(policy.defer_games_after_acquisition)

        scheduler = object.__new__(MatchScheduler)
        scheduler.players = {"core-only": object()}
        scheduler._llm_configs = {"core-only": {}}
        scheduler._acquisition_policy = policy
        scheduler._acquisition_state = {"core-only": {"core"}}

        self.assertTrue(scheduler._needs_position_benchmark("core-only"))
        self.assertEqual(
            [
                panel.name
                for panel in missing_panels(
                    "core-only",
                    policy,
                    scheduler._acquisition_state,
                )
            ],
            ["game_like", "continuation_stability"],
        )

        scheduler._acquisition_state["core-only"].update(
            {"game_like", "continuation_stability"}
        )
        self.assertFalse(scheduler._needs_position_benchmark("core-only"))

    def test_saved_gpt56_rows_satisfy_full_automatic_acquisition_policy(self) -> None:
        policy = load_acquisition_policy()
        state = load_acquisition_state(policy)
        expected = set(SUPPORTED_AUTOMATIC_PANELS)

        gpt56_players = {
            player_id for player_id in state if player_id.startswith("gpt-5.6-")
        }
        self.assertEqual(len(gpt56_players), 12)
        self.assertTrue(all(state[player_id] == expected for player_id in gpt56_players))

    def test_probe_scheduler_helpers_stamp_contract_and_sum_tokens(self) -> None:
        record = {
            "summary": {},
            "results": [
                {"tokens": {"prompt_tokens": 10, "completion_tokens": 2}},
                {"tokens": {"prompt_tokens": 20, "completion_tokens": 3}},
            ],
        }
        indexed_positions = [(0, {}), (12, {})]

        stamp_probe_record(
            record,
            player_id="model",
            positions_path=GAME_LIKE_POSITIONS_PATH,
            indexed_positions=indexed_positions,
            probe_plies=8,
            score_depth=30,
        )

        self.assertEqual(probe_token_usage(record), {"prompt": 30, "completion": 5})
        self.assertEqual(record["summary"]["selected_position_indices"], [0, 12])
        self.assertEqual(
            record["summary"]["stability_probe_version"],
            CURRENT_STABILITY_PROBE_VERSION,
        )
        self.assertTrue(
            all(row["score_depth"] == 30 for row in record["results"])
        )

    def test_scheduler_runs_missing_supplements_in_manifest_order(self) -> None:
        policy = load_acquisition_policy()
        scheduler = object.__new__(MatchScheduler)
        scheduler.players = {"core-only": object()}
        scheduler._llm_configs = {"core-only": {"model_name": "test/model"}}
        scheduler._acquisition_policy = policy
        scheduler._acquisition_state = {"core-only": {"core"}}
        scheduler._benchmark_completed = set()
        scheduler._games_played = {}
        scheduler.verbose = False
        scheduler.rating_store = SimpleNamespace(
            get=lambda _player_id: SimpleNamespace(
                rating=1000,
                rating_deviation=300,
                games_rd=350,
                games_played=0,
                wins=0,
                losses=0,
                draws=0,
            ),
            refresh_benchmark_predictions=lambda: None,
        )
        scheduler._freeze_checker = SimpleNamespace(
            is_frozen=lambda _player_id, _rd: False
        )
        scheduler._calculate_priority = lambda _player_id: 1.0
        scheduler._estimate_benchmark_cost = lambda _player_id, _panel: 0.01
        scheduler._actual_benchmark_cost = (
            lambda _player_id, _tokens, model_calls: 0.01
        )

        calls: list[str] = []
        shadow_calls: list[str] = []
        scheduler._record_stability_cap_shadow = shadow_calls.append

        async def fake_positions(**kwargs):
            calls.append(kwargs["results_path"].name)
            return {
                "success": True,
                "summary": {"avg_cpl": 10.0, "legal_pct": 100.0},
                "token_usage": {"prompt": 1, "completion": 1},
                "model_calls": 48,
            }

        async def fake_continuation(**_kwargs):
            calls.append("stability.json")
            return {
                "success": True,
                "summary": {"model_avg_cpl": 20.0, "model_legal_pct": 100.0},
                "token_usage": {"prompt": 1, "completion": 1},
                "model_calls": 32,
            }

        engine = SimpleNamespace(quit=lambda: None)
        counters = {"total_cost": 0.0}
        with (
            patch(
                "position_benchmark.run_benchmark.run_benchmark_for_scheduler",
                side_effect=fake_positions,
            ),
            patch(
                "scripts.run_stability_probe.run_probe_for_scheduler",
                side_effect=fake_continuation,
            ),
            patch(
                "game.match_scheduler.panel_readiness",
                return_value=PredictionReadiness(True, "ready"),
            ),
            patch("chess.engine.SimpleEngine.popen_uci", return_value=engine),
        ):
            deferred = asyncio.run(
                scheduler._run_position_benchmarks(
                    ["core-only"],
                    counters,
                    max_cost=10.0,
                )
            )

        self.assertEqual(calls, ["game_like.json", "stability.json"])
        self.assertEqual(deferred, {"core-only"})
        self.assertEqual(
            scheduler._acquisition_state["core-only"],
            set(SUPPORTED_AUTOMATIC_PANELS),
        )
        self.assertIn("core-only", scheduler._benchmark_completed)
        self.assertEqual(shadow_calls, ["core-only"])

    def test_legality_stress_candidate_is_selected_without_gpt56_leakage(self) -> None:
        positions = json.loads(CORE_POSITIONS_PATH.read_text())["positions"]
        results = json.loads(CORE_RESULTS_PATH.read_text())
        live_audit = json.loads(
            (
                ROOT
                / "position_benchmark"
                / "validation"
                / "2026-07-14-gpt56-game-retry-audit.json"
            ).read_text()
        )

        panel = select_panel(
            positions,
            results,
            holdout_prefix="gpt-5.6-",
            min_family_illegal_rate=0.25,
            live_audit=live_audit,
        )
        metadata = panel["metadata"]

        self.assertGreater(len(panel["positions"]), 0)
        self.assertFalse(metadata["selection_uses_holdout"])
        self.assertTrue(
            all(
                not player_id.startswith("gpt-5.6-")
                for player_id in metadata["selection_models"]
            )
        )
        self.assertTrue(
            all(
                row["family_balanced_illegal_rate"] >= 0.25
                for row in metadata["position_scores"]
            )
        )
        validation = metadata["held_out_validation"]
        self.assertGreater(validation["selected_attempts"], 0)
        self.assertGreater(len(validation["holdout_models"]), 0)

    def test_failure_transfer_matrix_excludes_each_target_base_model(self) -> None:
        matrix = json.loads(FAILURE_TRANSFER_MATRIX_PATH.read_text())
        metadata = matrix["metadata"]

        self.assertEqual(metadata["screen_version"], "failure-transfer-screen-v1")
        self.assertEqual(metadata["status"], "frozen-before-target-calls")
        self.assertEqual(metadata["selected_failure_count"], 12)
        self.assertEqual(metadata["matched_control_count"], 12)
        self.assertEqual(metadata["planned_first_attempt_calls"], 48)

        failures = {
            row["candidate_id"]: row for row in matrix["selected_failures"]
        }
        controls = {
            row["candidate_id"]: row for row in matrix["matched_controls"]
        }
        self.assertEqual(
            {row["base_model"] for row in failures.values()},
            {"luna", "terra", "sol"},
        )
        for target, position_ids in matrix["test_matrix"].items():
            target_family = next(
                family for family in ("luna", "terra", "sol") if family in target
            )
            self.assertEqual(len(position_ids), 16)
            self.assertEqual(sum(position_id.startswith("failure-") for position_id in position_ids), 8)
            self.assertEqual(sum(position_id.startswith("control-") for position_id in position_ids), 8)
            for position_id in position_ids:
                source = failures.get(position_id) or controls[position_id]
                self.assertNotEqual(source["base_model"], target_family)

        for target_file in metadata["target_files"].values():
            data = json.loads((ROOT / target_file).read_text())
            self.assertEqual(len(data["positions"]), 16)
            for position in data["positions"]:
                board = replay_position_board(position)
                self.assertEqual(
                    " ".join(board.fen().split()[:4]),
                    " ".join(position["fen"].split()[:4]),
                )

    def test_legality_stress_pilot_is_isolated_and_protocol_stamped(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text())
        panel = manifest["panels"]["legality_stress_candidate"]
        results = json.loads(LEGALITY_STRESS_RESULTS_PATH.read_text())

        self.assertEqual(panel["status"], "research-candidate")
        self.assertEqual(panel["production_effect"], "none")
        self.assertEqual(panel["positions"], repo_relative(LEGALITY_STRESS_POSITIONS_PATH))
        self.assertEqual(panel["results"], repo_relative(LEGALITY_STRESS_RESULTS_PATH))
        self.assertEqual(
            set(results),
            {
                "gpt-5.6-luna (medium)",
                "gpt-5.6-terra (low)",
                "gpt-5.6-sol (high)",
            },
        )

        retry_attempts = 0
        retry_recoveries = 0
        for model_data in results.values():
            rows = model_data["results"]
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(row["panel"] == "legality-stress" for row in rows))
            self.assertTrue(all(row["prompt_history_replay"] for row in rows))
            self.assertTrue(all(row["stockfish_depth"] == 30 for row in rows))
            self.assertTrue(
                all(
                    row["conditional_retry_protocol_version"]
                    == CONDITIONAL_RETRY_PROTOCOL_VERSION
                    for row in rows
                )
            )
            retry = model_data["summary"]["conditional_retry"]
            retry_attempts += retry["retry_attempts"]
            retry_recoveries += retry["retry_recoveries"]

        self.assertEqual(retry_attempts, 2)
        self.assertEqual(retry_recoveries, 2)

    def test_validation_rating_store_can_use_higher_rd_without_changing_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RatingStore(
                path=str(Path(tmpdir) / "ratings.json"),
                use_firestore=False,
                benchmark_seed_rd=300.0,
            )
            store._benchmark_predictions = {"validation-model": 1234.0}

            seeded = store.get("validation-model")

        self.assertEqual(seeded.rating, 1234.0)
        self.assertEqual(seeded.rating_deviation, 300.0)

    def test_validation_rating_store_can_disable_benchmark_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RatingStore(
                path=str(Path(tmpdir) / "ratings.json"),
                use_firestore=False,
                use_benchmark_predictions=False,
            )
            store.refresh_benchmark_predictions()

            seeded = store.get("unseeded-model")

        self.assertEqual(store._benchmark_predictions, {})
        self.assertEqual(seeded.rating, 1500.0)
        self.assertEqual(seeded.rating_deviation, 350.0)

    def test_local_rating_store_serialization_is_order_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = [
                Path(tmpdir) / "ratings-a.json",
                Path(tmpdir) / "ratings-b.json",
            ]
            orders = [
                ("model-b", "model-a"),
                ("model-a", "model-b"),
            ]
            for path, order in zip(paths, orders):
                store = RatingStore(
                    path=str(path),
                    use_firestore=False,
                    use_benchmark_predictions=False,
                )
                for player_id in order:
                    store.set(
                        PlayerRating(
                            player_id=player_id,
                            rating=1000,
                            last_updated="2026-01-01T00:00:00+00:00",
                        ),
                        auto_save=False,
                    )
                store.save()

            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())

    def test_depth30_cap_audit_report_matches_current_evidence_gate(self) -> None:
        audit_path = (
            ROOT
            / "position_benchmark/validation/2026-07-21-depth30-stability-cap-analysis.json"
        )
        report_path = audit_path.with_suffix(".md")
        audit = json.loads(audit_path.read_text())
        report = report_path.read_text()

        self.assertTrue(audit["evidence_gate"]["passed"])
        self.assertGreaterEqual(
            audit["configuration_count"],
            audit["evidence_gate"]["minimum_configurations"],
        )
        self.assertGreaterEqual(
            audit["family_count"],
            audit["evidence_gate"]["minimum_families"],
        )
        self.assertEqual(
            audit["production_decision"]["status"],
            "hold_current_production",
        )
        self.assertGreaterEqual(audit["lab_count"], 2)
        self.assertIn("The acquisition gate now passes", report)
        self.assertNotIn("both coverage checks fail", report)

        candidate = audit["production_decision"]["leading_candidate"]
        influential_family = audit["production_decision"]["influential_family"]
        for target in ("rd300", "no_position"):
            influence = audit["targets"][target]["candidates"][candidate][
                "family_influence"
            ]
            self.assertGreater(
                influence["leave_one_family_out_mae_delta"][influential_family],
                0.0,
            )
            lab_result = audit["targets"][target]["candidates"][candidate]
            self.assertIn("lab_bootstrap", lab_result)
            self.assertIn("lab_influence", lab_result)

        influential_lab = audit["production_decision"]["influential_lab"]
        rd_lab_influence = audit["targets"]["rd300"]["candidates"][candidate][
            "lab_influence"
        ]
        self.assertAlmostEqual(
            rd_lab_influence["leave_one_lab_out_mae_delta"][influential_lab],
            0.0,
        )

    def test_depth30_cap_development_freeze_matches_audit(self) -> None:
        freeze_path = (
            ROOT
            / "position_benchmark/validation/2026-07-23-depth30-cap-development-freeze.json"
        )
        freeze = json.loads(freeze_path.read_text())
        audit_path = ROOT / freeze["development_audit"]["path"]
        audit_bytes = audit_path.read_bytes()
        audit = json.loads(audit_bytes)

        self.assertEqual(freeze["status"], "frozen-development")
        self.assertEqual(freeze["production_effect"], "none")
        self.assertEqual(
            freeze["development_audit"]["sha256"],
            hashlib.sha256(audit_bytes).hexdigest(),
        )
        self.assertEqual(
            set(freeze["development_configuration_ids"]),
            {row["player_id"] for row in audit["rows"]},
        )
        self.assertEqual(
            freeze["candidate_order"],
            audit["candidate_order"],
        )

    def test_frozen_depth30_development_audit_refuses_implicit_overwrite(self) -> None:
        with patch("sys.argv", ["analyze_depth30_stability_cap.py"]):
            with self.assertRaisesRegex(
                SystemExit,
                "development cohort is frozen",
            ):
                run_depth30_cap_audit()

    def test_shadow_cap_candidates_match_frozen_development_audit(self) -> None:
        audit_path = (
            ROOT
            / "position_benchmark/validation/2026-07-21-depth30-stability-cap-analysis.json"
        )
        audit = json.loads(audit_path.read_text())
        core_results = json.loads(CORE_RESULTS_PATH.read_text())
        core_positions = json.loads(CORE_POSITIONS_PATH.read_text())["positions"]
        game_results = json.loads(GAME_LIKE_RESULTS_PATH.read_text())
        game_positions = json.loads(GAME_LIKE_POSITIONS_PATH.read_text())["positions"]
        stability_results = json.loads(STABILITY_RESULTS_PATH.read_text())
        blunder_results = json.loads(BLUNDER_RESULTS_PATH.read_text())
        blunder_positions = json.loads(BLUNDER_POSITIONS_PATH.read_text())["positions"]

        self.assertEqual(
            audit["production_reference_candidate"],
            "deduplicated_move_exposure_cap",
        )
        for row in audit["rows"]:
            player_id = row["player_id"]
            shadow = calculate_candidate_predictions(
                core_record=core_results[player_id],
                core_positions=core_positions,
                game_like_record=game_results[player_id],
                game_like_positions=game_positions,
                stability_record=stability_results[player_id],
                blunder_record=blunder_results.get(player_id),
                blunder_positions=blunder_positions,
            )
            for candidate, expected in row["candidates"].items():
                self.assertAlmostEqual(
                    shadow["candidates"][candidate],
                    expected,
                )

    def test_shadow_prediction_ledger_is_append_only(self) -> None:
        frozen = {
            "development_configuration_ids": [],
        }
        first_record = {
            "player_id": "future-model",
            "prediction_locked": True,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / "shadow.json"
            with (
                patch(
                    "position_benchmark.stability_cap_shadow._load_policy_and_freeze",
                    return_value=({}, frozen),
                ),
                patch(
                    "position_benchmark.stability_cap_shadow.build_shadow_record",
                    return_value=first_record,
                ) as build,
            ):
                saved, created = record_shadow_prediction(
                    "future-model",
                    ledger_path=ledger_path,
                )
                repeated, repeated_created = record_shadow_prediction(
                    "future-model",
                    ledger_path=ledger_path,
                )

        self.assertTrue(created)
        self.assertFalse(repeated_created)
        self.assertEqual(saved, first_record)
        self.assertEqual(repeated, first_record)
        build.assert_called_once()

    def test_shadow_policy_is_prospective_and_has_no_production_effect(self) -> None:
        policy = json.loads(STABILITY_CAP_SHADOW_POLICY_PATH.read_text())
        ledger = json.loads(STABILITY_CAP_SHADOW_LEDGER_PATH.read_text())
        freeze_path = ROOT / policy["development_freeze"]["path"]

        self.assertEqual(policy["production_effect"], "none")
        self.assertEqual(
            policy["comparison"]["reference_candidate"],
            "deduplicated_move_exposure_cap",
        )
        self.assertEqual(
            policy["comparison"]["challenger_candidate"],
            "repeated_forfeit_only",
        )
        self.assertTrue(
            policy["comparison"]["evaluate_affected_configurations_only"]
        )
        self.assertGreaterEqual(
            policy["coverage_gate"]["minimum_affected_labs"],
            4,
        )
        self.assertEqual(
            hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
            policy["development_freeze"]["sha256"],
        )
        self.assertEqual(ledger["production_effect"], "none")
        self.assertEqual(ledger["entries"], {})

    def test_shadow_holdout_requires_and_can_pass_fixed_gates(self) -> None:
        policy = {
            "policy_version": "test-v1",
            "primary_target": {
                "minimum_games": 8,
                "maximum_games_rd": 200,
            },
            "comparison": {
                "reference_candidate": "production",
                "challenger_candidate": "challenger",
                "evaluate_affected_configurations_only": True,
            },
            "coverage_gate": {
                "minimum_mature_holdout_configurations": 12,
                "minimum_affected_holdout_configurations": 8,
                "minimum_affected_families": 6,
                "minimum_affected_labs": 4,
            },
            "promotion_gate": {
                "minimum_family_bootstrap_probability_mae_improves": 0.95,
                "minimum_lab_bootstrap_probability_mae_improves": 0.95,
                "minimum_mae_improvement_elo": 10,
                "maximum_rmse_delta_elo": 0,
                "maximum_absolute_bias_increase_elo": 25,
                "maximum_leave_one_lab_out_mae_delta_elo": 10,
            },
            "bootstrap": {
                "resamples": 100,
                "random_seed": 42,
            },
        }
        ledger = {
            "entries": {},
        }
        ratings = {}
        for index in range(12):
            player_id = f"holdout-{index}"
            affected = index < 8
            ledger["entries"][player_id] = {
                "family": f"family-{index % 6}",
                "lab": f"lab-{index % 4}",
                "recorded_at": "2026-07-24T00:00:00+00:00",
                "eligibility": {"prospective_holdout": True},
                "candidates": {
                    "production": 1000.0,
                    "challenger": 1100.0 if affected else 1000.0,
                },
            }
            ratings[player_id] = {
                "rating": 1100.0 if affected else 1000.0,
                "games_played": 10,
                "games_rd": 100.0,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            policy_path = tmpdir_path / "policy.json"
            ledger_path = tmpdir_path / "ledger.json"
            ratings_path = tmpdir_path / "ratings.json"
            policy_path.write_text(json.dumps(policy))
            ledger_path.write_text(json.dumps(ledger))
            ratings_path.write_text(json.dumps(ratings))
            analysis = evaluate_cap_holdout(
                policy_path=policy_path,
                ledger_path=ledger_path,
                ratings_path=ratings_path,
            )

        self.assertEqual(analysis["status"], "promotion-candidate")
        self.assertTrue(analysis["coverage"]["passed"])
        self.assertTrue(analysis["promotion_passed"])
        self.assertEqual(
            analysis["coverage"]["affected_mature_configurations"],
            8,
        )

    def test_canonical_panel_sizes_and_defaults_match_saved_artifacts(self) -> None:
        positions_data = json.loads(CORE_POSITIONS_PATH.read_text())
        positions = positions_data["positions"]
        blunder_positions = json.loads(BLUNDER_POSITIONS_PATH.read_text())["positions"]
        game_like_data = json.loads(GAME_LIKE_POSITIONS_PATH.read_text())
        manifest = json.loads(MANIFEST_PATH.read_text())

        self.assertEqual(CURRENT_BENCHMARK_VERSION, "history-replay-v2")
        self.assertEqual(DEFAULT_MIN_EQUAL_POSITIONS, 50)
        self.assertEqual(DEFAULT_MIN_BLUNDER_POSITIONS, 25)
        self.assertEqual(DEFAULT_MIN_GAME_LIKE_POSITIONS, 48)
        self.assertEqual(DEFAULT_MIN_STABILITY_POSITIONS, 8)
        self.assertEqual(DEFAULT_MIN_STOCKFISH_DEPTH, 30)
        self.assertEqual(len(positions), 50)
        self.assertTrue(all(p.get("type") == "equal" for p in positions))
        self.assertEqual(len(blunder_positions), 25)
        self.assertTrue(all(p.get("type") == "blunder" for p in blunder_positions))
        self.assertEqual(len(game_like_data["positions"]), 48)
        for position in [p for p in positions if p.get("type") == "equal"]:
            board = replay_position_board(position)
            self.assertGreater(len(board.move_stack), 0)
            self.assertEqual(
                " ".join(board.fen().split()[:4]),
                " ".join(position["fen"].split()[:4]),
            )
        for position in game_like_data["positions"]:
            board = replay_position_board(position)
            self.assertGreater(len(board.move_stack), 0)
            self.assertEqual(
                " ".join(board.fen().split()[:4]),
                " ".join(position["fen"].split()[:4]),
            )
        self.assertEqual(DEFAULT_POSITION_LIMIT, 8)
        self.assertEqual(DEFAULT_PROBE_PLIES, 8)
        self.assertEqual(DEFAULT_SCORE_DEPTH, 30)
        self.assertEqual(PROTOCOL_SEQUENCE_POSITION_LIMIT, 4)
        self.assertEqual(PROTOCOL_SEQUENCE_PROBE_PLIES, 16)
        self.assertEqual(STABILITY_PROBE_VERSION, CURRENT_STABILITY_PROBE_VERSION)
        self.assertEqual(manifest["panels"]["core"]["position_count"], 50)
        self.assertEqual(
            manifest["conditional_retry_protocol_version"],
            CONDITIONAL_RETRY_PROTOCOL_VERSION,
        )
        self.assertEqual(
            manifest["panels"]["core"]["model_call_count"],
            {
                "base_first_attempt_calls": 50,
                "conditional_retry_calls_min": 0,
                "conditional_retry_calls_max": 50,
                "total_calls_min": 50,
                "total_calls_max": 100,
            },
        )
        self.assertEqual(
            manifest["panels"]["game_like"]["model_call_count"]["total_calls_max"],
            96,
        )
        continuation = manifest["panels"]["continuation_stability"]
        self.assertEqual(continuation["stockfish_score_depth"], 30)
        self.assertEqual(
            continuation["selected_position_indices"],
            list(CURRENT_STABILITY_POSITION_INDICES),
        )
        self.assertEqual(
            continuation["stability_probe_version"],
            CURRENT_STABILITY_PROBE_VERSION,
        )
        self.assertEqual(
            continuation["catastrophe_event_policy"],
            "first-1000cp-event-per-start-v1",
        )
        self.assertEqual(
            continuation["catastrophe_rate_denominator"], "scored_model_moves"
        )
        self.assertEqual(manifest["panels"]["blunder"]["status"], "optional-historical")
        sequence = manifest["panels"]["continuation_sequence_candidate"]
        self.assertEqual(sequence["status"], "research-candidate")
        self.assertEqual(sequence["production_effect"], "none")
        self.assertEqual(sequence["stability_probe_version"], PROTOCOL_SEQUENCE_VERSION)
        self.assertEqual(sequence["selection_policy"], PROTOCOL_SEQUENCE_SELECTION_POLICY)
        self.assertEqual(sequence["probe_plies"], PROTOCOL_SEQUENCE_PROBE_PLIES)
        self.assertEqual(sequence["planned_base_first_attempt_calls"], 32)
        self.assertEqual(sequence["results"], repo_relative(PROTOCOL_SEQUENCE_RESULTS_PATH))
        transfer = manifest["panels"]["failure_transfer_screen"]
        self.assertEqual(transfer["production_effect"], "none")
        self.assertEqual(transfer["matrix"], repo_relative(FAILURE_TRANSFER_MATRIX_PATH))
        self.assertEqual(transfer["results"], repo_relative(FAILURE_TRANSFER_RESULTS_PATH))
        self.assertEqual(transfer["planned_base_first_attempt_calls"], 48)
        self.assertEqual(transfer["stockfish_depth"], 30)
        acquisition = manifest["automatic_acquisition"]
        self.assertTrue(acquisition["enabled"])
        self.assertEqual(
            acquisition["panel_order"],
            ["core", "game_like", "continuation_stability"],
        )
        self.assertTrue(acquisition["defer_games_after_acquisition"])
        prospective = manifest["prospective_stability_cap_validation"]
        self.assertEqual(prospective["production_effect"], "none")
        self.assertEqual(
            prospective["recording_timing"],
            "after-current-automatic-suite-before-first-game",
        )
        self.assertEqual(
            prospective["reference_candidate"],
            "deduplicated_move_exposure_cap",
        )
        self.assertEqual(
            manifest["panels"]["continuation_stability"][
                "planned_base_first_attempt_calls"
            ],
            32,
        )
        shortlist = manifest["panels"]["failure_transfer_positive_shortlist"]
        self.assertEqual(shortlist["production_effect"], "none")
        self.assertEqual(shortlist["positions"], repo_relative(FAILURE_TRANSFER_SHORTLIST_PATH))
        self.assertEqual(shortlist["stockfish_depth"], 30)
        self.assertEqual(
            shortlist["required_next_gate"],
            "held-out-non-gpt56-model-families",
        )
        self.assertEqual(
            shortlist["status"],
            "research-only-heldout-gate-failed",
        )
        heldout = manifest["panels"]["failure_transfer_heldout"]
        self.assertEqual(heldout["status"], "complete-no-promotion")
        self.assertEqual(heldout["production_effect"], "none")
        self.assertEqual(heldout["planned_base_first_attempt_calls"], 48)
        self.assertEqual(heldout["actual_model_calls"], 58)
        self.assertFalse(heldout["primary_result"]["promotion_gate_passed"])
        self.assertEqual(
            heldout["primary_result"]["one_sided_exact_mcnemar_p"],
            0.34375,
        )

    def test_manifest_paths_match_runtime_paths(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text())

        self.assertEqual(manifest["panels"]["core"]["positions"], repo_relative(CORE_POSITIONS_PATH))
        self.assertEqual(manifest["panels"]["core"]["results"], repo_relative(CORE_RESULTS_PATH))
        self.assertEqual(
            manifest["panels"]["game_like"]["positions"],
            repo_relative(GAME_LIKE_POSITIONS_PATH),
        )
        self.assertEqual(
            manifest["panels"]["game_like"]["results"],
            repo_relative(GAME_LIKE_RESULTS_PATH),
        )
        self.assertEqual(
            manifest["panels"]["blunder"]["positions"],
            repo_relative(BLUNDER_POSITIONS_PATH),
        )
        self.assertEqual(
            manifest["panels"]["blunder"]["results"],
            repo_relative(BLUNDER_RESULTS_PATH),
        )
        self.assertEqual(
            manifest["panels"]["continuation_stability"]["results"],
            repo_relative(STABILITY_RESULTS_PATH),
        )

    def test_legacy_combined_panel_requires_explicit_override(self) -> None:
        legacy_data = json.loads(LEGACY_COMBINED_POSITIONS_PATH.read_text())

        with self.assertRaisesRegex(ValueError, "inactive legacy position input"):
            validate_position_input(legacy_data)

        validate_position_input(legacy_data, allow_legacy_input=True)

    def test_frozen_june_validation_snapshot_is_self_consistent(self) -> None:
        snapshot = json.loads(
            (ROOT / "position_benchmark" / "validation" / "2026-06-23.json").read_text()
        )

        self.assertEqual(snapshot["benchmark_version"], CURRENT_BENCHMARK_VERSION)
        self.assertEqual(snapshot["core"]["positions"], DEFAULT_MIN_EQUAL_POSITIONS)
        self.assertEqual(snapshot["core"]["stockfish_depth"], DEFAULT_MIN_STOCKFISH_DEPTH)
        for row in snapshot["gpt_5_5"]:
            self.assertAlmostEqual(
                row["actual_rating"] - row["predicted_rating"],
                row["error"],
                places=5,
            )

    def test_core_migration_preserves_every_model_prediction(self) -> None:
        legacy_positions = json.loads(LEGACY_COMBINED_POSITIONS_PATH.read_text())["positions"]
        legacy_results = json.loads(
            (ROOT / "position_benchmark" / "legacy" / "combined_results_75.json").read_text()
        )
        core_positions = json.loads(CORE_POSITIONS_PATH.read_text())["positions"]
        core_results = json.loads(CORE_RESULTS_PATH.read_text())

        self.assertLessEqual(set(legacy_results), set(core_results))
        rerun_after_migration = {
            "deepseek-v4-pro (no thinking)",
            "deepseek-v4-flash (max)",
            "gemma-4-31b-it (high)",
            "gemma-4-31b-it (no thinking)",
            "gpt-3.5-turbo",
            "gpt-5.5 (xhigh)",
            "kimi-k2-0905",
            "llama-4-maverick",
            "mistral-medium-3",
            "qwen3-235b-a22b-2507",
        }
        for player_id, legacy_data in legacy_results.items():
            if player_id in rerun_after_migration:
                continue
            legacy_prediction = predict_rating_from_results(
                legacy_data["results"],
                legacy_positions,
            )
            migrated_prediction = predict_rating_from_results(
                core_results[player_id]["results"],
                core_positions,
            )
            self.assertAlmostEqual(legacy_prediction, migrated_prediction, places=9)

    def test_legacy_combined_indices_resolve_against_new_core(self) -> None:
        core_positions = json.loads(CORE_POSITIONS_PATH.read_text())["positions"]
        legacy_rows = [
            {
                "position_idx": position["legacy_position_idx"],
                "fen": position["fen"],
                "model_move": position["best_move"],
                "best_move": position["best_move"],
                "cpl": 0,
                "is_legal": True,
                "position_benchmark_version": CURRENT_BENCHMARK_VERSION,
                "prompt_history_replay": True,
                "stockfish_depth": 30,
            }
            for position in core_positions
        ]

        readiness = benchmark_result_readiness({"results": legacy_rows}, core_positions)

        self.assertTrue(readiness.is_ready, readiness.reason)


class CostEstimateTests(unittest.TestCase):
    def test_refresh_retry_evidence_selects_only_unmeasured_rows(self) -> None:
        all_results = {
            "model-a": {
                "results": [
                    {
                        "position_idx": 0,
                        "conditional_retry_protocol_version": (
                            CONDITIONAL_RETRY_PROTOCOL_VERSION
                        ),
                    },
                    {"position_idx": 1},
                ]
            }
        }
        index_map = {0: 0, 1: 1, 2: 2}

        refresh_count = planned_position_count(
            "model-a",
            selected_count=3,
            type_filter_idx_map=index_map,
            all_results=all_results,
            retry_missing=False,
            refresh_retry_evidence=True,
        )
        missing_count = planned_position_count(
            "model-a",
            selected_count=3,
            type_filter_idx_map=index_map,
            all_results=all_results,
            retry_missing=True,
        )

        self.assertEqual(refresh_count, 2)
        self.assertEqual(missing_count, 1)

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
            self.assertAlmostEqual(cost, 0.303)

    def test_position_cost_estimate_uses_same_player_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pricing_path = root / "pricing.json"
            config_path = root / "benchmark.yaml"
            results_path = root / "position_benchmark" / "results"
            results_path.mkdir(parents=True)
            pricing_path.write_text(
                json.dumps(
                    {"provider/model": {"prompt": 1e-6, "completion": 1e-5}}
                )
            )
            config_path.write_text(
                "\n".join(
                    [
                        "llms:",
                        '  - player_id: "model-a"',
                        '    model_name: "provider/model"',
                        "    reasoning_effort: xhigh",
                    ]
                )
            )
            (results_path / "core.json").write_text(
                json.dumps(
                    {
                        "model-a": {
                            "results": [
                                {
                                    "prompt_tokens": 1000,
                                    "completion_tokens": 20000,
                                    "retry_attempted": False,
                                }
                            ]
                        }
                    }
                )
            )
            calculator = CostCalculator(
                pricing_path=pricing_path,
                config_path=config_path,
            )

            estimate = calculator.position_benchmark_token_estimate(
                "model-a",
                model_name="provider/model",
                reasoning=True,
            )
            cost = calculator.estimate_position_benchmark_cost(
                "model-a",
                model_name="provider/model",
                num_positions=10,
                reasoning=True,
            )

            self.assertEqual(estimate["source"], "same-player historical maximum")
            self.assertEqual(estimate["completion_tokens_per_call"], 20000)
            self.assertAlmostEqual(cost, 2.01)

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
                num_model_calls=7,
            )

            self.assertAlmostEqual(cost, 0.14)


class PositionReplayTests(unittest.TestCase):
    def test_replays_san_history_from_blunder_positions(self) -> None:
        with BLUNDER_POSITIONS_PATH.open() as f:
            positions = json.load(f)["positions"]

        position = positions[0]
        board = replay_position_board(position)

        self.assertGreater(len(board.move_stack), 0)
        self.assertEqual(
            " ".join(board.fen().split()[:4]),
            " ".join(position["fen"].split()[:4]),
        )

    def test_replays_uci_history_from_equal_positions(self) -> None:
        with CORE_POSITIONS_PATH.open() as f:
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


class FakeSequencePlayer:
    def __init__(
        self,
        responses: list[str],
        usage: list[tuple[int, int]] | None = None,
    ) -> None:
        self.responses = list(responses)
        self.usage = list(usage or [(0, 0)] * len(responses))
        self.calls: list[dict] = []
        self.prompt_tokens = 0
        self.completion_tokens = 0

    async def select_move(self, _board, **kwargs) -> str:
        self.calls.append(kwargs)
        self._last_prompt_tokens, self._last_completion_tokens = self.usage.pop(0)
        self.prompt_tokens += self._last_prompt_tokens
        self.completion_tokens += self._last_completion_tokens
        return self.responses.pop(0)


class FakeAnalysisEngine:
    def analyse(self, board, _limit) -> dict:
        return {
            "score": chess.engine.PovScore(chess.engine.Cp(0), board.turn),
        }


class PositionBenchmarkResultTests(unittest.IsolatedAsyncioTestCase):
    async def test_illegal_llm_move_uses_position_fen(self) -> None:
        with CORE_POSITIONS_PATH.open() as f:
            positions = json.load(f)["positions"]

        position = next(p for p in positions if p.get("type") == "equal")
        player = FakeSequencePlayer(["a1a1", "a1a1"])
        result = await run_llm_on_position(player, position, engine=None, depth=1)

        self.assertEqual(result.fen, position["fen"])
        self.assertFalse(result.is_legal)
        self.assertEqual(result.model_move, "a1a1")
        self.assertEqual(result.cpl, position["eval_before"] + 5000)
        self.assertTrue(result.retry_attempted)
        self.assertFalse(result.retry_is_legal)
        self.assertEqual(len(player.calls), 2)

    async def test_illegal_first_move_uses_exact_production_retry_arguments(self) -> None:
        with CORE_POSITIONS_PATH.open() as f:
            position = json.load(f)["positions"][0]
        player = FakeSequencePlayer(
            ["  NOT-A-MOVE  ", position["best_move"]],
            usage=[(100, 10), (120, 12)],
        )

        result = await run_llm_on_position(player, position, engine=None, depth=1)

        self.assertFalse(result.is_legal)
        self.assertEqual(result.model_move, "NOT-A-MOVE")
        self.assertTrue(result.retry_attempted)
        self.assertTrue(result.retry_is_legal)
        self.assertEqual(result.retry_move, position["best_move"])
        self.assertEqual(
            player.calls,
            [
                {"is_retry": False, "last_move_illegal": None},
                {"is_retry": True, "last_move_illegal": "NOT-A-MOVE"},
            ],
        )
        self.assertEqual(result.prompt_tokens, 220)
        self.assertEqual(result.completion_tokens, 22)
        self.assertEqual(result.initial_prompt_tokens, 100)
        self.assertEqual(result.retry_prompt_tokens, 120)
        self.assertEqual(
            result.conditional_retry_protocol_version,
            CONDITIONAL_RETRY_PROTOCOL_VERSION,
        )

    async def test_legal_first_move_does_not_add_retry_call(self) -> None:
        with CORE_POSITIONS_PATH.open() as f:
            position = json.load(f)["positions"][0]
        player = FakeSequencePlayer([position["best_move"]], usage=[(100, 10)])

        result = await run_llm_on_position(
            player,
            position,
            engine=FakeAnalysisEngine(),
            depth=1,
        )

        self.assertTrue(result.is_legal)
        self.assertFalse(result.retry_attempted)
        self.assertIsNone(result.retry_is_legal)
        self.assertEqual(len(player.calls), 1)
        self.assertEqual(result.prompt_tokens, 100)
        self.assertEqual(result.completion_tokens, 10)

    async def test_empty_first_response_retries_as_invalid_like_game_runner(self) -> None:
        with CORE_POSITIONS_PATH.open() as f:
            position = json.load(f)["positions"][0]
        player = FakeSequencePlayer(["", position["best_move"]])

        result = await run_llm_on_position(player, position, engine=None, depth=1)

        self.assertFalse(result.is_legal)
        self.assertTrue(result.retry_is_legal)
        self.assertEqual(player.calls[1]["last_move_illegal"], "invalid")

    def test_conditional_retry_summary_preserves_first_attempt_metrics(self) -> None:
        rows = [
            {
                "is_legal": True,
                "conditional_retry_protocol_version": CONDITIONAL_RETRY_PROTOCOL_VERSION,
                "retry_attempted": False,
                "retry_is_legal": None,
            },
            {
                "is_legal": False,
                "conditional_retry_protocol_version": CONDITIONAL_RETRY_PROTOCOL_VERSION,
                "retry_attempted": True,
                "retry_is_legal": True,
            },
            {
                "is_legal": False,
                "conditional_retry_protocol_version": CONDITIONAL_RETRY_PROTOCOL_VERSION,
                "retry_attempted": True,
                "retry_is_legal": False,
            },
            {"is_legal": False},
        ]

        summary = conditional_retry_summary(rows)

        self.assertIsNotNone(summary)
        self.assertEqual(summary["measured_positions"], 3)
        self.assertEqual(summary["initial_illegal_moves"], 2)
        self.assertEqual(summary["retry_attempts"], 2)
        self.assertEqual(summary["retry_recoveries"], 1)
        self.assertEqual(summary["retry_failures"], 1)
        self.assertEqual(summary["initial_illegals_without_retry"], 0)
        self.assertEqual(summary["recovery_pct"], 50.0)
        self.assertAlmostEqual(summary["post_retry_legal_pct"], 200 / 3)
        self.assertEqual(summary["total_model_calls"], 5)


class PredictionMetricTests(unittest.TestCase):
    def test_two_strike_survival_matches_runner_policy(self) -> None:
        self.assertAlmostEqual(
            two_strike_survival_probability(98.0, 0.0),
            survival_probability(98.0),
        )
        self.assertAlmostEqual(
            two_strike_survival_probability(98.0, 100.0),
            100.0 * 0.98**40,
        )
        self.assertGreater(
            two_strike_survival_probability(98.0, 10.0),
            two_strike_survival_probability(98.0, 90.0),
        )

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
                "stability_probe_version": CURRENT_STABILITY_PROBE_VERSION,
                "position_selection_policy": CURRENT_STABILITY_SELECTION_POLICY,
                "selected_position_indices": list(CURRENT_STABILITY_POSITION_INDICES),
                "attempted_positions": 4,
                "model_scored_moves": 12,
                "model_forfeit_pct": 50.0,
                "model_1000cp_catastrophe_pct": 25.0,
                "score_depth": 30,
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
                "stability_probe_version": CURRENT_STABILITY_PROBE_VERSION,
                "position_selection_policy": CURRENT_STABILITY_SELECTION_POLICY,
                "selected_position_indices": list(CURRENT_STABILITY_POSITION_INDICES),
                "attempted_positions": 8,
                "model_legal_moves": 24,
                "model_attempts": 24,
                "model_legal_pct": 100.0,
                "model_scored_moves": 24,
                "model_forfeit_pct": 25.0,
                "model_1000cp_catastrophe_pct": 12.5,
                "score_depth": 30,
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

    def test_stability_cap_deduplicates_catastrophes_within_one_trajectory(self) -> None:
        stability = {
            "summary": {
                "attempted_positions": 8,
                "model_scored_moves": 32,
                "model_legal_moves": 32,
                "model_attempts": 32,
                "model_legal_pct": 100.0,
                "model_forfeit_pct": 0.0,
                "model_1000cp_catastrophes": 3,
                "model_1000cp_catastrophe_pct": 9.375,
                "model_1000cp_catastrophe_positions": 1,
            }
        }

        self.assertIsNone(stability_probe_prediction_cap(stability))

        stability["summary"]["model_1000cp_catastrophe_positions"] = 2
        self.assertEqual(stability_probe_prediction_cap(stability), 600.0)

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
                "stability_probe_version": CURRENT_STABILITY_PROBE_VERSION,
                "position_selection_policy": CURRENT_STABILITY_SELECTION_POLICY,
                "selected_position_indices": list(CURRENT_STABILITY_POSITION_INDICES),
                "attempted_positions": 8,
                "model_legal_moves": 0,
                "model_attempts": 16,
                "model_scored_moves": 0,
                "model_forfeits": 8,
                "model_forfeit_pct": 100.0,
                "model_1000cp_catastrophe_pct": 0.0,
                "score_depth": 30,
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
        for player_id in [
            "gemini-2.0-flash-001",
            "grok-4-fast",
            "grok-4.1-fast",
            "mimo-v2-flash (no thinking)",
        ]:
            self.assertIn(player_id, UNAVAILABLE_PLAYER_IDS)
            self.assertTrue(AD_HOC_PLAYER_HINTS[player_id]["unavailable"])


class StabilityProbeTests(unittest.TestCase):
    def test_concurrent_player_saves_preserve_every_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "stability.json"
            player_ids = [f"model-{index}" for index in range(12)]

            def record(player_id: str) -> dict:
                return {
                    "summary": {
                        "player_id": player_id,
                        "attempted_positions": 1,
                        "api_errors": 0,
                        "score_depth": 30,
                        "selected_position_indices": [0],
                    },
                    "results": [
                        {
                            "position_idx": 0,
                            "termination": "max_moves",
                            "score_depth": 30,
                        }
                    ],
                }

            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = [
                    executor.submit(
                        save_player_record,
                        results_path,
                        player_id,
                        record(player_id),
                    )
                    for player_id in player_ids
                ]
                for future in futures:
                    future.result()

            saved = json.loads(results_path.read_text())
            self.assertEqual(set(saved), set(player_ids))
            for player_id in player_ids:
                self.assertEqual(saved[player_id]["summary"]["player_id"], player_id)

    def test_invalid_probe_cannot_replace_valid_canonical_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "stability.json"
            valid = {
                "summary": {
                    "attempted_positions": 1,
                    "api_errors": 0,
                    "score_depth": 30,
                    "selected_position_indices": [0],
                },
                "results": [
                    {
                        "position_idx": 0,
                        "termination": "max_moves",
                        "score_depth": 30,
                    }
                ],
            }
            save_player_record(results_path, "model", valid)
            failed = json.loads(json.dumps(valid))
            failed["summary"]["api_errors"] = 1
            failed["results"][0]["termination"] = "api_error"

            with self.assertRaisesRegex(ValueError, "API errors"):
                save_player_record(results_path, "model", failed)

            self.assertEqual(json.loads(results_path.read_text())["model"], valid)

    def test_probe_aborts_after_first_exhausted_api_failure(self) -> None:
        positions = json.loads(GAME_LIKE_POSITIONS_PATH.read_text())["positions"]
        indexed_positions = [(0, positions[0]), (1, positions[1])]
        player = SimpleNamespace(close=AsyncMock(), last_api_error="upstream failed")
        game_result = SimpleNamespace(
            termination="api_error",
            winner="draw",
            moves=0,
            illegal_moves_white=0,
            illegal_moves_black=0,
            total_moves_white=0,
            total_moves_black=0,
            tokens_white={},
            tokens_black={},
            illegal_move_details=[],
        )

        with (
            patch(
                "scripts.run_stability_probe.create_llm_player",
                return_value=player,
            ),
            patch(
                "scripts.run_stability_probe.GameRunner.play_game",
                new=AsyncMock(return_value=(game_result, "")),
            ) as play_game,
        ):
            with self.assertRaises(ProbeRunError):
                asyncio.run(
                    run_probe_for_player(
                        "model",
                        {"model_name": "test/model"},
                        indexed_positions,
                        probe_plies=8,
                        api_backend="openrouter",
                        random_seed=1729,
                        verbose=False,
                        respect_config_api=False,
                    )
                )

        self.assertEqual(play_game.await_count, 1)

    def test_layout_verifier_rejects_depth_10_replayable_stability_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "stability.json"
            results_path.write_text(
                json.dumps(
                    {
                        "model-a": {
                            "summary": {
                                "stability_probe_version": CURRENT_STABILITY_PROBE_VERSION,
                                "position_selection_policy": CURRENT_STABILITY_SELECTION_POLICY,
                                "selected_position_indices": list(
                                    CURRENT_STABILITY_POSITION_INDICES
                                ),
                                "score_depth": 10,
                            },
                            "results": [{"score_depth": 10}],
                        }
                    }
                )
            )
            manifest = {
                "panels": {
                    "continuation_stability": {
                        "results": str(results_path),
                        "stockfish_score_depth": 30,
                        "stability_probe_version": CURRENT_STABILITY_PROBE_VERSION,
                        "selection_policy": CURRENT_STABILITY_SELECTION_POLICY,
                        "selected_position_indices": list(
                            CURRENT_STABILITY_POSITION_INDICES
                        ),
                    }
                }
            }

            issues = verify_stability_results(manifest)

        self.assertIn("stability/model-a: summary depth mismatch", issues)
        self.assertIn("stability/model-a: result row depth mismatch", issues)

    def test_continuation_scoring_reuses_shared_board_evaluations(self) -> None:
        class CountingEngine:
            def __init__(self) -> None:
                self.calls = 0

            def analyse(self, board, limit):
                self.calls += 1
                return {
                    "score": chess.engine.PovScore(
                        chess.engine.Cp(0), chess.WHITE
                    )
                }

        engine = CountingEngine()
        cache = {}
        model_scores, opponent_scores = score_continuation_moves_from_pgn(
            '[Result "*"]\n\n1. e4 e5 2. Nf3 Nc6 *\n',
            model_side=chess.WHITE,
            pre_move_count=0,
            stockfish=engine,
            depth=30,
            analysis_cache=cache,
        )

        self.assertEqual(len(model_scores), 2)
        self.assertEqual(len(opponent_scores), 2)
        self.assertEqual(engine.calls, 5)

        score_continuation_moves_from_pgn(
            '[Result "*"]\n\n1. e4 e5 2. Nf3 Nc6 *\n',
            model_side=chess.BLACK,
            pre_move_count=0,
            stockfish=engine,
            depth=30,
            analysis_cache=cache,
        )
        self.assertEqual(engine.calls, 5)

    def test_rescore_skips_summary_only_legacy_records_by_default(self) -> None:
        results = {"legacy": {"summary": {"score_depth": 10}}}

        players, rows = rescore_existing_results(
            results,
            player_ids=None,
            stockfish_path="unused",
            score_depth=30,
            workers=1,
        )

        self.assertEqual((players, rows), (0, 0))
        self.assertEqual(results["legacy"]["summary"]["score_depth"], 10)

    def test_rescore_rejects_explicit_record_without_replayable_rows(self) -> None:
        results = {"legacy": {"summary": {"score_depth": 10}}}

        with self.assertRaisesRegex(ValueError, "without stored result rows"):
            rescore_existing_results(
                results,
                player_ids={"legacy"},
                stockfish_path="unused",
                score_depth=30,
                workers=1,
            )

    def test_derives_continuation_retry_outcomes_from_events_and_pgn(self) -> None:
        three_ply_pgn = '[Result "*"]\n\n1. e4 e5 2. Nf3 *\n'
        recovery = {
            "illegal_move_details": [{"move_number": 2}],
            "pgn": three_ply_pgn,
            "termination": "max_moves",
        }
        failed_same_ply = {
            "illegal_move_details": [
                {"move_number": 4},
                {"move_number": 4},
            ],
            "pgn": three_ply_pgn,
            "termination": "forfeit_illegal_move",
        }
        recovered_then_later_second_strike = {
            "illegal_move_details": [
                {"move_number": 2},
                {"move_number": 4},
            ],
            "pgn": three_ply_pgn,
            "termination": "forfeit_illegal_move",
        }
        unknown_api_outcome = {
            "illegal_move_details": [{"move_number": 4}],
            "pgn": three_ply_pgn,
            "termination": "api_error",
        }

        self.assertEqual(
            derive_retry_metrics(recovery),
            {
                "model_retry_attempts": 1,
                "model_retry_recoveries": 1,
                "model_retry_failures": 0,
                "model_retry_unknown": 0,
            },
        )
        self.assertEqual(derive_retry_metrics(failed_same_ply)["model_retry_failures"], 1)
        self.assertEqual(
            derive_retry_metrics(recovered_then_later_second_strike)[
                "model_retry_recoveries"
            ],
            1,
        )
        self.assertEqual(derive_retry_metrics(unknown_api_outcome)["model_retry_unknown"], 1)

    def test_backfills_continuation_rows_and_summary_without_model_calls(self) -> None:
        row = {
            "model_legal_moves": 2,
            "model_illegal_attempts": 1,
            "model_forfeited": False,
            "termination": "max_moves",
            "probe_plies_played": 4,
            "model_move_scores": [],
            "illegal_move_details": [{"move_number": 2}],
            "pgn": '[Result "*"]\n\n1. e4 e5 *\n',
        }
        results = {"model-a": {"summary": {"player_id": "model-a"}, "results": [row]}}

        players, rows = backfill_retry_metrics(results)

        self.assertEqual((players, rows), (1, 1))
        self.assertEqual(row["model_retry_recoveries"], 1)
        self.assertEqual(
            row["conditional_retry_protocol_version"],
            CONDITIONAL_RETRY_PROTOCOL_VERSION,
        )
        retry = results["model-a"]["summary"]["conditional_retry"]
        self.assertEqual(retry["retry_attempts"], 1)
        self.assertEqual(retry["recovery_pct"], 100.0)

    def test_stability_shard_merge_replaces_only_validated_players(self) -> None:
        indices = [0, 12, 24, 36, 1, 13, 25, 37]
        record = {
            "summary": {
                "stability_probe_version": CURRENT_STABILITY_PROBE_VERSION,
                "position_selection_policy": CURRENT_STABILITY_SELECTION_POLICY,
                "selected_position_indices": indices,
                "attempted_positions": 8,
                "model_legal_moves": 32,
                "model_attempts": 32,
                "model_scored_moves": 32,
                "model_forfeits": 0,
                "score_depth": 30,
            },
            "results": [{"position_idx": index} for index in indices],
        }

        merged, players = merge_shards(
            {"existing-model": {"summary": {}}},
            [{"new-model": record}],
            expected_indices=indices,
        )

        self.assertEqual(players, ["new-model"])
        self.assertIn("existing-model", merged)
        self.assertEqual(merged["new-model"], record)

    def test_stability_row_repair_preserves_canonical_contract(self) -> None:
        indices = [0, 12, 24, 36, 1, 13, 25, 37]

        def row(index: int, *, failed: bool = False) -> dict:
            scored_moves = [] if failed else [{"cpl": 10} for _ in range(4)]
            return {
                "position_idx": index,
                "position_id": f"position-{index}",
                "termination": "api_error" if failed else "max_moves",
                "probe_plies_played": 4 if failed else 8,
                "model_legal_moves": 2 if failed else 4,
                "model_illegal_attempts": 0,
                "model_forfeited": False,
                "model_move_scores": scored_moves,
                "opponent_move_scores": scored_moves,
                "illegal_move_details": [],
                "conditional_retry_protocol_version": CONDITIONAL_RETRY_PROTOCOL_VERSION,
            }

        canonical_rows = [row(index, failed=index == 25) for index in indices]
        canonical = {"summary": summarize_player(canonical_rows), "results": canonical_rows}
        canonical["summary"].update(
            {
                "player_id": "model-a",
                "positions_file": "game_like_48.json",
                "probe_plies": 8,
                "score_depth": 30,
                "stability_probe_version": CURRENT_STABILITY_PROBE_VERSION,
                "position_selection_policy": CURRENT_STABILITY_SELECTION_POLICY,
                "selected_position_indices": indices,
            }
        )
        repair_row = row(25)
        repair = {
            "summary": {
                "player_id": "model-a",
                "position_selection_policy": "explicit-indices",
            },
            "results": [repair_row],
        }

        merged = merge_repair_rows("model-a", canonical, repair)

        self.assertTrue(stability_probe_readiness(merged).is_ready)
        self.assertEqual(merged["summary"]["selected_position_indices"], indices)
        self.assertEqual(merged["summary"]["api_errors"], 0)
        self.assertEqual(merged["results"][6], repair_row)

    def test_sequence_shard_merge_accepts_frozen_research_contract(self) -> None:
        indices = [0, 12, 24, 36]
        record = {
            "summary": {
                "stability_probe_version": PROTOCOL_SEQUENCE_VERSION,
                "position_selection_policy": PROTOCOL_SEQUENCE_SELECTION_POLICY,
                "selected_position_indices": indices,
                "attempted_positions": 4,
                "api_errors": 0,
            },
            "results": [{"position_idx": index} for index in indices],
        }

        merged, players = merge_shards(
            {},
            [{"sequence-model": record}],
            expected_indices=indices,
            expected_version=PROTOCOL_SEQUENCE_VERSION,
            expected_policy=PROTOCOL_SEQUENCE_SELECTION_POLICY,
        )

        self.assertEqual(players, ["sequence-model"])
        self.assertEqual(merged["sequence-model"], record)

    def test_default_selection_is_stratified_across_game_like_categories(self) -> None:
        positions = json.loads(GAME_LIKE_POSITIONS_PATH.read_text())["positions"]

        selected = selected_positions(positions, position_indices=None, limit=8)

        self.assertEqual([index for index, _ in selected], [0, 12, 24, 36, 1, 13, 25, 37])
        buckets = [position["regan_bucket"] for _, position in selected]
        self.assertEqual(
            {bucket: buckets.count(bucket) for bucket in set(buckets)},
            {
                "advantage_conversion": 2,
                "defense": 2,
                "quiet_equal": 2,
                "tactical_equal": 2,
            },
        )

    def test_sequence_selection_uses_one_start_from_each_category(self) -> None:
        positions = json.loads(GAME_LIKE_POSITIONS_PATH.read_text())["positions"]

        selected = selected_positions(
            positions,
            position_indices=None,
            limit=PROTOCOL_SEQUENCE_POSITION_LIMIT,
        )

        self.assertEqual([index for index, _ in selected], [0, 12, 24, 36])
        self.assertEqual(
            [position["regan_bucket"] for _, position in selected],
            [
                "advantage_conversion",
                "defense",
                "quiet_equal",
                "tactical_equal",
            ],
        )

    def test_explicit_selection_preserves_requested_panel_order(self) -> None:
        positions = json.loads(GAME_LIKE_POSITIONS_PATH.read_text())["positions"]

        selected = selected_positions(positions, position_indices=[37, 2, 25], limit=2)

        self.assertEqual([index for index, _ in selected], [2, 25])

    def test_outdated_position_selection_is_not_prediction_ready(self) -> None:
        readiness = stability_probe_readiness(
            {
                "summary": {
                    "attempted_positions": 8,
                    "model_scored_moves": 32,
                    "model_attempts": 32,
                    "score_depth": 10,
                }
            }
        )

        self.assertFalse(readiness.is_ready)
        self.assertEqual(readiness.reason, "outdated stability position selection")

    def test_depth_10_stability_record_is_not_prediction_ready(self) -> None:
        readiness = stability_probe_readiness(
            {
                "summary": {
                    "stability_probe_version": CURRENT_STABILITY_PROBE_VERSION,
                    "position_selection_policy": CURRENT_STABILITY_SELECTION_POLICY,
                    "selected_position_indices": list(CURRENT_STABILITY_POSITION_INDICES),
                    "attempted_positions": 8,
                    "model_scored_moves": 32,
                    "model_attempts": 32,
                    "api_errors": 0,
                    "score_depth": 10,
                }
            }
        )

        self.assertFalse(readiness.is_ready)
        self.assertEqual(readiness.reason, "stability score depth < 30")

    def test_combined_legality_uses_worst_fresh_panel_and_reports_pool(self) -> None:
        positions = [
            {"type": "equal", "fen": f"fen-{index}", "best_move": "a1b1"}
            for index in range(10)
        ]
        primary_rows = [
            {"position_idx": index, "fen": position["fen"], "is_legal": index < 9, "cpl": 0}
            for index, position in enumerate(positions)
        ]
        game_rows = [
            {"position_idx": index, "fen": position["fen"], "is_legal": True, "cpl": 0}
            for index, position in enumerate(positions)
        ]
        stability = collect_stability_probe_metrics(
            {
                "summary": {
                    "attempted_positions": 8,
                    "model_scored_moves": 24,
                    "model_legal_moves": 18,
                    "model_attempts": 20,
                    "model_legal_pct": 90.0,
                    "model_forfeit_pct": 0.0,
                    "model_1000cp_catastrophe_pct": 0.0,
                }
            }
        )
        primary = collect_equal_position_metrics(primary_rows, positions)
        game_like = collect_equal_position_metrics(game_rows, positions)

        self.assertIsNotNone(primary)
        self.assertIsNotNone(game_like)
        self.assertIsNotNone(stability)
        combined = combine_legality_metrics(primary, game_like, stability)

        self.assertEqual(combined.panel_legal_pcts, (90.0, 100.0, 90.0))
        self.assertEqual(combined.conservative_legal_pct, 90.0)
        self.assertAlmostEqual(combined.pooled_legal_pct, 37 / 40 * 100)

    def test_pre_moves_reconstruct_nonopening_position(self) -> None:
        with GAME_LIKE_POSITIONS_PATH.open() as f:
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
                    {"cpl": 25, "model_turn_index": 1},
                    {"cpl": 350, "model_turn_index": 2},
                ],
                "opponent_move_scores": [{"cpl": 75}],
            },
            {
                "model_legal_moves": 1,
                "model_illegal_attempts": 2,
                "model_forfeited": True,
                "termination": "forfeit_illegal_move",
                "probe_plies_played": 3,
                "model_move_scores": [
                    {"cpl": 1200, "model_turn_index": 1},
                ],
                "opponent_move_scores": [{"cpl": 225}],
            },
        ]

        summary = summarize_player(rows)

        self.assertEqual(summary["attempted_positions"], 2)
        self.assertEqual(summary["model_legal_moves"], 5)
        self.assertEqual(summary["model_illegal_attempts"], 2)
        self.assertAlmostEqual(summary["model_legal_pct"], 100 * 5 / 7)
        self.assertIsNone(summary["model_first_attempt_turns"])
        self.assertIsNone(summary["model_first_attempt_illegals"])
        self.assertIsNone(summary["model_first_attempt_illegal_pct"])
        self.assertEqual(summary["model_forfeit_pct"], 50.0)
        self.assertEqual(summary["model_scored_moves"], 3)
        self.assertAlmostEqual(summary["model_avg_cpl"], 525.0)
        self.assertAlmostEqual(summary["model_first_move_avg_cpl"], 612.5)
        self.assertEqual(summary["model_later_move_avg_cpl"], 350.0)
        self.assertEqual(summary["opponent_scored_moves"], 2)
        self.assertEqual(summary["opponent_avg_cpl"], 150.0)
        self.assertEqual(summary["model_300cp_blunders"], 2)
        self.assertAlmostEqual(summary["model_300cp_blunder_pct"], 100 * 2 / 3)
        self.assertEqual(summary["model_300cp_blunder_positions"], 2)
        self.assertEqual(summary["model_300cp_blunder_position_pct"], 100.0)
        self.assertEqual(summary["model_1000cp_catastrophe_positions"], 1)
        self.assertEqual(summary["model_1000cp_catastrophe_position_pct"], 50.0)
        self.assertAlmostEqual(
            summary["model_1000cp_deduplicated_catastrophe_pct"], 100 / 3
        )
        self.assertEqual(summary["conditional_retry"]["incomplete_evidence_games"], 1)
        self.assertNotIn("conditional_retry_protocol_version", summary)


class OverlayReevaluationTests(unittest.TestCase):
    def test_row_token_totals_are_canonical(self) -> None:
        rows = [
            {"prompt_tokens": 10, "completion_tokens": 2},
            {"prompt_tokens": 20, "completion_tokens": 3},
            {},
        ]

        self.assertEqual(
            sum_result_row_tokens(rows),
            {"prompt": 30, "completion": 5},
        )

    def test_filter_results_limits_players_and_positions(self) -> None:
        results = {
            "model-a": {
                "summary": {"prompt_history_replay": True},
                "results": [
                    {"position_idx": 1, "prompt_tokens": 10, "completion_tokens": 1},
                    {"position_idx": 2, "prompt_tokens": 20, "completion_tokens": 2},
                ],
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
        self.assertEqual(
            filtered["model-a"]["results"],
            [{"position_idx": 2, "prompt_tokens": 20, "completion_tokens": 2}],
        )
        self.assertEqual(filtered["model-a"]["summary"]["prompt_history_replay"], True)
        self.assertEqual(filtered["model-a"]["token_usage"], {"prompt": 20, "completion": 2})

    def test_overlay_replacement_does_not_double_token_usage(self) -> None:
        base = {
            "model-a": {
                "summary": {},
                "results": [
                    {"position_idx": 0, "prompt_tokens": 10, "completion_tokens": 1},
                ],
                "token_usage": {"prompt": 10, "completion": 1},
            }
        }
        overlay = {
            "model-a": {
                "summary": {},
                "results": [
                    {"position_idx": 0, "prompt_tokens": 10, "completion_tokens": 1},
                ],
                "token_usage": {"prompt": 10, "completion": 1},
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            overlay_path = Path(tmpdir) / "overlay.json"
            overlay_path.write_text(json.dumps(overlay))
            merged, _ = merge_overlays(base, [overlay_path])

        self.assertEqual(merged["model-a"]["token_usage"], {"prompt": 10, "completion": 1})

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

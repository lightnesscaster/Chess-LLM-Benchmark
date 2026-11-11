"""Pure, frozen-policy validation of corrected-CPL rating calibration.

No remote data, rating writes, model calls, or production coefficient mutation.
Targets must come from the independent, seed-disabled saved-game replay.
"""
from __future__ import annotations

from collections import Counter
import math
from typing import Any, TYPE_CHECKING

import numpy as np
from scipy.optimize import lsq_linear

from position_benchmark.predictions import (
    DEFAULT_GAME_LIKE_CPL_CAP, benchmark_result_readiness,
    collect_equal_position_metrics, collect_stability_probe_metrics,
    combine_legality_metrics, combine_prediction_with_downside_cap,
    predict_rating_from_model_data_with_supplement, stability_probe_prediction_cap,
    stability_probe_readiness, survival_probability,
)

from position_benchmark.scoring_provenance import validate_scoring_record

if TYPE_CHECKING:
    from rating.rating_store import PositionBenchmarkInputs

LEGACY_COEFFICIENTS = (1298.57, -200.43, 15.39, 5.85)
# Existing supplemental-analysis groups, plus explicitly frozen new model lines.
FAMILY_PREFIXES = (
    "gpt-6.1-sol", "gpt-6-astra", "gemini-4-argon", "gemini-4-carbon",
    "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "gemini-3.5",
    "gemini-3.1", "gemini-3", "gemini-2.5", "gemini-2.0", "gpt-5.5",
    "gpt-5.4", "gpt-5.2", "gpt-5.1", "gpt-5-mini", "gpt-5", "gpt-oss",
    "gpt-3.5", "grok-4.3", "grok-4.1", "grok-4", "grok-3", "claude",
    "deepseek-v4", "deepseek-v3.2", "deepseek-v3.1", "deepseek-chat",
    "deepseek-r1", "kimi-k2.5", "kimi-k2", "llama-4", "llama-3.3",
    "mistral", "qwen3", "glm", "gemma", "mimo", "o1", "o3",
)


def model_family(player_id: str) -> str:
    """Keep all efforts of a model line in one fold, using longest prefixes."""
    lowered = player_id.lower().split("/")[-1]
    for prefix in sorted(FAMILY_PREFIXES, key=len, reverse=True):
        if lowered.startswith(prefix):
            return prefix
    return lowered.split()[0].split("-")[0]


def model_lab(player_id: str) -> str:
    """Normalize model providers for the independent provider stress check."""
    lowered = player_id.lower().split("/")[-1]
    for prefixes, lab in (
        (("gpt", "o1", "o3", "o4"), "OpenAI"),
        (("gemini", "gemma"), "Google"), (("claude",), "Anthropic"),
        (("deepseek",), "DeepSeek"), (("grok",), "xAI"),
        (("kimi",), "Moonshot"), (("llama",), "Meta"),
        (("mistral", "mixtral"), "Mistral"), (("qwen",), "Alibaba"),
        (("glm",), "Zhipu"), (("mimo",), "Xiaomi"),
    ):
        if lowered.startswith(prefixes):
            return lab
    return lowered.split()[0].split("-")[0]


def _features(metrics: Any, legal_pct: float | None = None) -> list[float]:
    return [math.log1p(max(0, metrics.equal_cpl)), metrics.best_pct,
            survival_probability(metrics.legal_pct if legal_pct is None else legal_pct)]


def predict_core(row: dict, coefficients: list | tuple) -> float:
    """Evaluate coefficients with the unchanged production panel order/deadbands."""
    def value(features: list) -> float:
        return float(coefficients[0] + np.dot(coefficients[1:], features))
    predicted = value(row["core_features"])
    for panel in ("blunder_features", "game_like_features"):
        if row.get(panel) is not None:
            predicted = combine_prediction_with_downside_cap(predicted, value(row[panel]))
    return combine_prediction_with_downside_cap(predicted, row.get("ready_stability_cap"))


def build_rows(inputs: PositionBenchmarkInputs, targets: dict) -> list[dict]:
    """Build a reproducible feature matrix and assert full production parity."""
    rows = []
    for player_id, core in sorted(inputs.core_results.items()):
        if player_id not in targets or not benchmark_result_readiness(core, inputs.core_positions).is_ready:
            continue
        validate_scoring_record(core, required_version="root-reference-v1")
        target = targets[player_id]
        supplements = core.get("supplements", {})
        def panel(name: str, records: dict | None) -> dict | None:
            return supplements.get(name) or (records or {}).get(player_id)
        blunder = panel("blunder", inputs.blunder_results)
        game = panel("game_like", inputs.game_like_results)
        stability = panel("stability", inputs.stability_results)
        production = predict_rating_from_model_data_with_supplement(
            core, inputs.core_positions, blunder_model_data=blunder,
            blunder_positions=inputs.blunder_positions, game_like_model_data=game,
            game_like_positions=inputs.game_like_positions, stability_probe_model_data=stability,
        )
        metrics = collect_equal_position_metrics(core.get("results", []), inputs.core_positions)
        if metrics is None or production is None:
            continue
        ready_stability = stability is not None and stability_probe_readiness(stability).is_ready
        if ready_stability:
            validate_scoring_record(stability, required_version="root-reference-v1")
        stability_metrics = collect_stability_probe_metrics(stability) if ready_stability else None
        row = dict(player_id=player_id, family=model_family(player_id), lab=model_lab(player_id),
                   games=int(target.get("games", target.get("games_played", 0))),
                   game_only_rd=float(target.get("game_only_rd", target.get("rating_deviation", 350))),
                   games_rd=target.get("games_rd"),
                   target=float(target.get("target", target.get("rating"))),
                   core_features=_features(metrics), blunder_features=None, game_like_features=None,
                   ready_stability_cap=stability_probe_prediction_cap(stability) if ready_stability else None,
                   baseline_prediction=float(production))
        selected_blunder, blunder_positions = blunder, inputs.blunder_positions
        if selected_blunder is None and any(p.get("type") == "blunder" for p in inputs.core_positions):
            selected_blunder, blunder_positions = core, inputs.core_positions
        if selected_blunder is not None and blunder_positions is not None and benchmark_result_readiness(
            selected_blunder, blunder_positions, min_equal_positions=25, position_type="blunder"
        ).is_ready:
            validate_scoring_record(selected_blunder, required_version="root-reference-v1")
            bm = collect_equal_position_metrics(selected_blunder.get("results", []), blunder_positions,
                                               cpl_cap=DEFAULT_GAME_LIKE_CPL_CAP, position_type="blunder")
            if bm is not None:
                row["blunder_features"] = _features(bm)
        if game is not None and inputs.game_like_positions is not None and benchmark_result_readiness(
            game, inputs.game_like_positions, min_equal_positions=48
        ).is_ready:
            validate_scoring_record(game, required_version="root-reference-v1")
            gm = collect_equal_position_metrics(game.get("results", []), inputs.game_like_positions,
                                               cpl_cap=DEFAULT_GAME_LIKE_CPL_CAP)
            if gm is not None:
                legality = combine_legality_metrics(metrics, gm, stability_metrics)
                row["game_like_features"] = _features(gm, legality.conservative_legal_pct)
        if not math.isclose(predict_core(row, LEGACY_COEFFICIENTS), production, abs_tol=1e-8):
            raise AssertionError(f"Production coefficient/cap parity failed for {player_id}")
        rows.append(row)
    return rows


def _weights(rows: list[dict]) -> np.ndarray:
    counts = Counter(row["family"] for row in rows)
    weights = np.array([1 / counts[row["family"]] for row in rows])
    return weights * len(rows) / weights.sum()


def fit_candidate(rows: list[dict], candidate: str) -> dict:
    """Fit using only supplied training rows and family-balanced weights."""
    if not rows:
        raise ValueError("Cannot fit an empty training fold")
    if candidate == "fixed":
        return {"kind": "fixed", "candidate": candidate}
    weights = _weights(rows)
    y = np.array([r["target"] for r in rows], dtype=float)
    baseline = np.array([r["baseline_prediction"] for r in rows], dtype=float)
    if candidate == "offset":
        return {"kind": "offset", "candidate": candidate,
                "offset": float(np.average(y - baseline, weights=weights))}
    if candidate == "affine":
        center = float(np.average(baseline, weights=weights))
        x = np.column_stack((np.ones(len(rows)), baseline - center))
        result = lsq_linear(x * np.sqrt(weights[:, None]), y * np.sqrt(weights),
                            bounds=([-np.inf, 0], [np.inf, np.inf]), tol=1e-12, method="bvls")
        if not result.success:
            raise ValueError(f"Affine fit failed: {result.message}")
        return {"kind": "affine", "candidate": candidate,
                "intercept": float(result.x[0] - result.x[1] * center), "slope": float(result.x[1])}
    if not candidate.startswith("core-ridge-"):
        raise ValueError(f"Unknown candidate: {candidate}")
    alpha = float(candidate.removeprefix("core-ridge-"))
    features = np.array([r["core_features"] for r in rows], dtype=float)
    mean = np.average(features, axis=0, weights=weights)
    scale = np.sqrt(np.average((features - mean) ** 2, axis=0, weights=weights))
    scale[scale < 1e-12] = 1
    x = np.column_stack((np.ones(len(rows)), (features - mean) / scale))
    prior = np.array(LEGACY_COEFFICIENTS[1:]) * scale
    augmented_x = np.vstack((x * np.sqrt(weights[:, None]),
                              np.column_stack((np.zeros(3), np.sqrt(alpha) * np.eye(3)))))
    augmented_y = np.concatenate((y * np.sqrt(weights), np.sqrt(alpha) * prior))
    result = lsq_linear(augmented_x, augmented_y,
                        bounds=([-np.inf, -np.inf, 0, 0], [np.inf, 0, np.inf, np.inf]), tol=1e-12, method="bvls")
    if not result.success:
        raise ValueError(f"Core fit failed: {result.message}")
    slopes = result.x[1:] / scale
    coefficients = [float(result.x[0] - np.dot(slopes, mean)), *slopes.tolist()]
    return {"kind": "core", "candidate": candidate, "alpha": alpha, "coefficients": coefficients,
            "normalization": {"mean": mean.tolist(), "scale": scale.tolist()},
            "weight_sum": float(weights.sum())}


def predict_candidate(rows: list[dict], model: dict) -> list[float]:
    """Apply an offset/affine once to the full prediction, or refit panel means."""
    if model["kind"] == "core":
        return [predict_core(row, model["coefficients"]) for row in rows]
    return [float(model.get("intercept", model.get("offset", 0)) +
                  model.get("slope", 1) * row["baseline_prediction"]) for row in rows]


def metrics(rows: list[dict], predictions: list[float]) -> dict:
    """Row and equal-family prediction errors; positive bias means underprediction."""
    if not rows:
        return {key: None for key in ("row_mae", "row_rmse", "macro_mae", "macro_rmse", "bias", "max_error")}
    errors = np.array([r["target"] - p for r, p in zip(rows, predictions, strict=True)])
    families = sorted({r["family"] for r in rows})
    grouped = [errors[[r["family"] == f for r in rows]] for f in families]
    return dict(row_mae=float(np.mean(np.abs(errors))), row_rmse=float(np.sqrt(np.mean(errors ** 2))),
                macro_mae=float(np.mean([np.mean(np.abs(e)) for e in grouped])),
                macro_rmse=float(np.sqrt(np.mean([np.mean(e ** 2) for e in grouped]))),
                bias=float(np.mean(errors)), max_error=float(np.max(np.abs(errors))))


def select_candidate(rows: list[dict], policy: dict) -> tuple[str, list[dict]]:
    """Inner leave-family-out selection; never inspect an outer held-out target."""
    groups = sorted({r["family"] for r in rows})
    if len(groups) < 2:
        return "fixed", []
    scores = []
    for candidate in policy["candidates"]:
        predictions = [None] * len(rows)
        for group in groups:
            train = [r for r in rows if r["family"] != group]
            indices = [i for i, r in enumerate(rows) if r["family"] == group]
            model = fit_candidate(train, candidate)
            values = predict_candidate([rows[i] for i in indices], model)
            for i, value in zip(indices, values):
                predictions[i] = value
        scores.append({"candidate": candidate, **metrics(rows, predictions)})
    tolerance = policy.get("numeric_tie_tolerance", 1e-9)
    best_mae = min(s["macro_mae"] for s in scores)
    eligible = [s for s in scores if s["macro_mae"] <= best_mae + tolerance]
    best_rmse = min(s["macro_rmse"] for s in eligible)
    eligible = [s for s in eligible if s["macro_rmse"] <= best_rmse + tolerance]
    order = policy["simplicity_order"]
    selected = min(eligible, key=lambda s: order.index(s["candidate"]))["candidate"]
    return selected, scores


def _outer(rows: list[dict], policy: dict, field: str) -> tuple[list[float], list[dict]]:
    predictions = [None] * len(rows)
    folds = []
    groups = sorted({r[field] for r in rows})
    if len(groups) < 2:
        return [], []
    for group in groups:
        train = [r for r in rows if r[field] != group]
        indices = [i for i, r in enumerate(rows) if r[field] == group]
        test = [rows[i] for i in indices]
        candidate, scores = select_candidate(train, policy)
        model = fit_candidate(train, candidate)
        for i, value in zip(indices, predict_candidate(test, model)):
            predictions[i] = value
        folds.append(dict(held_out_group=group, train_groups=sorted({r[field] for r in train}),
                          test_groups=[group], train_families=sorted({r["family"] for r in train}),
                          test_families=sorted({r["family"] for r in test}),
                          train_player_ids=[r["player_id"] for r in train],
                          test_player_ids=[r["player_id"] for r in test],
                          selected_candidate=candidate, selected_model=model, inner_scores=scores))
    return predictions, folds


def _bootstrap(rows: list[dict], fixed: list[float], adaptive: list[float], policy: dict) -> dict:
    """Paired family resampling, retaining all configurations in sampled families."""
    families = sorted({r["family"] for r in rows})
    if not families:
        return {"mae_improvement_probability": None, "fixed": {}, "adaptive": {}}
    count = np.array([sum(r["family"] == f for r in rows) for f in families])
    rng = np.random.default_rng(policy["bootstrap"]["seed"])
    draws = rng.integers(0, len(families), (policy["bootstrap"]["resamples"], len(families)))
    result = {}
    family_maes = {}
    for method, values in (("fixed", fixed), ("adaptive", adaptive)):
        errors = np.array([r["target"] - p for r, p in zip(rows, values, strict=True)])
        family_mae = np.array([np.mean(np.abs(errors[[r["family"] == f for r in rows]])) for f in families])
        family_mse = np.array([np.mean(errors[[r["family"] == f for r in rows]] ** 2) for f in families])
        macro_rmse = np.sqrt(np.mean(family_mse[draws], axis=1))
        row_rmse = np.sqrt(np.sum((family_mse * count)[draws], axis=1) / np.sum(count[draws], axis=1))
        maximum = np.maximum(row_rmse, macro_rmse)
        family_maes[method] = np.mean(family_mae[draws], axis=1)
        result[method] = dict(
            row_rmse_ci95=np.percentile(row_rmse, [2.5, 97.5]).tolist(),
            macro_rmse_ci95=np.percentile(macro_rmse, [2.5, 97.5]).tolist(),
            conservative_rmse_ci95=np.percentile(maximum, [2.5, 97.5]).tolist(),
            conservative_rmse_lower=float(np.percentile(maximum, policy["uncertainty_gate"]["bootstrap_rmse_lower_percentile"])),
        )
    result["mae_improvement_probability"] = float(np.mean(family_maes["adaptive"] < family_maes["fixed"]))
    result["paired_macro_mae_improvement_ci95"] = np.percentile(
        family_maes["fixed"] - family_maes["adaptive"], [2.5, 97.5]).tolist()
    result.update(resamples=policy["bootstrap"]["resamples"], seed=policy["bootstrap"]["seed"], unit="model-line-family")
    return result


def _cohort(row: dict, rules: dict) -> bool:
    return row["games"] >= rules["minimum_games"] and row["game_only_rd"] <= rules["maximum_game_only_rd"]


def _ratio(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    if baseline == 0:
        return 1.0 if value == 0 else None
    return value / baseline


def analyze(rows: list[dict], policy: dict) -> dict:
    """Evaluate frozen nested procedures and return explicit, componentwise gates."""
    excluded = policy["primary_cohort"]["exclude_family"]
    if len({r["player_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate player IDs in calibration matrix")
    for row in rows:
        if not math.isclose(predict_core(row, LEGACY_COEFFICIENTS), row["baseline_prediction"], abs_tol=1e-8):
            raise ValueError(f"Legacy coefficient/panel parity failed: {row['player_id']}")
        if not all(math.isfinite(float(row[k])) for k in ("target", "game_only_rd", "baseline_prediction")):
            raise ValueError(f"Non-finite calibration row: {row['player_id']}")
    primary = [r for r in rows if r["family"] != excluded and _cohort(r, policy["primary_cohort"])]
    families = sorted({r["family"] for r in primary})
    if len(families) < 3:
        raise ValueError("Nested validation requires at least three eligible families")
    fixed = [r["baseline_prediction"] for r in primary]
    adaptive, folds = _outer(primary, policy, "family")
    stress, provider_folds = _outer(primary, policy, "lab")
    fixed_metrics, adaptive_metrics = metrics(primary, fixed), metrics(primary, adaptive)
    stress_metrics = metrics(primary, stress) if stress else metrics([], [])
    bootstrap = _bootstrap(primary, fixed, adaptive, policy)
    family_results = []
    for family in families:
        indices = [i for i, r in enumerate(primary) if r["family"] == family]
        subset = [primary[i] for i in indices]
        fm = metrics(subset, [fixed[i] for i in indices])
        am = metrics(subset, [adaptive[i] for i in indices])
        family_results.append(dict(family=family, configurations=len(indices), fixed=fm, adaptive=am,
                                   mae_improvement=fm["row_mae"] - am["row_mae"]))
    sensitivities = {}
    for name in ("reliable", "mature"):
        indices = [i for i, r in enumerate(primary) if _cohort(r, policy[f"{name}_cohort"])]
        subset = [primary[i] for i in indices]
        fm, am = metrics(subset, [fixed[i] for i in indices]), metrics(subset, [adaptive[i] for i in indices])
        sensitivities[name] = dict(configurations=len(subset), families=len({r["family"] for r in subset}),
                                   fixed=fm, adaptive=am,
                                   ratios={key: _ratio(am[key], fm[key]) for key in
                                           ("row_mae", "row_rmse", "macro_mae", "macro_rmse")})
    gate = policy["mean_promotion_gate"]
    mae_fraction = 1 - _ratio(adaptive_metrics["macro_mae"], fixed_metrics["macro_mae"]) if fixed_metrics["macro_mae"] else 0
    rmse_fraction = 1 - _ratio(adaptive_metrics["macro_rmse"], fixed_metrics["macro_rmse"]) if fixed_metrics["macro_rmse"] else 0
    checks = dict(
        minimum_configurations=len(primary) >= gate["minimum_configurations"],
        minimum_families=len(families) >= gate["minimum_families"],
        macro_mae_improvement=mae_fraction >= gate["minimum_macro_mae_improvement_fraction"],
        macro_rmse_improvement=rmse_fraction >= gate["minimum_macro_rmse_improvement_fraction"],
        bootstrap_probability=bootstrap["mae_improvement_probability"] >= gate["minimum_bootstrap_mae_improvement_probability"],
        family_win_fraction=sum(f["mae_improvement"] > 0 for f in family_results) / len(families) >= gate["minimum_family_win_fraction"],
        maximum_family_worsening=min(f["mae_improvement"] for f in family_results) >= -gate["maximum_single_family_mae_worsening"],
    )
    for name, sensitivity in sensitivities.items():
        checks[f"{name}_subset"] = all(value is not None and value <= gate[f"maximum_{name}_subset_metric_ratio"]
                                           for value in sensitivity["ratios"].values())
    stress_ratios = {key: _ratio(stress_metrics[key], fixed_metrics[key]) for key in ("row_mae", "row_rmse", "macro_mae", "macro_rmse")}
    checks["provider_stress"] = all(v is not None and v <= gate["maximum_provider_heldout_metric_ratio"] for v in stress_ratios.values())
    promote = all(checks.values())
    # Only now select/refit on the full eligible development cohort.
    final_candidate, final_scores = select_candidate(primary, policy)
    final_model = fit_candidate(primary, final_candidate)
    method = "adaptive" if promote else "fixed"
    uncertainty_metrics = adaptive_metrics if promote else fixed_metrics
    uncertainty_policy = policy["uncertainty_gate"]
    uncertainty_checks = dict(
        minimum_configurations=len(primary) >= uncertainty_policy["minimum_configurations"],
        minimum_families=len(families) >= uncertainty_policy["minimum_families"],
        lower_bound_exceeds_existing_rd=bootstrap[method]["conservative_rmse_lower"] > uncertainty_policy["existing_rd"],
    )
    widen = all(uncertainty_checks.values())
    estimated_rd = min(uncertainty_policy["maximum_rd"], max(uncertainty_policy["existing_rd"],
                       math.ceil(max(uncertainty_metrics["row_rmse"], uncertainty_metrics["macro_rmse"]) / 25) * 25))
    recommended_rd = estimated_rd if widen else uncertainty_policy["existing_rd"]
    return dict(experiment_id=policy["experiment_id"], policy=policy,
                cohort=dict(configurations=len(primary), families=len(families), family_names=families,
                            providers=sorted({r["lab"] for r in primary}), player_ids=[r["player_id"] for r in primary]),
                excluded_rows=[dict(player_id=r["player_id"], family=r["family"], games=r["games"], game_only_rd=r["game_only_rd"])
                               for r in rows if r not in primary],
                fixed_metrics=fixed_metrics, adaptive_metrics=adaptive_metrics,
                provider_stress=dict(metrics=stress_metrics, ratios=stress_ratios),
                family_results=family_results, sensitivity_subsets=sensitivities,
                family_folds=folds, provider_folds=provider_folds, bootstrap=bootstrap,
                oof_predictions=[dict(player_id=r["player_id"], family=r["family"], lab=r["lab"], target=r["target"],
                                      fixed=fixed[i], adaptive=adaptive[i], provider_stress=stress[i] if stress else None)
                                 for i, r in enumerate(primary)],
                mean_gate=dict(passed=promote, checks=checks, macro_mae_improvement_fraction=mae_fraction,
                               macro_rmse_improvement_fraction=rmse_fraction),
                final_selection=dict(candidate=final_candidate, inner_scores=final_scores, model=final_model),
                carbon_diagnostics=[dict(player_id=r["player_id"], target=r["target"], baseline_prediction=r["baseline_prediction"],
                                         proposed_prediction=predict_candidate([r], final_model)[0]) for r in rows if r["family"] == excluded],
                uncertainty=dict(method=method, metrics=uncertainty_metrics, checks=uncertainty_checks,
                                 passed=widen, existing_rd=uncertainty_policy["existing_rd"],
                                 bootstrap_rmse_lower=bootstrap[method]["conservative_rmse_lower"],
                                 estimated_rd=estimated_rd, recommended_rd=recommended_rd),
                recommendation=dict(mean_action="promote" if promote else "retain",
                                    production_model=final_model if promote else {"kind": "fixed", "candidate": "fixed"},
                                    rd_action="widen" if recommended_rd > uncertainty_policy["existing_rd"] else "retain",
                                    seed_rd=recommended_rd, forced_refit=False))


def render_report(result: dict) -> str:
    """Render a concise, auditable Markdown validation report."""
    cohort, mean, uncertainty = result["cohort"], result["mean_gate"], result["uncertainty"]
    lines = [f"# {result['experiment_id']}", "", f"Eligible configurations: {cohort['configurations']}; families: {cohort['families']}.",
             "Carbon is diagnostic only. Targets use the benchmark-seed-free game replay.", "",
             "| Method | Row MAE | Row RMSE | Macro MAE | Macro RMSE | Bias | Max error |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for label, values in (("Fixed production", result["fixed_metrics"]), ("Nested selected", result["adaptive_metrics"]),
                          ("Provider held out", result["provider_stress"]["metrics"])):
        cells = ["unavailable" if values[k] is None else f"{values[k]:.3f}" for k in
                 ("row_mae", "row_rmse", "macro_mae", "macro_rmse", "bias", "max_error")]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", f"Mean promotion: {'PASS' if mean['passed'] else 'FAIL'}.",
              f"Paired bootstrap MAE improvement probability: {result['bootstrap']['mae_improvement_probability']:.4f}.", ""]
    lines += [f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in mean["checks"].items()]
    lines += ["", "Sensitivity and provider stress metric ratios (maximum 1.05):"]
    for name, values in result["sensitivity_subsets"].items():
        lines.append(f"- {name} ({values['configurations']} configurations): {json_ratios(values['ratios'])}")
    lines.append(f"- provider stress: {json_ratios(result['provider_stress']['ratios'])}")
    lines += ["", f"Full-cohort candidate after validation: `{result['final_selection']['candidate']}`.",
              f"Production mean action: **{result['recommendation']['mean_action']}**.",
              f"Uncertainty uses **{uncertainty['method']}** outer family predictions; lower bootstrap RMSE: {uncertainty['bootstrap_rmse_lower']:.3f}.",
              f"Seed RD recommendation: **{uncertainty['recommended_rd']}** (previous {uncertainty['existing_rd']}).", "",
              "No forced refit. Only components passing their separate frozen gates may be published.",
              "Complete fold membership, inner selection scores, normalization, coefficients, sensitivity subsets, and hashes are in the JSON artifact.", ""]
    environment = result.get("analysis_environment")
    if environment:
        lines += ["", f"Tested environment: Python {environment['python']}, NumPy {environment['numpy']}, SciPy {environment['scipy']}."]
    for limitation in result["policy"].get("limitations", []):
        lines.append(f"- Limitation: {limitation}")
    return "\n".join(lines)


def json_ratios(ratios: dict) -> str:
    """Format checked ratio evidence without suppressing missing subsets."""
    return "; ".join(f"{key}={value:.4f}" if value is not None else f"{key}=unavailable"
                     for key, value in ratios.items())

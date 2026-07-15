"""Canonical filesystem layout for production position-benchmark artifacts."""

from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RESULT_SCHEMA_VERSION = 2

MANIFEST_PATH = BASE_DIR / "benchmark_manifest.json"

PANELS_DIR = BASE_DIR / "panels"
CORE_POSITIONS_PATH = PANELS_DIR / "core_equal_50.json"
GAME_LIKE_POSITIONS_PATH = PANELS_DIR / "game_like_48.json"
BLUNDER_POSITIONS_PATH = PANELS_DIR / "optional_blunder_25.json"

RESULTS_DIR = BASE_DIR / "results"
CORE_RESULTS_PATH = RESULTS_DIR / "core.json"
GAME_LIKE_RESULTS_PATH = RESULTS_DIR / "game_like.json"
BLUNDER_RESULTS_PATH = RESULTS_DIR / "optional_blunder.json"
STABILITY_RESULTS_PATH = RESULTS_DIR / "stability.json"
LEGALITY_STRESS_RESULTS_PATH = RESULTS_DIR / "legality_stress.json"

CANDIDATES_DIR = BASE_DIR / "candidates"
EQUAL_CANDIDATES_PATH = CANDIDATES_DIR / "equal_100_depth16.json"
BLUNDER_CANDIDATES_PATH = CANDIDATES_DIR / "blunder_1000_depth16.json"
LEGALITY_STRESS_POSITIONS_PATH = CANDIDATES_DIR / "legality_stress_6.json"

LEGACY_DIR = BASE_DIR / "legacy"
LEGACY_COMBINED_POSITIONS_PATH = LEGACY_DIR / "combined_positions_75.json"


def repo_relative(path: Path) -> str:
    """Return a stable repository-relative display path."""
    return str(path.relative_to(BASE_DIR.parent))

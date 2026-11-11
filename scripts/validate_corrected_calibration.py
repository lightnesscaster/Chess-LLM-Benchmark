#!/usr/bin/env python3
"""Run frozen corrected-CPL validation without remote calls or production writes."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform

import numpy as np
import scipy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from position_benchmark.calibration_validation import analyze, build_rows, render_report

DEFAULT_POLICY = ROOT / "position_benchmark/validation/2026-11-11-corrected-cpl-calibration-plan.json"


def path_key(path: Path) -> str:
    """Use checkout-relative provenance keys when a file belongs to this checkout."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def sha256(path: Path) -> str:
    """Hash the exact bytes consumed by this invocation."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(manifest: dict, args: argparse.Namespace) -> dict:
    """Fail closed on frozen raw inputs; distinguish matrix reference-only mode."""
    checks = []
    def verify(path: Path, expected: str | None, label: str) -> None:
        if not isinstance(expected, str) or not expected:
            raise ValueError(f"Manifest is missing required hash: {label}")
        if sha256(path) != expected:
            raise ValueError(f"Manifest hash mismatch: {label} ({path_key(path)})")
        checks.append(dict(file=path_key(path), manifest_key=label, sha256=expected))
    policy_key = path_key(args.policy)
    files = manifest.get("files", {})
    if args.inputs is not None:
        verify(args.inputs, manifest.get("inputs_sha256"), "inputs_sha256")
        verify(args.targets, manifest.get("game_only_target_sha256"), "game_only_target_sha256")
        verify(args.policy, files.get(policy_key), f"files.{policy_key}")
        return dict(mode="verified-inputs-targets-policy", checks=checks)
    matrix_declared = False
    for key in ("matrix_sha256", "rows_sha256"):
        if key in manifest:
            verify(args.rows, manifest[key], key)
            matrix_declared = True
    matrix_key = path_key(args.rows)
    if matrix_key in files:
        verify(args.rows, files[matrix_key], f"files.{matrix_key}")
        matrix_declared = True
    if policy_key in files:
        verify(args.policy, files[policy_key], f"files.{policy_key}")
    return dict(mode="verified-matrix" if matrix_declared else "declared-reference-only", checks=checks,
                raw_inputs_verified=False,
                note="Raw input and game-only target files are unavailable in public matrix reproduction.")


def main() -> None:
    """Validate a saved matrix or explicit frozen input/benchmark-seed-free target files."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--rows", type=Path, help="Saved non-private feature matrix")
    source.add_argument("--inputs", type=Path, help="JSON dataclass fields for PositionBenchmarkInputs")
    parser.add_argument("--targets", type=Path, help="Benchmark-seed-free game-only player-rating map")
    parser.add_argument("--manifest", type=Path, help="Verify declared source hashes before analyzing")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--save-rows", type=Path, help="Save the derived reproducible feature matrix")
    args = parser.parse_args()
    if args.inputs is not None and args.targets is None:
        parser.error("--inputs requires --targets")
    if args.rows is not None and args.targets is not None:
        parser.error("--targets only applies with --inputs")
    def read(path: Path):
        return json.loads(path.read_text())
    policy = read(args.policy)
    manifest_verification = None
    if args.manifest is not None:
        try:
            manifest_verification = verify_manifest(read(args.manifest), args)
        except ValueError as error:
            parser.error(str(error))
    if args.rows is not None:
        matrix = read(args.rows)
        rows = matrix["rows"] if isinstance(matrix, dict) else matrix
        source_paths = [args.rows]
    else:
        from rating.rating_store import PositionBenchmarkInputs
        rows = build_rows(PositionBenchmarkInputs(**read(args.inputs)), read(args.targets))
        source_paths = [args.inputs, args.targets]
    if args.save_rows is not None:
        args.save_rows.parent.mkdir(parents=True, exist_ok=True)
        args.save_rows.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    if args.manifest is not None:
        source_paths.append(args.manifest)
    result = analyze(rows, policy)
    result["analysis_environment"] = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    if manifest_verification is not None:
        result["manifest_verification"] = manifest_verification
    result["source_hashes"] = {path_key(p): sha256(p)
                               for p in source_paths + [args.policy]}
    result["implementation_hashes"] = {
        path_key(p): sha256(p)
        for p in (Path(__file__), ROOT / "position_benchmark/calibration_validation.py",
                  ROOT / "position_benchmark/predictions.py")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.report.write_text(render_report(result))
    print(json.dumps({"output": str(args.output), "report": str(args.report),
                      "recommendation": result["recommendation"]}, indent=2))


if __name__ == "__main__":
    main()

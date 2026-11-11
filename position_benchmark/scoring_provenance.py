"""Keep corrected root evaluations from mixing with historical CPL scores."""

from __future__ import annotations

from typing import Any
import copy
import hashlib
import json


def compact_record_profiles(record: dict[str, Any]) -> dict[str, Any]:
    """Compact an earlier running scorer's output without repeating analysis.

    Retain the full engine options once in the summary for audit. This keeps a
    complete benchmark correction small enough for atomic Firestore publication.
    """
    from position_benchmark.evaluation import SCORING_VERSION, scoring_profile_identity

    corrected = copy.deepcopy(record)
    summary = corrected.get("summary", {})
    if summary.get("scoring_version") != SCORING_VERSION:
        return corrected
    validate_scoring_record(corrected)
    full = summary.get("scoring_profile")
    if full and "sha256" not in full:
        summary["scoring_engine_profile"] = copy.deepcopy(full)

    def compact(value):
        if isinstance(value, dict):
            for key, item in list(value.items()):
                if key == "scoring_profile" and item:
                    value[key] = scoring_profile_identity(item)
                elif key != "scoring_engine_profile":
                    compact(item)
        elif isinstance(value, list):
            for item in value:
                compact(item)

    compact(corrected)
    for row in corrected.get("results", []):
        if "model_move_scores" in row or "opponent_move_scores" in row:
            row["scoring_version"] = SCORING_VERSION
            row["scoring_profile"] = copy.deepcopy(summary["scoring_profile"])
    if "rescoring" in corrected:
        body = {key: value for key, value in corrected.items() if key != "rescoring"}
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        corrected["rescoring"]["output_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    validate_scoring_record(corrected)
    return corrected


def validate_scoring_record(
    record: dict[str, Any],
    *,
    required_version: str | None = None,
    required_profile: dict | None = None,
) -> None:
    """Validate every score's provenance, optionally enforcing a production epoch.

    Unstamped historical records remain readable for audit. Once production has
    migrated, its state supplies the required version and engine profile so an
    old runner cannot silently replace corrected data.
    """
    summary = record.get("summary", {})
    version = summary.get("scoring_version")
    profile = summary.get("scoring_profile")
    if required_version is not None and version != required_version:
        raise ValueError("Panel scoring version differs from production; rescore saved moves first")
    if required_profile is not None and profile != required_profile:
        raise ValueError("Panel scoring profile differs from production; use the production engine profile")
    scores = []
    for row in record.get("results", []):
        if not isinstance(row, dict):
            if version or profile:
                raise ValueError("Mixed or invalid scoring rows in panel")
            continue
        if "model_move_scores" in row or "opponent_move_scores" in row:
            if row.get("scoring_version") or row.get("scoring_profile"):
                if row.get("scoring_version") != version or row.get("scoring_profile") != profile:
                    raise ValueError("Mixed scoring versions or engine profiles in continuation row")
            scores.extend(row.get("model_move_scores", []))
            scores.extend(row.get("opponent_move_scores", []))
        else:
            scores.append(row)
    stamped = any(score.get("scoring_version") or score.get("scoring_profile") for score in scores)
    if not version and not profile and not stamped:
        return
    if not isinstance(version, str) or not version or not isinstance(profile, dict) or not profile:
        raise ValueError("Mixed or missing scoring provenance in panel summary")
    if summary.get("scoring_engine_profile"):
        from position_benchmark.evaluation import scoring_profile_identity
        try:
            audit_identity = scoring_profile_identity(summary["scoring_engine_profile"])
            expected_identity = scoring_profile_identity(profile)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid full scoring engine profile") from error
        if audit_identity != expected_identity:
            raise ValueError("Full scoring engine profile does not match its compact identity")
    if any(score.get("scoring_version") != version or score.get("scoring_profile") != profile for score in scores):
        raise ValueError("Mixed scoring versions or engine profiles in panel")

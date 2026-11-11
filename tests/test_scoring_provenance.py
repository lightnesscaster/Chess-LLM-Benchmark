import copy

import pytest

from position_benchmark.scoring_provenance import validate_scoring_record


def record():
    stamp = {"scoring_version": "root-reference-v1", "scoring_profile": {"engine": "Stockfish"}}
    return {"summary": dict(stamp), "results": [{**stamp, "cpl": 0}]}


def test_corrected_static_records_require_every_row_to_match():
    data = record()
    validate_scoring_record(data)
    data["results"].append({"cpl": 10})
    with pytest.raises(ValueError, match="Mixed"):
        validate_scoring_record(data)


def test_legacy_is_readable_but_cannot_replace_corrected_production():
    validate_scoring_record({"results": [{"cpl": 10}]})
    with pytest.raises(ValueError, match="scoring version"):
        validate_scoring_record({"results": [{"cpl": 10}]}, required_version="root-reference-v1")


def test_different_engine_profile_is_rejected():
    with pytest.raises(ValueError, match="profile"):
        validate_scoring_record(record(), required_profile={"engine": "Other"})


@pytest.mark.parametrize("panel", ["core", "game_like", "stability"])
def test_panel_merge_cannot_downgrade_previously_corrected_scores(panel):
    from position_benchmark.sync import merge_panel_record

    old = record() if panel == "core" else {"supplements": {panel: record()}}
    with pytest.raises(ValueError, match="scoring version"):
        merge_panel_record(old, panel, {"results": [{"cpl": 1000}]})


def test_continuation_checks_both_sides_and_accepts_terminal_empty_start():
    data = record()
    stamp = copy.deepcopy(data["summary"])
    data["results"] = [
        {"model_move_scores": [{**stamp, "cpl": 0}], "opponent_move_scores": [{**stamp, "cpl": 2}]},
        {"model_move_scores": [], "opponent_move_scores": []},
    ]
    validate_scoring_record(data)
    data["results"][0]["opponent_move_scores"][0]["scoring_profile"] = {"engine": "Other"}
    with pytest.raises(ValueError, match="Mixed"):
        validate_scoring_record(data)


def test_compacting_running_job_output_preserves_full_profile_and_responses():
    from position_benchmark.scoring_provenance import compact_record_profiles
    from position_benchmark.evaluation import scoring_profile_identity

    full = {"scoring_version": "root-reference-v1", "engine": {"name": "Stockfish"}, "depth": 30, "options": {"Hash": 64}}
    stamp = {"scoring_version": "root-reference-v1", "scoring_profile": full}
    old = {"summary": dict(stamp), "results": [{**stamp, "cpl": 0, "model_move": "e2e4", "prompt_tokens": 10}]}
    saved = copy.deepcopy(old)
    compact = compact_record_profiles(old)
    assert old == saved
    assert compact["summary"]["scoring_profile"] == scoring_profile_identity(full)
    assert compact["summary"]["scoring_engine_profile"] == full
    assert compact["results"][0]["model_move"] == "e2e4"
    assert compact["results"][0]["prompt_tokens"] == 10
    validate_scoring_record(compact)
    assert compact_record_profiles(compact) == compact

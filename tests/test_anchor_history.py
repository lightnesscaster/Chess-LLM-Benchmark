import pytest

from rating.anchor_history import anchor_rating_at


@pytest.mark.parametrize("timestamp,expected", [
    ("2026-09-08T13:02:04Z", 2211),
    ("2026-09-08T13:02:05Z", 2346),
    ("2026-09-08T15:02:04+02:00", 2211),
    ("2026-09-08T13:02:04", 2211),
    ("2027-01-01T00:00:00Z", 2346),
])
def test_history_boundary_uses_game_time(timestamp, expected):
    config = {"rating": 2346, "rating_history": [
        {"before": "2026-09-08T13:02:05Z", "rating": 2211},
    ]}
    assert anchor_rating_at(config, timestamp) == expected


def test_unchanged_anchor_does_not_require_timestamp():
    assert anchor_rating_at({"rating": 1628}, "") == 1628


def test_changed_anchor_rejects_unknown_timestamp_instead_of_rerating_history():
    with pytest.raises(ValueError):
        anchor_rating_at({"rating": 2346, "rating_history": [
            {"before": "2026-09-08T13:02:05Z", "rating": 2211},
        ]}, "invalid")

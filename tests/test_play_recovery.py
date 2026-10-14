import copy
import importlib
from types import SimpleNamespace

import pytest


@pytest.fixture
def recovery_case():
    state = {
        "game_id": "11111111-1111-4111-8111-111111111111",
        "started_at": "1970-01-01T00:10:00+00:00",
        "model_name": "provider/test", "model_id": "test",
        "rated_model_id": "test (medium)", "reasoning_effort": "medium",
        "human_color": "white", "status": "active", "winner": None,
        "termination": None, "moves": ["e2e4", "e7e5", "f1c4", "b8c6"],
        "human_profile": {"username": "Player", "rating": 1847, "rating_deviation": 73},
        "rating_context": {"self": {"rating": 1700}, "opponent": {"rating": 1847}},
        "llm_tokens": {"prompt_tokens": 100, "completion_tokens": 7, "total_tokens": 107},
        "llm_accounting_status": "complete", "llm_illegal_moves": 0,
    }
    record = {
        "owner_email": "player@example.com", "model_name": "provider/test",
        "human_color": "white", "expected_moves": list(state["moves"]),
        "restored_moves": ["e2e4", "e7e5", "g1f3", "g8f6"],
        "created_at": 1000, "expires_at": 3000,
    }
    return state, record


def test_recovery_preserves_rated_identity_and_snapshot_without_replaying(recovery_case):
    service = importlib.import_module("web.play_recovery")
    state, record = recovery_case
    original = copy.deepcopy(state)

    restored = service.prepare_recovery(state, record, "player@example.com", now=2000)

    assert restored["moves"] == ["e2e4", "e7e5", "g1f3", "g8f6"]
    for field in ["game_id", "started_at", "rated_model_id", "reasoning_effort", "human_profile", "rating_context", "llm_tokens"]:
        assert restored[field] == original[field]
    assert restored["llm_accounting_status"] == "partial"
    assert state == original


@pytest.mark.parametrize("change", [
    {"owner_email": "someone-else@example.com"},
    {"model_name": "provider/other"},
    {"human_color": "black"},
    {"expected_moves": ["e2e4", "e7e5"]},
    {"restored_moves": ["d2d4", "d7d5", "c2c4", "e7e6"]},
    {"restored_moves": ["e2e4", "e7e5", "g1f3", "e8e1"]},
    {"expires_at": 1900},
    {"created_at": 500},
    {"used_game_id": "22222222-2222-4222-8222-222222222222"},
])
def test_recovery_rejects_unapproved_or_expired_changes(recovery_case, change):
    service = importlib.import_module("web.play_recovery")
    state, record = recovery_case
    with pytest.raises(service.RecoveryError):
        service.prepare_recovery(state, {**record, **change}, "player@example.com", now=2000)


def test_recovery_does_not_reopen_finished_games(recovery_case):
    service = importlib.import_module("web.play_recovery")
    state, record = recovery_case
    with pytest.raises(service.RecoveryError):
        service.prepare_recovery({**state, "status": "finished"}, record, "player@example.com", now=2000)


@pytest.fixture
def recovery_store(recovery_case, monkeypatch):
    from firebase_admin import firestore
    from web import play_recovery as service

    state, record = recovery_case
    key = service.recovery_id(state, "player@example.com")
    documents = {(service.RECOVERIES_COLLECTION, key): copy.deepcopy(record)}
    reads, writes = [], []

    class Reference:
        def __init__(self, collection, document_id):
            self.key = (collection, document_id)

        def get(self, transaction=None):
            reads.append((self.key, transaction))
            data = documents.get(self.key)
            return SimpleNamespace(exists=data is not None, to_dict=lambda: copy.deepcopy(data))

    class Transaction:
        def update(self, reference, fields):
            writes.append((reference.key, copy.deepcopy(fields)))
            documents[reference.key].update(fields)

    transaction = Transaction()
    db = SimpleNamespace(
        collection=lambda name: SimpleNamespace(document=lambda key: Reference(name, key)),
        transaction=lambda: transaction,
    )
    monkeypatch.setattr(firestore, "transactional", lambda function: function)
    monkeypatch.setattr(service.time, "time", lambda: 2000)
    return service.PlayRecoveryStore(db), state, key, documents, reads, writes, transaction


def test_recovery_claim_reads_original_result_and_can_be_consumed_only_once(recovery_store):
    from web.play_recovery import RecoveryError
    store, state, key, documents, reads, writes, transaction = recovery_store
    assert store.lookup(state, "player@example.com") == {"id": key}

    restored = store.restore(state, "player@example.com", key)

    assert restored["moves"] == ["e2e4", "e7e5", "g1f3", "g8f6"]
    assert (("results", state["game_id"]), transaction) in reads
    assert len(writes) == 1
    assert writes[0][0] == ("play_recoveries", key)
    assert writes[0][1]["used_game_id"] == state["game_id"]
    assert writes[0][1]["used_at"] == 2000
    assert len(writes[0][1]["source_state_sha256"]) == 64
    assert store.lookup(restored, "player@example.com") is None
    assert store.lookup(state, "player@example.com") == {"id": key}
    # A lost response leaves the original signed cookie; redelivery must be safe.
    assert store.restore(state, "player@example.com", key) == restored
    assert len(writes) == 1


def test_consumed_recovery_cannot_restore_another_identity_or_changed_snapshot(recovery_store):
    from web.play_recovery import RecoveryError
    store, state, key, documents, reads, writes, transaction = recovery_store
    store.restore(state, "player@example.com", key)
    for changed in [
        {**state, "game_id": "22222222-2222-4222-8222-222222222222"},
        {**state, "human_profile": {**state["human_profile"], "rating": 2200}},
    ]:
        with pytest.raises(RecoveryError):
            store.restore(changed, "player@example.com", key)
    assert len(writes) == 1


@pytest.mark.parametrize("rejection", ["completed", "wrong_owner", "wrong_approval", "advanced_moves"])
def test_recovery_claim_cannot_target_completed_or_different_games(recovery_store, rejection):
    from web.play_recovery import RecoveryError
    store, state, key, documents, reads, writes, transaction = recovery_store
    email = "player@example.com"
    if rejection == "completed":
        documents[("results", state["game_id"])] = {"winner": "white"}
    elif rejection == "wrong_owner":
        email = "other@example.com"
    elif rejection == "wrong_approval":
        key = "different-key"
    else:
        state = {**state, "moves": state["moves"] + ["d2d3", "g8f6"]}
    with pytest.raises(RecoveryError):
        store.restore(state, email, key)
    assert not writes

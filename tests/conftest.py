"""Isolated storage for interactive-play route and concurrency tests."""
import copy
import threading
from types import SimpleNamespace

import pytest


class FakePlayFirestore:
    def __init__(self):
        self.records = {}
        self.lock = threading.RLock()

    def collection(self, name):
        db = self
        class Reference:
            def __init__(self, key):
                self.key = (name, key)
            def get(self, transaction=None):
                data = copy.deepcopy(db.records.get(self.key))
                return SimpleNamespace(exists=data is not None, to_dict=lambda: data)
        return SimpleNamespace(document=Reference)

    def transaction(self):
        db = self
        class Transaction:
            def set(self, ref, data):
                if getattr(db, "fail_next_write", False):
                    db.fail_next_write = False
                    raise OSError("Temporary database outage")
                db.records[ref.key] = copy.deepcopy(data)
        return Transaction()


@pytest.fixture
def play_store(monkeypatch, tmp_path):
    from firebase_admin import firestore
    from web.play_store import ActivePlayStore
    import web.app as web_app
    db = FakePlayFirestore()
    def transactional(function):
        def run(transaction):
            with db.lock:
                return function(transaction)
        return run
    monkeypatch.setattr(firestore, "transactional", transactional)
    store = ActivePlayStore(db, journal_dir=tmp_path / "turns")
    monkeypatch.setattr(web_app, "get_active_play_store", lambda: store, raising=False)
    return store

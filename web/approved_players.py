"""Firestore-backed allowlist for rated human players."""

from __future__ import annotations

import re
import hashlib
from datetime import datetime, timezone
from functools import wraps

from flask import abort, jsonify, redirect, request, url_for

from firebase_client import APPROVED_PLAYERS_COLLECTION, get_firestore_client


EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def normalize_email(value: object) -> str:
    """Normalize a user-provided email for document IDs and comparisons."""
    return str(value or "").strip().casefold()


def _document_id(email: str) -> str:
    """Use a fixed safe Firestore ID even for unusual valid email characters."""
    return hashlib.sha256(email.encode("utf-8")).hexdigest()


class ApprovedPlayerStore:
    """Manage approved player records without exposing Firestore to routes."""

    def __init__(self, db=None):
        self._db = db or get_firestore_client()
        self._collection = self._db.collection(APPROVED_PLAYERS_COLLECTION)

    def list_players(self) -> list[dict]:
        records = []
        for document in self._collection.stream():
            data = document.to_dict() or {}
            email = normalize_email(data.get("email") or document.id)
            if email:
                records.append({**data, "email": email})
        return sorted(records, key=lambda record: record["email"])

    def get_player(self, email: object) -> dict | None:
        normalized = normalize_email(email)
        if not normalized:
            return None
        document = self._collection.document(_document_id(normalized)).get()
        if not document.exists:
            return None
        data = document.to_dict() or {}
        return {**data, "email": normalized}

    def approve(self, email: object, approved_by: object) -> dict:
        normalized = normalize_email(email)
        if len(normalized) > 254 or not EMAIL_PATTERN.fullmatch(normalized):
            raise ValueError("Enter a valid email address.")
        existing = self.get_player(normalized)
        if existing:
            return existing
        record = {
            "email": normalized,
            "approved_by": normalize_email(approved_by),
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "lichess_username": None,
        }
        self._collection.document(_document_id(normalized)).set(record)
        return record

    def remove(self, email: object) -> bool:
        normalized = normalize_email(email)
        if not normalized:
            return False
        reference = self._collection.document(_document_id(normalized))
        existed = reference.get().exists
        if existed:
            reference.delete()
        return existed

    def claim_lichess_username(
        self,
        email: object,
        username: str,
        *,
        allow_missing: bool = False,
    ) -> dict:
        """Atomically set the first username or confirm the existing claim."""
        from firebase_admin import firestore

        normalized = normalize_email(email)
        canonical_username = str(username or "").strip()
        if not normalized or not canonical_username:
            raise ValueError("A verified Lichess username is required.")
        reference = self._collection.document(_document_id(normalized))
        transaction = self._db.transaction()

        @firestore.transactional
        def claim(transaction):
            snapshot = reference.get(transaction=transaction)
            if not snapshot.exists and not allow_missing:
                raise ValueError("Player approval is required.")
            record = (
                snapshot.to_dict()
                if snapshot.exists
                else {
                    "email": normalized,
                    "approved_by": normalized,
                    "approved_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            existing_username = str(record.get("lichess_username") or "").strip()
            if (
                existing_username
                and existing_username.casefold() != canonical_username.casefold()
            ):
                raise ValueError(
                    "This account is already linked to a different Lichess username."
                )
            updated = {**record, "lichess_username": canonical_username}
            transaction.set(reference, updated, merge=True)
            return updated

        return claim(transaction)

    def save_lichess_username(self, email: object, username: str) -> dict:
        """Backward-compatible alias for atomic username claiming."""
        return self.claim_lichess_username(email, username)


def get_approved_player_store() -> ApprovedPlayerStore:
    """Return the default approved-player store."""
    return ApprovedPlayerStore()


def is_approved_player(user: dict | None = None, store=None) -> bool:
    """Return whether a verified account may enter the rated arena."""
    from web.auth import current_user, is_admin

    selected_user = user if user is not None else current_user()
    if not selected_user or selected_user.get("email_verified") is not True:
        return False
    if is_admin(selected_user):
        return True
    selected_store = store or get_approved_player_store()
    return selected_store.get_player(selected_user.get("email")) is not None


def player_required(view):
    """Require a signed-in approved account for an HTML page."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        from web.auth import current_user

        if current_user() is None:
            return redirect(url_for("login", next=request.path))
        if not is_approved_player():
            abort(403)
        return view(*args, **kwargs)

    return wrapped


def player_api_required(view):
    """Require an approved account for a JSON API."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not is_approved_player():
            return jsonify({"error": "Approved player access is required."}), 403
        return view(*args, **kwargs)

    return wrapped

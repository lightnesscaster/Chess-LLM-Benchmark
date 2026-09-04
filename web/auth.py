"""Firebase-backed login sessions and authorization helpers."""

from __future__ import annotations

import json
import os
import secrets
from functools import wraps
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

import firebase_admin
from firebase_admin import auth as firebase_auth
from firebase_admin import exceptions as firebase_exceptions
from flask import abort, current_app, jsonify, redirect, request, session, url_for


def _normalize_email(value: object) -> str:
    return str(value or "").strip().casefold()


def _configured_admin_emails() -> set[str]:
    return {
        _normalize_email(email)
        for email in os.environ.get("ADMIN_EMAILS", "").split(",")
        if _normalize_email(email)
    }


def _ensure_firebase_app():
    if firebase_admin._apps:
        return firebase_admin.get_app()

    from firebase_client import _get_credentials

    return firebase_admin.initialize_app(_get_credentials())


def verify_firebase_token(id_token: str) -> dict | None:
    """Verify a Firebase ID token and return its claims."""
    if not isinstance(id_token, str) or not id_token.strip():
        return None
    try:
        return firebase_auth.verify_id_token(
            id_token.strip(),
            app=_ensure_firebase_app(),
            check_revoked=True,
        )
    except (ValueError, firebase_exceptions.FirebaseError):
        return None


def current_user() -> dict | None:
    """Return the verified user stored in the signed Flask session."""
    user = session.get("user")
    if not isinstance(user, dict):
        return None
    if not user.get("uid") or not user.get("email"):
        return None
    return user


def is_admin(user: dict | None = None) -> bool:
    """Return whether a verified session user is in the admin allowlist."""
    selected_user = user if user is not None else current_user()
    if not selected_user or selected_user.get("email_verified") is not True:
        return False
    return _normalize_email(selected_user.get("email")) in _configured_admin_emails()


def ensure_csrf_token() -> str:
    """Create and return the per-session CSRF token."""
    if not current_app.secret_key:
        abort(503, description="Login is not configured.")
    token = session.get("csrf_token")
    if not isinstance(token, str) or len(token) < 32:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token


def validate_csrf() -> None:
    """Reject a state-changing request without the session CSRF token."""
    expected = session.get("csrf_token")
    provided = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token")
    if (
        not isinstance(expected, str)
        or not isinstance(provided, str)
        or not secrets.compare_digest(expected, provided)
    ):
        abort(403, description="Invalid CSRF token.")


def safe_next_url(value: object, default: str = "/") -> str:
    """Allow only local absolute paths as post-login destinations."""
    candidate = str(value or "")
    parsed = urlsplit(candidate)
    if (
        not candidate.startswith("/")
        or candidate.startswith("//")
        or "\\" in candidate
        or parsed.scheme
        or parsed.netloc
    ):
        return default
    return candidate


def firebase_web_config() -> dict | None:
    """Return the public Firebase Web SDK configuration."""
    api_key = os.environ.get("FIREBASE_WEB_API_KEY", "").strip()
    project_id = os.environ.get("FIREBASE_PROJECT_ID", "").strip()

    if not project_id:
        credentials_json = os.environ.get("FIREBASE_CREDENTIALS_JSON", "")
        if credentials_json:
            try:
                project_id = str(json.loads(credentials_json).get("project_id", "")).strip()
            except json.JSONDecodeError:
                project_id = ""

    if not project_id:
        credentials_path = Path(__file__).parent.parent / "firebase-key.json"
        try:
            project_id = str(json.loads(credentials_path.read_text()).get("project_id", "")).strip()
        except (OSError, json.JSONDecodeError):
            project_id = ""

    if not api_key or not project_id:
        return None

    auth_domain = os.environ.get(
        "FIREBASE_AUTH_DOMAIN",
        f"{project_id}.firebaseapp.com",
    ).strip()
    return {
        "apiKey": api_key,
        "authDomain": auth_domain,
        "projectId": project_id,
    }


def login_required(view: Callable):
    """Require an authenticated Firebase session for a page."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if current_user() is None:
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def admin_required(view: Callable):
    """Require an authenticated allowlisted administrator for a page."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if current_user() is None:
            return redirect(url_for("login", next=request.path))
        if not is_admin():
            abort(403)
        return view(*args, **kwargs)

    return wrapped


def admin_api_required(view: Callable):
    """Require an allowlisted administrator for a JSON API."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not is_admin():
            return jsonify({"error": "Administrator access is required."}), 403
        return view(*args, **kwargs)

    return wrapped

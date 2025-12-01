"""
Firebase client initialization and utilities.
"""

import json
import os
from pathlib import Path
from functools import lru_cache

import firebase_admin
from firebase_admin import credentials, firestore


def _get_credentials():
    """Get Firebase credentials from file or environment variable."""
    # Check for JSON credentials in environment variable (for Render/production)
    creds_json = os.environ.get("FIREBASE_CREDENTIALS_JSON")
    if creds_json:
        try:
            creds_dict = json.loads(creds_json)
            return credentials.Certificate(creds_dict)
        except json.JSONDecodeError:
            pass

    # Check GOOGLE_APPLICATION_CREDENTIALS path
    env_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env_path and Path(env_path).exists():
        return credentials.Certificate(env_path)

    # Check common locations
    possible_paths = [
        Path(__file__).parent / "firebase-key.json",
        Path.cwd() / "firebase-key.json",
        Path.home() / ".config" / "firebase-key.json",
    ]

    for path in possible_paths:
        if path.exists():
            return credentials.Certificate(str(path))

    raise FileNotFoundError(
        "Firebase credentials not found. Set FIREBASE_CREDENTIALS_JSON, "
        "GOOGLE_APPLICATION_CREDENTIALS, or place firebase-key.json in the project root."
    )


@lru_cache(maxsize=1)
def get_firestore_client():
    """
    Get a Firestore client instance (singleton).

    Returns:
        Firestore client
    """
    if not firebase_admin._apps:
        cred = _get_credentials()
        firebase_admin.initialize_app(cred)

    return firestore.client()


# Collection names
RATINGS_COLLECTION = "ratings"
GAMES_COLLECTION = "games"
RESULTS_COLLECTION = "results"

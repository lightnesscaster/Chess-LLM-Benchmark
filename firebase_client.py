"""
Firebase client initialization and utilities.
"""

import os
from pathlib import Path
from functools import lru_cache

import firebase_admin
from firebase_admin import credentials, firestore


def _find_credentials_path() -> str:
    """Find the Firebase credentials file."""
    # Check environment variable first
    env_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env_path and Path(env_path).exists():
        return env_path

    # Check common locations
    possible_paths = [
        Path(__file__).parent / "firebase-key.json",
        Path.cwd() / "firebase-key.json",
        Path.home() / ".config" / "firebase-key.json",
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        "Firebase credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS "
        "or place firebase-key.json in the project root."
    )


@lru_cache(maxsize=1)
def get_firestore_client():
    """
    Get a Firestore client instance (singleton).

    Returns:
        Firestore client
    """
    if not firebase_admin._apps:
        cred_path = _find_credentials_path()
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)

    return firestore.client()


# Collection names
RATINGS_COLLECTION = "ratings"
GAMES_COLLECTION = "games"
RESULTS_COLLECTION = "results"

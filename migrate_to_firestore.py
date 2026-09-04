#!/usr/bin/env python3
"""
Migrate local data to Firestore.

Usage:
    python migrate_to_firestore.py

Requires firebase-key.json in project root.
"""

import json
from pathlib import Path

from firebase_client import get_firestore_client, RATINGS_COLLECTION, GAMES_COLLECTION, RESULTS_COLLECTION


def migrate_ratings():
    """Migrate ratings.json to Firestore."""
    ratings_path = Path("data/ratings.json")
    if not ratings_path.exists():
        print("No ratings.json found, skipping ratings migration.")
        return 0

    db = get_firestore_client()

    with open(ratings_path) as f:
        ratings = json.load(f)

    batch = db.batch()
    count = 0
    for player_id, rating_data in ratings.items():
        ref = db.collection(RATINGS_COLLECTION).document(player_id)
        batch.set(ref, rating_data)
        count += 1

        # Firestore batches are limited to 500 operations
        if count % 400 == 0:
            batch.commit()
            batch = db.batch()

    if count % 400 != 0:
        batch.commit()

    print(f"Migrated {count} ratings to Firestore.")
    return count


def migrate_results():
    """Migrate result JSON files to Firestore."""
    results_dir = Path("data/results")
    if not results_dir.exists():
        print("No results directory found, skipping results migration.")
        return 0

    db = get_firestore_client()

    batch = db.batch()
    count = 0
    for path in results_dir.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)

            game_id = path.stem
            ref = db.collection(RESULTS_COLLECTION).document(game_id)
            batch.set(ref, data)
            count += 1

            if count % 400 == 0:
                batch.commit()
                batch = db.batch()
                print(f"  Migrated {count} results...")

        except Exception as e:
            print(f"  Error migrating {path.name}: {e}")

    if count % 400 != 0:
        batch.commit()

    print(f"Migrated {count} game results to Firestore.")
    return count


def migrate_games():
    """Migrate PGN files to Firestore."""
    games_dir = Path("data/games")
    if not games_dir.exists():
        print("No games directory found, skipping PGN migration.")
        return 0

    db = get_firestore_client()

    batch = db.batch()
    count = 0
    for path in games_dir.glob("*.pgn"):
        try:
            with open(path) as f:
                pgn_content = f.read()

            game_id = path.stem
            ref = db.collection(GAMES_COLLECTION).document(game_id)
            batch.set(ref, {
                "game_id": game_id,
                "pgn": pgn_content,
            })
            count += 1

            if count % 400 == 0:
                batch.commit()
                batch = db.batch()
                print(f"  Migrated {count} games...")

        except Exception as e:
            print(f"  Error migrating {path.name}: {e}")

    if count % 400 != 0:
        batch.commit()

    print(f"Migrated {count} PGN files to Firestore.")
    return count


def main():
    print("Starting migration to Firestore...")
    print()

    print("1. Migrating ratings...")
    migrate_ratings()
    print()

    print("2. Migrating game results...")
    migrate_results()
    print()

    print("3. Migrating PGN files...")
    migrate_games()
    print()

    print("Migration complete!")


if __name__ == "__main__":
    main()

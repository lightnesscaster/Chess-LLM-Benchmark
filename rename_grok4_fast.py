#!/usr/bin/env python3
"""
Rename 'grok-4-fast (high)' to 'grok-4-fast' in all Firebase collections.

Usage:
    python rename_grok4_fast.py
"""

from firebase_client import get_firestore_client, RATINGS_COLLECTION, GAMES_COLLECTION, RESULTS_COLLECTION

OLD_NAME = "grok-4-fast (high)"
NEW_NAME = "grok-4-fast"


def rename_in_ratings():
    """Rename rating document from old name to new name."""
    db = get_firestore_client()

    old_ref = db.collection(RATINGS_COLLECTION).document(OLD_NAME)
    old_doc = old_ref.get()

    if not old_doc.exists:
        print(f"No rating document found for '{OLD_NAME}'")
        return 0

    # Copy to new document
    new_ref = db.collection(RATINGS_COLLECTION).document(NEW_NAME)
    new_ref.set(old_doc.to_dict())

    # Delete old document
    old_ref.delete()

    print(f"Renamed rating document: '{OLD_NAME}' -> '{NEW_NAME}'")
    return 1


def rename_in_results():
    """Update white_id and black_id fields in results collection."""
    db = get_firestore_client()
    results_ref = db.collection(RESULTS_COLLECTION)

    # Find documents where white_id matches
    white_query = results_ref.where("white_id", "==", OLD_NAME).stream()
    white_count = 0
    for doc in white_query:
        doc.reference.update({"white_id": NEW_NAME})
        white_count += 1

    print(f"Updated {white_count} results with white_id = '{OLD_NAME}'")

    # Find documents where black_id matches
    black_query = results_ref.where("black_id", "==", OLD_NAME).stream()
    black_count = 0
    for doc in black_query:
        doc.reference.update({"black_id": NEW_NAME})
        black_count += 1

    print(f"Updated {black_count} results with black_id = '{OLD_NAME}'")

    return white_count + black_count


def rename_in_games():
    """Update PGN content in games collection to replace old name with new name."""
    db = get_firestore_client()
    games_ref = db.collection(GAMES_COLLECTION)

    # Need to scan all games and check PGN content
    count = 0
    batch = db.batch()
    batch_count = 0

    for doc in games_ref.stream():
        data = doc.to_dict()
        pgn = data.get("pgn", "")

        if OLD_NAME in pgn:
            new_pgn = pgn.replace(OLD_NAME, NEW_NAME)
            batch.update(doc.reference, {"pgn": new_pgn})
            count += 1
            batch_count += 1

            if batch_count >= 400:
                batch.commit()
                batch = db.batch()
                batch_count = 0
                print(f"  Updated {count} games so far...")

    if batch_count > 0:
        batch.commit()

    print(f"Updated {count} games with PGN containing '{OLD_NAME}'")
    return count


def main():
    print(f"Renaming '{OLD_NAME}' to '{NEW_NAME}' in Firebase...")
    print()

    print("1. Updating ratings collection...")
    rename_in_ratings()
    print()

    print("2. Updating results collection...")
    rename_in_results()
    print()

    print("3. Updating games collection (PGN content)...")
    rename_in_games()
    print()

    print("Rename complete!")


if __name__ == "__main__":
    main()

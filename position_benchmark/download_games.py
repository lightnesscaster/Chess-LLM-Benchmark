"""
Download all games from Firestore to local PGN files.
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from firebase_client import get_firestore_client, GAMES_COLLECTION, RESULTS_COLLECTION


def main():
    output_dir = Path("position_benchmark/games")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Connecting to Firestore...")
    db = get_firestore_client()

    # Download games
    print("Fetching games collection...")
    games_docs = list(db.collection(GAMES_COLLECTION).stream())
    print(f"Found {len(games_docs)} games")

    # Download results for metadata
    print("Fetching results collection...")
    results_docs = list(db.collection(RESULTS_COLLECTION).stream())
    results_map = {doc.id: doc.to_dict() for doc in results_docs}
    print(f"Found {len(results_docs)} results")

    # Save each game
    saved = 0
    skipped = 0

    for doc in games_docs:
        game_id = doc.id
        data = doc.to_dict()
        pgn = data.get("pgn")

        if not pgn:
            print(f"  Skipping {game_id}: no PGN data")
            skipped += 1
            continue

        # Save PGN
        pgn_path = output_dir / f"{game_id}.pgn"
        with open(pgn_path, "w") as f:
            f.write(pgn)

        # Get result metadata if available
        result = results_map.get(game_id, {})
        white = result.get("white_id", "unknown")
        black = result.get("black_id", "unknown")

        saved += 1
        if saved % 100 == 0:
            print(f"  Saved {saved} games...")

    print(f"\nDone! Saved {saved} games, skipped {skipped}")
    print(f"Games saved to: {output_dir}")

    # Also save results as JSON for reference
    import json
    results_path = output_dir / "_results.json"
    with open(results_path, "w") as f:
        json.dump(results_map, f, indent=2, default=str)
    print(f"Results metadata saved to: {results_path}")


if __name__ == "__main__":
    main()

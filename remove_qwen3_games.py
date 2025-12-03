"""
Remove games involving qwen3-235b-a22b-thinking-2507 from Firestore.
"""

from firebase_client import get_firestore_client, GAMES_COLLECTION, RESULTS_COLLECTION

MODEL_TO_REMOVE = "qwen3-235b-a22b-thinking-2507"

def main():
    db = get_firestore_client()

    # Find games in results collection
    results_ref = db.collection(RESULTS_COLLECTION)
    results = results_ref.stream()

    games_to_delete = []

    for doc in results:
        data = doc.to_dict()

        white_id = data.get("white_id", "")
        black_id = data.get("black_id", "")

        if MODEL_TO_REMOVE not in white_id and MODEL_TO_REMOVE not in black_id:
            continue

        game_id = doc.id
        games_to_delete.append({
            "id": game_id,
            "white": white_id,
            "black": black_id,
            "result": data.get("result", ""),
        })

    print(f"Found {len(games_to_delete)} games involving {MODEL_TO_REMOVE}:")
    for g in games_to_delete:
        print(f"  {g['id']}: {g['white']} vs {g['black']} -> {g['result']}")

    if not games_to_delete:
        print("No games to delete.")
        return

    confirm = input(f"\nDelete {len(games_to_delete)} games from Firestore? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return

    # Delete from both collections
    deleted_results = 0
    deleted_games = 0

    for g in games_to_delete:
        game_id = g["id"]

        # Delete from results collection
        try:
            db.collection(RESULTS_COLLECTION).document(game_id).delete()
            deleted_results += 1
        except Exception as e:
            print(f"  Error deleting result {game_id}: {e}")

        # Delete from games collection (PGN)
        try:
            db.collection(GAMES_COLLECTION).document(game_id).delete()
            deleted_games += 1
        except Exception as e:
            print(f"  Error deleting game {game_id}: {e}")

    print(f"\nDeleted {deleted_results} results and {deleted_games} games from Firestore.")

if __name__ == "__main__":
    main()

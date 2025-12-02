"""
Remove bad gpt-5.1-chat games from Firestore.
Games played today with 4 or fewer moves.
"""

from datetime import datetime, timezone
from firebase_client import get_firestore_client, GAMES_COLLECTION, RESULTS_COLLECTION

def main():
    db = get_firestore_client()

    # Get today's date
    today = datetime.now(timezone.utc).date()

    # Find bad games in results collection
    results_ref = db.collection(RESULTS_COLLECTION)
    results = results_ref.stream()

    games_to_delete = []

    for doc in results:
        data = doc.to_dict()

        # Check if gpt-5.1-chat is involved
        white_id = data.get("white_id", "")
        black_id = data.get("black_id", "")

        if "gpt-5.1-chat" not in white_id and "gpt-5.1-chat" not in black_id:
            continue

        # Check move count (4 or fewer)
        moves = data.get("moves", 0)
        if moves > 4:
            continue

        # Check if played today
        created_at = data.get("created_at", "")
        if created_at:
            try:
                game_date = datetime.fromisoformat(created_at.replace("Z", "+00:00")).date()
                if game_date != today:
                    continue
            except (ValueError, AttributeError):
                continue
        else:
            continue

        game_id = doc.id
        games_to_delete.append({
            "id": game_id,
            "white": white_id,
            "black": black_id,
            "moves": moves,
            "created_at": created_at
        })

    print(f"Found {len(games_to_delete)} bad games to delete:")
    for g in games_to_delete:
        print(f"  {g['id']}: {g['white']} vs {g['black']} ({g['moves']} moves)")

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

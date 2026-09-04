#!/usr/bin/env python3
"""Clear invalid accounting metadata from the first Gemini 3.6 game cohort."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firebase_client import RESULTS_COLLECTION, get_firestore_client  # noqa: E402


PLAYER_ID = "gemini-3.6-flash (medium)"
GAME_IDS = (
    "016aab21-96a9-4651-b132-2ecc25e903a0",
    "06ca2443-f457-4aad-bad2-d9b0899de4de",
    "0a517c8b-7472-4c2f-a3ad-27b967c72493",
    "2a67a75e-3efd-41f3-876e-8a2e78ddc4c4",
    "3ea17f4b-7f54-4544-9a31-de51f9f3086f",
    "404877e5-929b-4198-8f88-0b1e9d37401f",
    "40cd6964-4c15-4e12-ab49-31480b2b1cf5",
    "5d3cc59e-d5ef-440c-91a4-da02339bcf30",
    "62c1d216-e819-416c-bfbc-50be9c2d8a1e",
    "8238d5c9-064c-43b9-93c9-dbf1f552ce41",
    "88b6812d-537e-4949-a6ee-1d8e8383f2d2",
    "95b3e9e0-2f7b-4f9b-b033-c403cbd78656",
    "983d1272-4863-4f2c-89e8-d0143c3799a1",
    "b5f7b2d5-6976-4cbd-b202-da2fc788e1e8",
    "e5324e9c-6cd9-4073-8e3f-553735531d12",
)
ACCOUNTING_STATUS = (
    "unavailable: initial Gemini 3.6 cohort shared mutable player state "
    "across concurrent games and omitted hidden thinking tokens"
)


def repair(*, apply: bool) -> int:
    """Verify the fixed cohort and optionally clear only its invalid fields."""
    db = get_firestore_client()
    collection = db.collection(RESULTS_COLLECTION)
    pending: list[tuple[object, dict]] = []

    for game_id in GAME_IDS:
        ref = collection.document(game_id)
        snapshot = ref.get()
        if not snapshot.exists:
            raise RuntimeError(f"Missing result document: {game_id}")
        record = snapshot.to_dict()
        sides = [
            side
            for side in ("white", "black")
            if record.get(f"{side}_id") == PLAYER_ID
        ]
        if len(sides) != 1:
            raise RuntimeError(
                f"{game_id}: expected {PLAYER_ID!r} on exactly one side"
            )
        side = sides[0]
        updates = {
            f"tokens_{side}": None,
            f"timing_{side}": None,
            f"accounting_status_{side}": ACCOUNTING_STATUS,
        }
        if all(record.get(field) == value for field, value in updates.items()):
            print(f"{game_id}: {side}: already repaired")
            continue
        pending.append((ref, updates))
        print(f"{game_id}: {side}: {'repair' if apply else 'would repair'}")

    if apply:
        batch = db.batch()
        for ref, updates in pending:
            batch.update(ref, updates)
        if pending:
            batch.commit()
        print(f"Repaired {len(pending)} result documents")
    else:
        print(
            f"Verified {len(GAME_IDS)} result documents; "
            f"{len(pending)} need repair; no writes made"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the verified updates (default is a dry run)",
    )
    args = parser.parse_args()
    return repair(apply=args.apply)


if __name__ == "__main__":
    raise SystemExit(main())

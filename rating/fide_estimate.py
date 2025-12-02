"""
FIDE rating estimation from Lichess Classical ratings.

Uses empirical data from ChessGoals.com rating comparison survey.
"""

import json
from pathlib import Path
from typing import Optional

# Path to mapping data
DATA_PATH = Path(__file__).parent.parent / "data" / "lichess_to_fide.json"

# Cache for loaded mappings
_mappings: Optional[list] = None


def _load_mappings() -> list:
    """Load and cache the Lichess to FIDE mappings."""
    global _mappings
    if _mappings is None:
        with open(DATA_PATH) as f:
            data = json.load(f)
        _mappings = sorted(data["mappings"], key=lambda x: x["lichess_classical"])
    return _mappings


def estimate_fide(lichess_classical: float) -> Optional[int]:
    """
    Estimate FIDE rating from Lichess Classical rating using linear interpolation.

    Args:
        lichess_classical: The Lichess Classical rating

    Returns:
        Estimated FIDE rating, or None if outside the data range
    """
    mappings = _load_mappings()

    if not mappings:
        return None

    # Get range bounds
    min_lichess = mappings[0]["lichess_classical"]
    max_lichess = mappings[-1]["lichess_classical"]

    # Handle out-of-range values by extrapolation
    if lichess_classical < min_lichess:
        # Extrapolate below range using first two points
        if len(mappings) < 2:
            return None
        x1, y1 = mappings[0]["lichess_classical"], mappings[0]["fide"]
        x2, y2 = mappings[1]["lichess_classical"], mappings[1]["fide"]
        slope = (y2 - y1) / (x2 - x1)
        fide = y1 + slope * (lichess_classical - x1)
        return round(fide)

    if lichess_classical > max_lichess:
        # Extrapolate above range using last two points
        if len(mappings) < 2:
            return None
        x1, y1 = mappings[-2]["lichess_classical"], mappings[-2]["fide"]
        x2, y2 = mappings[-1]["lichess_classical"], mappings[-1]["fide"]
        slope = (y2 - y1) / (x2 - x1)
        fide = y2 + slope * (lichess_classical - x2)
        return round(fide)

    # Find surrounding points for interpolation
    for i in range(len(mappings) - 1):
        x1 = mappings[i]["lichess_classical"]
        x2 = mappings[i + 1]["lichess_classical"]

        if x1 <= lichess_classical <= x2:
            y1 = mappings[i]["fide"]
            y2 = mappings[i + 1]["fide"]

            # Handle exact match
            if x1 == x2:
                return y1

            # Linear interpolation
            t = (lichess_classical - x1) / (x2 - x1)
            fide = y1 + t * (y2 - y1)
            return round(fide)

    return None

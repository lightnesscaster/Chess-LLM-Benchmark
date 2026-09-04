#!/usr/bin/env python3
"""
Fetch model pricing from OpenRouter API and save to config file.

Usage:
    python scripts/fetch_pricing.py
"""

import json
import sys
from pathlib import Path

import requests


def fetch_pricing():
    """Fetch pricing from OpenRouter API."""
    print("Fetching model pricing from OpenRouter...")

    response = requests.get("https://openrouter.ai/api/v1/models", timeout=30)
    response.raise_for_status()

    data = response.json()
    models = data.get("data", [])

    pricing = {}
    for model in models:
        model_id = model.get("id")
        model_pricing = model.get("pricing", {})

        if not model_id:
            continue

        # Extract relevant pricing fields (cost per token)
        prompt_cost = model_pricing.get("prompt", "0")
        completion_cost = model_pricing.get("completion", "0")

        # Convert to float (API returns strings)
        try:
            prompt_cost = float(prompt_cost)
            completion_cost = float(completion_cost)
        except (ValueError, TypeError):
            continue

        # Include all models (even free ones with zero cost)
        pricing[model_id] = {
            "prompt": prompt_cost,
            "completion": completion_cost,
        }

    return pricing


def main():
    output_path = Path(__file__).parent.parent / "config" / "pricing.json"

    try:
        pricing = fetch_pricing()
    except requests.RequestException as e:
        print(f"Error fetching pricing: {e}", file=sys.stderr)
        return 1

    # Save to config file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(pricing, f, indent=2, sort_keys=True)

    print(f"Saved pricing for {len(pricing)} models to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Debug script to compare cost chart output with and without annotation."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from web.cost_chart import create_cost_chart, PROVIDER_COLORS, DEFAULT_COLOR, get_provider, is_reasoning_model
from rating.rating_store import RatingStore
from rating.leaderboard import Leaderboard
from game.pgn_logger import PGNLogger
from game.stats_collector import StatsCollector
import yaml

DATA_DIR = Path(__file__).parent / "data"
CONFIG_PATH = Path(__file__).parent / "config" / "benchmark.yaml"
RATINGS_PATH = DATA_DIR / "ratings.json"


def get_leaderboard_data():
    """Get leaderboard data."""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    anchors = {
        engine["player_id"]: engine["rating"]
        for engine in config.get("engines", [])
    }
    anchor_ids = set(anchors.keys())
    rating_store = RatingStore(path=str(RATINGS_PATH), anchor_ids=anchor_ids)

    for anchor_id, rating in anchors.items():
        if not rating_store.has_player(anchor_id):
            rating_store.set_anchor(anchor_id, rating, auto_save=False)

    pgn_logger = PGNLogger()
    stats_collector = StatsCollector()
    stats_collector.add_results(pgn_logger.load_all_results())

    leaderboard = Leaderboard(rating_store, stats_collector)
    return leaderboard.get_leaderboard(min_games=1, sort_by="rating")


def main():
    leaderboard_data = get_leaderboard_data()

    # Create chart WITH annotation (working version)
    fig_with = create_cost_chart(leaderboard_data)

    # Get the figure as JSON
    json_with = fig_with.to_json()
    data_with = json.loads(json_with)

    print("=" * 80)
    print("CHART WITH ANNOTATION")
    print("=" * 80)
    print(f"Number of traces: {len(data_with['data'])}")
    print(f"Layout keys: {list(data_with['layout'].keys())}")
    print(f"Has annotations: {'annotations' in data_with['layout']}")
    if 'annotations' in data_with['layout']:
        print(f"Number of annotations: {len(data_with['layout']['annotations'])}")
        for i, ann in enumerate(data_with['layout']['annotations']):
            print(f"  Annotation {i}: x={ann.get('x')}, y={ann.get('y')}, text={ann.get('text', '')[:50]}")

    print(f"\nXaxis type: {data_with['layout'].get('xaxis', {}).get('type')}")
    print(f"Paper bgcolor: {data_with['layout'].get('paper_bgcolor')}")
    print(f"Plot bgcolor: {data_with['layout'].get('plot_bgcolor')}")

    # Save full JSON for comparison
    with open("/tmp/cost_chart_with_annotation.json", "w") as f:
        json.dump(data_with, f, indent=2)
    print(f"\nFull JSON saved to /tmp/cost_chart_with_annotation.json")

    # Now create a version WITHOUT the annotation by modifying the figure
    fig_without = create_cost_chart(leaderboard_data)
    # Remove all annotations
    fig_without.layout.annotations = []

    json_without = fig_without.to_json()
    data_without = json.loads(json_without)

    print("\n" + "=" * 80)
    print("CHART WITHOUT ANNOTATION (annotations cleared)")
    print("=" * 80)
    print(f"Number of traces: {len(data_without['data'])}")
    print(f"Layout keys: {list(data_without['layout'].keys())}")
    print(f"Has annotations: {'annotations' in data_without['layout']}")
    if 'annotations' in data_without['layout']:
        print(f"Annotations array: {data_without['layout']['annotations']}")

    print(f"\nXaxis type: {data_without['layout'].get('xaxis', {}).get('type')}")
    print(f"Paper bgcolor: {data_without['layout'].get('paper_bgcolor')}")
    print(f"Plot bgcolor: {data_without['layout'].get('plot_bgcolor')}")

    with open("/tmp/cost_chart_without_annotation.json", "w") as f:
        json.dump(data_without, f, indent=2)
    print(f"\nFull JSON saved to /tmp/cost_chart_without_annotation.json")

    # Compare the layouts
    print("\n" + "=" * 80)
    print("LAYOUT COMPARISON")
    print("=" * 80)

    for key in set(data_with['layout'].keys()) | set(data_without['layout'].keys()):
        val_with = data_with['layout'].get(key)
        val_without = data_without['layout'].get(key)
        if val_with != val_without:
            print(f"\nDifference in '{key}':")
            print(f"  With annotation: {val_with}")
            print(f"  Without annotation: {val_without}")


def compare_html():
    """Compare the actual HTML output."""
    leaderboard_data = get_leaderboard_data()

    # Create chart WITH annotation
    fig_with = create_cost_chart(leaderboard_data)
    html_with = fig_with.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id="cost-chart",
        config={
            "displayModeBar": "hover",
            "displaylogo": False,
            "modeBarButtons": [["pan2d", "zoomIn2d", "zoomOut2d", "resetScale2d"]],
            "scrollZoom": False,
            "responsive": True,
        },
    )

    # Create chart WITHOUT annotation
    fig_without = create_cost_chart(leaderboard_data)
    fig_without.layout.annotations = []
    html_without = fig_without.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id="cost-chart",
        config={
            "displayModeBar": "hover",
            "displaylogo": False,
            "modeBarButtons": [["pan2d", "zoomIn2d", "zoomOut2d", "resetScale2d"]],
            "scrollZoom": False,
            "responsive": True,
        },
    )

    print("=" * 80)
    print("HTML COMPARISON")
    print("=" * 80)
    print(f"HTML with annotation length: {len(html_with)}")
    print(f"HTML without annotation length: {len(html_without)}")
    print(f"Difference: {len(html_with) - len(html_without)} chars")

    # Save both for manual inspection
    with open("/tmp/cost_chart_with.html", "w") as f:
        f.write(html_with)
    with open("/tmp/cost_chart_without.html", "w") as f:
        f.write(html_without)

    print("\nHTML saved to /tmp/cost_chart_with.html and /tmp/cost_chart_without.html")

    # Check for any structural differences
    # Extract the Plotly.newPlot call from both
    import re
    plot_call_with = re.search(r'Plotly\.newPlot\([^)]+\)', html_with)
    plot_call_without = re.search(r'Plotly\.newPlot\([^)]+\)', html_without)

    if plot_call_with and plot_call_without:
        print(f"\nPlotly.newPlot call (with): {plot_call_with.group()[:100]}...")
        print(f"Plotly.newPlot call (without): {plot_call_without.group()[:100]}...")


if __name__ == "__main__":
    main()
    print("\n\n")
    compare_html()

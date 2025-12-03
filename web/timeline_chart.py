"""
Timeline chart generation for LLM Chess Benchmark.

Creates an interactive Plotly visualization showing the progression
of top chess-playing LLMs over time.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

# Provider colors - consistent branding
PROVIDER_COLORS = {
    "anthropic": "#D97706",  # Orange/amber
    "openai": "#10B981",     # Green
    "google": "#3B82F6",     # Blue
    "deepseek": "#8B5CF6",   # Purple
    "meta-llama": "#EC4899", # Pink
    "x-ai": "#6B7280",       # Gray
    "mistralai": "#F97316",  # Orange
    "qwen": "#14B8A6",       # Teal
    "moonshotai": "#EAB308", # Yellow
    "z-ai": "#EF4444",       # Red
}

DEFAULT_COLOR = "#9CA3AF"  # Gray for unknown providers


def get_provider(model_id: str) -> str:
    """Extract provider from model_id (e.g., 'anthropic/claude-3.5-sonnet' -> 'anthropic')."""
    if "/" in model_id:
        return model_id.split("/")[0]
    return "unknown"


def is_reasoning_model(player_id: str) -> bool:
    """Check if a model is a reasoning model based on player_id patterns."""
    reasoning_indicators = [
        "(thinking)",
        "(high)",
        "(medium)",
        "-r1",
        "o3",
        "o4-mini",
    ]
    player_lower = player_id.lower()
    return any(indicator in player_lower for indicator in reasoning_indicators)


def create_timeline_chart(leaderboard_data: list[dict[str, Any]]) -> go.Figure:
    """
    Create an interactive timeline chart showing LLM chess rating progression.

    Args:
        leaderboard_data: List of leaderboard entries with rating and publish date info

    Returns:
        Plotly Figure object
    """
    # Filter to models with publish dates (exclude anchors)
    models_with_dates = [
        entry for entry in leaderboard_data
        if entry.get("publish_timestamp") and not entry.get("is_anchor")
    ]

    if not models_with_dates:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No models with release dates available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#9CA3AF")
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#16213e",
        )
        return fig

    # Sort by release date, then by rating (ascending) for same-day releases
    # This ensures lower-rated models are processed first, so higher-rated ones
    # become the "champion" at that date point
    models_with_dates.sort(key=lambda x: (x["publish_timestamp"], x["rating"]))

    # Load publish dates for model_id lookup
    publish_dates_path = Path(__file__).parent.parent / "data" / "model_publish_dates.json"
    model_id_lookup = {}
    try:
        with open(publish_dates_path) as f:
            publish_data = json.load(f)
            for player_id, info in publish_data.items():
                model_id_lookup[player_id] = info.get("model_id", "")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Prepare data for scatter plot
    dates = []
    ratings = []
    player_ids = []
    providers = []
    colors = []
    symbols = []
    hover_texts = []

    for entry in models_with_dates:
        timestamp = entry["publish_timestamp"]
        date = datetime.fromtimestamp(timestamp)
        dates.append(date)
        ratings.append(entry["rating"])
        player_ids.append(entry["player_id"])

        # Get provider from model_id
        model_id = model_id_lookup.get(entry["player_id"], "")
        provider = get_provider(model_id)
        providers.append(provider)
        colors.append(PROVIDER_COLORS.get(provider, DEFAULT_COLOR))

        # Symbol: diamond for reasoning, circle for non-reasoning
        is_reasoning = is_reasoning_model(entry["player_id"])
        symbols.append("diamond" if is_reasoning else "circle")

        # Hover text
        reasoning_label = " (reasoning)" if is_reasoning else ""
        hover_texts.append(
            f"<b>{entry['player_id']}</b>{reasoning_label}<br>"
            f"Rating: {entry['rating']}<br>"
            f"Released: {entry.get('publish_date', 'Unknown')}<br>"
            f"Provider: {provider.title()}<br>"
            f"Games: {entry.get('games_played', 'N/A')}"
        )

    # Calculate frontier (champion) line
    frontier_dates = []
    frontier_ratings = []
    frontier_players = []
    current_best = float("-inf")

    for i, (date, rating, player_id) in enumerate(zip(dates, ratings, player_ids)):
        if rating > current_best:
            # Add horizontal segment from previous point to current date at old rating
            if frontier_dates:
                frontier_dates.append(date)
                frontier_ratings.append(current_best)
                frontier_players.append(frontier_players[-1] if frontier_players else "")

            # Add new champion point
            frontier_dates.append(date)
            frontier_ratings.append(rating)
            frontier_players.append(player_id)
            current_best = rating

    # Extend frontier to present day
    if frontier_dates:
        today = datetime.now()
        frontier_dates.append(today)
        frontier_ratings.append(current_best)
        frontier_players.append(frontier_players[-1])

    # Create figure
    fig = go.Figure()

    # Add frontier line first (so it's behind scatter points)
    fig.add_trace(go.Scatter(
        x=frontier_dates,
        y=frontier_ratings,
        mode="lines",
        name="Champion Line",
        line=dict(color="#e94560", width=4, shape="hv"),  # step line
        hoverinfo="skip",
    ))

    # Add scatter points grouped by provider for legend
    providers_seen = set()
    for provider in PROVIDER_COLORS.keys():
        mask = [p == provider for p in providers]
        if not any(mask):
            continue
        providers_seen.add(provider)

        provider_dates = [d for d, m in zip(dates, mask) if m]
        provider_ratings = [r for r, m in zip(ratings, mask) if m]
        provider_symbols = [s for s, m in zip(symbols, mask) if m]
        provider_hovers = [h for h, m in zip(hover_texts, mask) if m]

        fig.add_trace(go.Scatter(
            x=provider_dates,
            y=provider_ratings,
            mode="markers",
            name=provider.replace("-", " ").title(),
            marker=dict(
                size=16,
                color=PROVIDER_COLORS[provider],
                opacity=0.85,
                symbol=provider_symbols,
                line=dict(width=1.5, color="rgba(255,255,255,0.8)"),
            ),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=provider_hovers,
        ))

    # Add any remaining providers not in PROVIDER_COLORS
    other_mask = [p not in providers_seen for p in providers]
    if any(other_mask):
        other_dates = [d for d, m in zip(dates, other_mask) if m]
        other_ratings = [r for r, m in zip(ratings, other_mask) if m]
        other_symbols = [s for s, m in zip(symbols, other_mask) if m]
        other_hovers = [h for h, m in zip(hover_texts, other_mask) if m]

        fig.add_trace(go.Scatter(
            x=other_dates,
            y=other_ratings,
            mode="markers",
            name="Other",
            marker=dict(
                size=16,
                color=DEFAULT_COLOR,
                symbol=other_symbols,
                line=dict(width=1.5, color="white"),
            ),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=other_hovers,
        ))

    # Highlight champion models with larger markers
    champion_set = set(frontier_players)
    champion_dates = [d for d, p in zip(dates, player_ids) if p in champion_set]
    champion_ratings = [r for r, p in zip(ratings, player_ids) if p in champion_set]
    champion_hovers = [h for h, p in zip(hover_texts, player_ids) if p in champion_set]

    if champion_dates:
        fig.add_trace(go.Scatter(
            x=champion_dates,
            y=champion_ratings,
            mode="markers",
            name="Champions",
            marker=dict(
                size=26,
                color="rgba(233, 69, 96, 0.25)",
                symbol="circle",
                line=dict(width=3, color="#e94560"),
            ),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=champion_hovers,
            showlegend=False,
        ))

    # Update layout - clean, minimal design
    fig.update_layout(
        xaxis=dict(
            title=dict(text="Release Date", font=dict(size=16, color="#a0a0a0"), standoff=15),
            gridcolor="rgba(74, 90, 122, 0.3)",
            griddash="dot",
            showgrid=True,
            tickformat="%b %Y",
            tickfont=dict(size=14),
        ),
        yaxis=dict(
            title=dict(text="Lichess Classical Rating", font=dict(size=16, color="#a0a0a0"), standoff=10),
            gridcolor="rgba(74, 90, 122, 0.3)",
            griddash="dot",
            showgrid=True,
            tickfont=dict(size=14),
            zeroline=True,
            zerolinecolor="rgba(74, 90, 122, 0.6)",
            zerolinewidth=1,
        ),
        template="plotly_dark",
        paper_bgcolor="#16213e",
        plot_bgcolor="#16213e",
        showlegend=False,  # Hide legend - we use HTML legend below
        hovermode="closest",
        margin=dict(t=30, b=60, l=80, r=30),
    )

    # Add annotation for current champion (top-rated model)
    if models_with_dates:
        top_model = max(models_with_dates, key=lambda x: x["rating"])
        top_date = datetime.fromtimestamp(top_model["publish_timestamp"])
        fig.add_annotation(
            x=top_date,
            y=top_model["rating"],
            text=f"<b>{top_model['player_id']}</b>",
            showarrow=True,
            arrowhead=0,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#e94560",
            ax=40,
            ay=-40,
            font=dict(size=13, color="#eaeaea"),
            bgcolor="rgba(22, 33, 62, 0.8)",
            bordercolor="#e94560",
            borderwidth=1,
            borderpad=6,
        )

    return fig


def export_timeline_png(fig: go.Figure, output_path: str | Path) -> None:
    """
    Export the timeline chart as a PNG image.

    Args:
        fig: Plotly Figure object
        output_path: Path to save the PNG file

    Raises:
        RuntimeError: If PNG export fails (e.g., kaleido not installed)
    """
    try:
        fig.write_image(
            str(output_path),
            width=1200,
            height=700,
            scale=2,  # 2x resolution for crisp images
        )
    except Exception as e:
        raise RuntimeError(f"Failed to export PNG: {e}. Ensure kaleido is installed.") from e


def get_timeline_html(leaderboard_data: list[dict[str, Any]]) -> str:
    """
    Get the timeline chart as an HTML div string.

    Args:
        leaderboard_data: List of leaderboard entries

    Returns:
        HTML string containing the chart
    """
    fig = create_timeline_chart(leaderboard_data)
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,  # We'll include it via CDN
        div_id="timeline-chart",
        config={
            "displayModeBar": "hover",  # Only show on hover
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
        },
    )

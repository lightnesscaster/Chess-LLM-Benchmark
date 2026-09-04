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

from utils import is_reasoning_model

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


def create_timeline_chart(leaderboard_data: list[dict[str, Any]]) -> go.Figure:
    """
    Create an interactive timeline chart showing LLM chess rating progression.

    Args:
        leaderboard_data: List of leaderboard entries with rating and publish date info

    Returns:
        Plotly Figure object
    """
    # Load publish dates for model_id lookup (needed for provider extraction)
    publish_dates_path = Path(__file__).parent.parent / "data" / "model_publish_dates.json"
    model_id_lookup = {}
    try:
        with open(publish_dates_path) as f:
            publish_data = json.load(f)
            for player_id, info in publish_data.items():
                model_id_lookup[player_id] = info.get("model_id", "")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Filter to models with publish dates (exclude anchors), sufficient confidence (RD <= 150),
    # and rating >= -500
    models_with_dates = [
        entry for entry in leaderboard_data
        if entry.get("publish_timestamp")
        and not entry.get("is_anchor")
        and entry.get("rating_deviation", 350) <= 150
        and entry.get("rating", 0) >= -500
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

    # Sort by release date (ascending) for filtering
    models_with_dates.sort(key=lambda x: x["publish_timestamp"])

    # Filter to only models that were the top-rated from their lab in their category
    # (reasoning vs non-reasoning) at the time of release
    def was_top_in_category_at_release(entry: dict, all_models: list) -> bool:
        """Check if model was the best from its lab in its category when released."""
        model_id = model_id_lookup.get(entry["player_id"], "")
        provider = get_provider(model_id)
        is_reasoning = is_reasoning_model(entry["player_id"])
        release_time = entry["publish_timestamp"]
        rating = entry["rating"]

        # Check all earlier models from same provider in same category
        for other in all_models:
            if other["player_id"] == entry["player_id"]:
                continue
            other_model_id = model_id_lookup.get(other["player_id"], "")
            other_provider = get_provider(other_model_id)
            other_is_reasoning = is_reasoning_model(other["player_id"])
            other_release = other["publish_timestamp"]

            # Same provider, same category, released before this model
            if (other_provider == provider
                and other_is_reasoning == is_reasoning
                and other_release < release_time):
                # If earlier model has higher rating, this model wasn't top at release
                if other["rating"] > rating:
                    return False

        return True

    # Apply the top-in-category filter
    models_with_dates = [
        entry for entry in models_with_dates
        if was_top_in_category_at_release(entry, models_with_dates)
    ]

    if not models_with_dates:
        fig = go.Figure()
        fig.add_annotation(
            text="No qualifying models found",
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

    # Re-sort by release date, then by rating (ascending) for same-day releases
    # This ensures lower-rated models are processed first, so higher-rated ones
    # become the "champion" at that date point
    models_with_dates.sort(key=lambda x: (x["publish_timestamp"], x["rating"]))

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

        # Hover card
        reasoning_label = "Reasoning" if is_reasoning else "Non-reasoning"
        release_fmt = date.strftime("%b %Y")
        hover_texts.append(
            f"<b style='font-size:14px'>{entry['player_id']}</b><br>"
            f"<span style='color:#e94560;font-size:13px'><b>{entry['rating']}</b></span>"
            f" <span style='color:#a0a0a0'>rating</span><br>"
            f"<span style='color:#a0a0a0'>{provider.title()} · {reasoning_label}</span><br>"
            f"<span style='color:#a0a0a0'>Released {release_fmt} · "
            f"{entry.get('games_played', 'N/A')} games</span>"
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
    if frontier_dates and frontier_players:
        today = datetime.now()
        frontier_dates.append(today)
        frontier_ratings.append(current_best)
        frontier_players.append(frontier_players[-1])

    # Create figure
    fig = go.Figure()

    # Soft translucent fill under the champion line — gives the frontier
    # visual weight without drawing attention away from the points.
    if frontier_dates:
        fig.add_trace(go.Scatter(
            x=frontier_dates,
            y=frontier_ratings,
            mode="lines",
            line=dict(color="rgba(233, 69, 96, 0)", shape="hv"),
            fill="tozeroy",
            fillcolor="rgba(233, 69, 96, 0.08)",
            hoverinfo="skip",
            showlegend=False,
        ))

    # Champion (frontier) line on top of the fill
    fig.add_trace(go.Scatter(
        x=frontier_dates,
        y=frontier_ratings,
        mode="lines",
        name="Champion Line",
        line=dict(color="#e94560", width=3.5, shape="hv"),
        hoverinfo="skip",
    ))

    # Champion halo — drawn BEFORE provider markers so the halo sits behind
    # the colored dot, leaving the provider color visible at the center.
    champion_set = set(frontier_players)
    champion_dates = [d for d, p in zip(dates, player_ids) if p in champion_set]
    champion_ratings = [r for r, p in zip(ratings, player_ids) if p in champion_set]

    if champion_dates:
        fig.add_trace(go.Scatter(
            x=champion_dates,
            y=champion_ratings,
            mode="markers",
            marker=dict(
                size=32,
                color="rgba(233, 69, 96, 0.18)",
                symbol="circle",
                line=dict(width=0),
            ),
            hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=champion_dates,
            y=champion_ratings,
            mode="markers",
            marker=dict(
                size=24,
                color="rgba(0,0,0,0)",
                symbol="circle",
                line=dict(width=2.5, color="#e94560"),
            ),
            hoverinfo="skip",
            showlegend=False,
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
                size=14,
                color=PROVIDER_COLORS[provider],
                opacity=0.9,
                symbol=provider_symbols,
                line=dict(width=1.5, color="rgba(22, 33, 62, 0.9)"),
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
                size=14,
                color=DEFAULT_COLOR,
                opacity=0.9,
                symbol=other_symbols,
                line=dict(width=1.5, color="rgba(22, 33, 62, 0.9)"),
            ),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=other_hovers,
        ))

    # Calculate x-axis range. Start at GPT-4 release; extend the end past
    # the latest release (or today) so recent models aren't clipped.
    x_start = datetime(2023, 3, 1)
    now = datetime.now()
    latest_release = max(
        (datetime.fromtimestamp(e["publish_timestamp"]) for e in models_with_dates),
        default=now,
    )
    x_end = max(now, latest_release) + (now - datetime(2023, 3, 1)) * 0.03

    # Compute y-axis range with a little headroom above top rating.
    max_rating = max(ratings) if ratings else 2000
    y_top = max(2200, int(max_rating * 1.10 / 100) * 100 + 100)
    y_bottom = -500

    # Skill tiers anchored to Lichess classical. Labels sit at the right
    # edge; low-opacity horizontal lines separate the tiers.
    tier_breaks = [
        (500,  "Novice"),
        (1000, "Beginner"),
        (1500, "Intermediate"),
        (1800, "Advanced"),
        (2100, "Expert"),
    ]
    tier_shapes = []
    tier_annotations = []
    for y_val, label in tier_breaks:
        if y_val < y_bottom or y_val > y_top:
            continue
        tier_shapes.append(dict(
            type="line",
            xref="x", yref="y",
            x0=x_start, x1=x_end,
            y0=y_val, y1=y_val,
            line=dict(color="rgba(160, 160, 160, 0.15)", width=1, dash="dot"),
            layer="below",
        ))
        tier_annotations.append(dict(
            x=x_end, y=y_val,
            xref="x", yref="y",
            xanchor="right", yanchor="bottom",
            text=f"<i>{label}</i>",
            showarrow=False,
            font=dict(size=11, color="rgba(200, 200, 200, 0.45)"),
            xshift=-6, yshift=2,
        ))

    # Clean, minimal layout
    fig.update_layout(
        xaxis=dict(
            title=dict(text="Release Date", font=dict(size=14, color="#a0a0a0"), standoff=12),
            gridcolor="rgba(74, 90, 122, 0.2)",
            griddash="dot",
            showgrid=True,
            tickformat="%b %Y",
            tickfont=dict(size=13, color="#c8c8d0"),
            range=[x_start, x_end],
        ),
        yaxis=dict(
            title=dict(text="Rating (Lichess classical)", font=dict(size=14, color="#a0a0a0"), standoff=8),
            gridcolor="rgba(74, 90, 122, 0.2)",
            griddash="dot",
            showgrid=True,
            tickfont=dict(size=13, color="#c8c8d0"),
            zeroline=True,
            zerolinecolor="rgba(74, 90, 122, 0.5)",
            zerolinewidth=1,
            range=[y_bottom, y_top],
        ),
        template="plotly_dark",
        paper_bgcolor="#16213e",
        plot_bgcolor="#16213e",
        showlegend=False,
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="rgba(22, 33, 62, 0.95)",
            bordercolor="#e94560",
            font=dict(size=12, color="#eaeaea"),
        ),
        dragmode="pan",
        margin=dict(t=30, b=55, l=70, r=25),
        autosize=True,
        shapes=tier_shapes,
        annotations=tier_annotations,
    )

    # Annotate the current top model. Auto-flip label to the LEFT of the
    # marker when the point is close to the right edge, so it never clips.
    if models_with_dates:
        top_model = max(models_with_dates, key=lambda x: x["rating"])
        top_date = datetime.fromtimestamp(top_model["publish_timestamp"])
        x_span = (x_end - x_start).total_seconds()
        right_fraction = (top_date - x_start).total_seconds() / x_span if x_span > 0 else 0.5
        # If marker is in the right third, anchor label to its left; else right.
        anchor_left = right_fraction > 0.65
        ax = -80 if anchor_left else 80
        ay = -45
        fig.add_annotation(
            x=top_date,
            y=top_model["rating"],
            text=(
                f"<b>{top_model['player_id']}</b><br>"
                f"<span style='color:#e94560'>{top_model['rating']}</span>"
                f" <span style='color:#a0a0a0'>· {top_date.strftime('%b %Y')}</span>"
            ),
            showarrow=True,
            arrowhead=0,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#e94560",
            ax=ax,
            ay=ay,
            font=dict(size=12, color="#eaeaea"),
            bgcolor="rgba(22, 33, 62, 0.95)",
            bordercolor="#e94560",
            borderwidth=1.5,
            borderpad=8,
            align="left",
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
            width=1600,
            height=1300,
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
            "modeBarButtons": [["zoomIn2d", "zoomOut2d", "resetScale2d"]],
            "scrollZoom": False,
            "responsive": True,
        },
    )

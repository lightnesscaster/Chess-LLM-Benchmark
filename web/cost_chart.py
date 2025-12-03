"""
Cost vs Rating chart generation for LLM Chess Benchmark.

Creates an interactive Plotly visualization showing the relationship
between LLM chess rating and cost per game.
"""

import json
import math
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

# Provider colors - consistent branding (same as timeline_chart.py)
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
        "(minimal)",
        "-r1",
        "o3",
        "o4-mini",
        "o1",
        "gemini-3",
        "gemini-2.5-pro",
        "grok-4",
        "-thinking",
    ]
    player_lower = player_id.lower()
    return any(indicator in player_lower for indicator in reasoning_indicators)


def create_cost_chart(leaderboard_data: list[dict[str, Any]]) -> go.Figure:
    """
    Create an interactive scatter chart showing LLM rating vs cost per game.

    Args:
        leaderboard_data: List of leaderboard entries with rating and cost info

    Returns:
        Plotly Figure object
    """
    # Load model_id lookup for provider extraction
    publish_dates_path = Path(__file__).parent.parent / "data" / "model_publish_dates.json"
    model_id_lookup = {}
    try:
        with open(publish_dates_path) as f:
            publish_data = json.load(f)
            for player_id, info in publish_data.items():
                model_id_lookup[player_id] = info.get("model_id", "")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Filter to models with cost data, sufficient confidence (RD <= 80),
    # exclude anchors, and rating >= -500
    models_with_cost = [
        entry for entry in leaderboard_data
        if entry.get("avg_cost_per_game") is not None
        and entry.get("avg_cost_per_game", 0) > 0
        and not entry.get("is_anchor")
        and entry.get("rating_deviation", 350) <= 80
        and entry.get("rating", 0) >= -500
    ]

    if not models_with_cost:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No models with cost data available",
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

    # Prepare data for scatter plot
    costs = []
    ratings = []
    player_ids = []
    providers = []
    colors = []
    symbols = []
    hover_texts = []

    for entry in models_with_cost:
        cost = entry["avg_cost_per_game"]
        costs.append(cost)
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
            f"Cost/Game: ${cost:.4f}<br>"
            f"Provider: {provider.title()}<br>"
            f"Games: {entry.get('games_played', 'N/A')}"
        )

    # Create figure
    fig = go.Figure()

    # Add scatter points grouped by provider (no legend - using HTML legend)
    providers_seen = set()
    for provider in PROVIDER_COLORS.keys():
        mask = [p == provider for p in providers]
        if not any(mask):
            continue
        providers_seen.add(provider)

        provider_costs = [c for c, m in zip(costs, mask) if m]
        provider_ratings = [r for r, m in zip(ratings, mask) if m]
        provider_symbols = [s for s, m in zip(symbols, mask) if m]
        provider_hovers = [h for h, m in zip(hover_texts, mask) if m]

        fig.add_trace(go.Scatter(
            x=provider_costs,
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
            showlegend=False,
        ))

    # Add any remaining providers not in PROVIDER_COLORS
    other_mask = [p not in providers_seen for p in providers]
    if any(other_mask):
        other_costs = [c for c, m in zip(costs, other_mask) if m]
        other_ratings = [r for r, m in zip(ratings, other_mask) if m]
        other_symbols = [s for s, m in zip(symbols, other_mask) if m]
        other_hovers = [h for h, m in zip(hover_texts, other_mask) if m]

        fig.add_trace(go.Scatter(
            x=other_costs,
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
            showlegend=False,
        ))

    # Find the Pareto frontier (best rating for each cost level)
    # Sort by cost ascending
    sorted_models = sorted(zip(costs, ratings, player_ids), key=lambda x: x[0])
    pareto_costs = []
    pareto_ratings = []
    pareto_players = []
    best_rating = float("-inf")

    for cost, rating, player_id in sorted_models:
        if rating > best_rating:
            pareto_costs.append(cost)
            pareto_ratings.append(rating)
            pareto_players.append(player_id)
            best_rating = rating

    # Add Pareto frontier line
    if len(pareto_costs) > 1:
        fig.add_trace(go.Scatter(
            x=pareto_costs,
            y=pareto_ratings,
            mode="lines",
            name="Efficiency Frontier",
            line=dict(color="#e94560", width=3, dash="dot"),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Highlight Pareto-optimal models
    pareto_set = set(pareto_players)
    pareto_x = [c for c, p in zip(costs, player_ids) if p in pareto_set]
    pareto_y = [r for r, p in zip(ratings, player_ids) if p in pareto_set]

    if pareto_x:
        fig.add_trace(go.Scatter(
            x=pareto_x,
            y=pareto_y,
            mode="markers",
            name="Pareto Optimal",
            marker=dict(
                size=26,
                color="rgba(233, 69, 96, 0.25)",
                symbol="circle",
                line=dict(width=3, color="#e94560"),
            ),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Calculate axis ranges for positioning
    min_cost = min(costs)
    max_cost = max(costs)

    # Update layout
    fig.update_layout(
        xaxis=dict(
            title=dict(text="Cost per Game ($)", font=dict(size=16, color="#a0a0a0"), standoff=15),
            gridcolor="rgba(74, 90, 122, 0.3)",
            griddash="dot",
            showgrid=True,
            tickfont=dict(size=12),
            type="log",  # Log scale for cost
            tickvals=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
            ticktext=["$0.0001", "$0.001", "$0.01", "$0.10", "$1", "$10", "$100"],
            minor=dict(showgrid=True, gridcolor="rgba(74, 90, 122, 0.15)"),
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
        dragmode=False,
        margin=dict(t=20, b=55, l=70, r=30),
        autosize=True,
    )

    # Add annotation for best value model (highest rating on Pareto frontier)
    if pareto_players:
        # Find the model with highest rating on the Pareto frontier
        best_idx = pareto_ratings.index(max(pareto_ratings))
        best_cost = pareto_costs[best_idx]
        best_rating_val = pareto_ratings[best_idx]
        best_player = pareto_players[best_idx]

        # Position annotation to the left if model is on the right side of the chart
        # Use log scale midpoint for comparison
        log_midpoint = math.sqrt(min_cost * max_cost)
        if best_cost > log_midpoint:
            ax, ay = -80, -25  # Point left and up
        else:
            ax, ay = 80, -25   # Point right and up

        fig.add_annotation(
            x=best_cost,
            y=best_rating_val,
            text=f"<b>{best_player}</b>",
            showarrow=True,
            arrowhead=0,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#e94560",
            ax=ax,
            ay=ay,
            font=dict(size=12, color="#eaeaea"),
            bgcolor="rgba(22, 33, 62, 0.9)",
            bordercolor="#e94560",
            borderwidth=1,
            borderpad=5,
            xref="x",
            yref="y",
        )

    return fig


def get_cost_chart_html(leaderboard_data: list[dict[str, Any]]) -> str:
    """
    Get the cost chart as an HTML div string.

    Args:
        leaderboard_data: List of leaderboard entries

    Returns:
        HTML string containing the chart
    """
    fig = create_cost_chart(leaderboard_data)
    return fig.to_html(
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

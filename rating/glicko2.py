"""
Glicko-2 Rating System Implementation.

Based on Mark Glickman's Glicko-2 paper:
http://www.glicko.net/glicko/glicko2.pdf
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple
from datetime import datetime, timezone


@dataclass
class PlayerRating:
    """Represents a player's Glicko-2 rating."""
    player_id: str
    rating: float = 1500.0          # μ (mu) - rating
    rating_deviation: float = 350.0  # φ (phi) - rating deviation
    volatility: float = 0.06        # σ (sigma) - volatility
    games_played: int = 0
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "player_id": self.player_id,
            "rating": self.rating,
            "rating_deviation": self.rating_deviation,
            "volatility": self.volatility,
            "games_played": self.games_played,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlayerRating":
        """Create from dictionary."""
        return cls(**data)


class Glicko2System:
    """Implements the Glicko-2 rating system."""

    def __init__(self, tau: float = 1.2):
        """
        Initialize the Glicko-2 system.

        Args:
            tau: System constant (0.3 to 1.2). Higher = faster adaptation.
        """
        self.TAU = tau
        self.EPSILON = 0.000001  # Convergence tolerance
        self.SCALE_FACTOR = 173.7178  # 400 / ln(10)

    def glicko_to_glicko2(self, rating: float, rd: float) -> Tuple[float, float]:
        """Convert Glicko rating/RD to Glicko-2 scale."""
        mu = (rating - 1500) / self.SCALE_FACTOR
        phi = rd / self.SCALE_FACTOR
        return mu, phi

    def glicko2_to_glicko(self, mu: float, phi: float) -> Tuple[float, float]:
        """Convert Glicko-2 scale back to Glicko rating/RD."""
        rating = mu * self.SCALE_FACTOR + 1500
        rd = phi * self.SCALE_FACTOR
        return rating, rd

    def g_function(self, phi: float) -> float:
        """G function for Glicko-2 calculations."""
        return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))

    def e_function(self, mu: float, mu_j: float, phi_j: float) -> float:
        """E function for expected score calculation."""
        g = self.g_function(phi_j)
        exponent = g * (mu_j - mu)

        # Numerical stability for extreme values
        if exponent > 35:
            return 0.0
        elif exponent < -35:
            return 1.0
        else:
            return 1.0 / (1.0 + math.exp(exponent))

    def calculate_variance(self, mu: float, opponents: List[Tuple[float, float]]) -> float:
        """Calculate variance (v) for Glicko-2."""
        v = 0.0
        for mu_j, phi_j in opponents:
            g_phi_j = self.g_function(phi_j)
            e_val = self.e_function(mu, mu_j, phi_j)
            term = g_phi_j * g_phi_j * e_val * (1.0 - e_val)
            if math.isfinite(term):
                v += term

        v = max(v, 1e-6)  # Prevent division by zero
        return 1.0 / v

    def calculate_delta(
        self,
        mu: float,
        opponents: List[Tuple[float, float]],
        scores: List[float]
    ) -> float:
        """Calculate delta (Δ) for Glicko-2."""
        delta = 0.0
        for i, (mu_j, phi_j) in enumerate(opponents):
            g_phi_j = self.g_function(phi_j)
            e_val = self.e_function(mu, mu_j, phi_j)
            term = g_phi_j * (scores[i] - e_val)
            if math.isfinite(term):
                delta += term
        return delta

    def calculate_new_volatility(
        self,
        sigma: float,
        phi: float,
        v: float,
        delta: float
    ) -> float:
        """Calculate new volatility using Illinois algorithm."""
        phi_squared = phi * phi
        delta_squared = delta * delta
        a = math.log(sigma * sigma)

        def f(x):
            ex = math.exp(x)
            numerator = ex * (delta_squared - phi_squared - v - ex)
            denominator = 2 * (phi_squared + v + ex) ** 2
            return numerator / denominator - (x - a) / (self.TAU ** 2)

        # Find bounds
        A = a
        if delta_squared > phi_squared + v:
            B = math.log(delta_squared - phi_squared - v)
        else:
            k = 1
            while f(a - k * self.TAU) < 0:
                k += 1
                if k > 100:  # Safety limit
                    break
            B = a - k * self.TAU

        # Illinois algorithm
        fA = f(A)
        fB = f(B)

        iterations = 0
        while abs(B - A) > self.EPSILON and iterations < 100:
            C = A + (A - B) * fA / (fB - fA)
            fC = f(C)

            if fC * fB < 0:
                A = B
                fA = fB
            else:
                fA = fA / 2

            B = C
            fB = fC
            iterations += 1

        return math.exp(A / 2)

    def update_rating(
        self,
        player: PlayerRating,
        opponents: List[PlayerRating],
        scores: List[float],
    ) -> PlayerRating:
        """
        Update a player's rating based on game results.

        Args:
            player: The player to update
            opponents: List of opponent PlayerRating objects
            scores: List of scores (1.0 = win, 0.5 = draw, 0.0 = loss)

        Returns:
            New PlayerRating with updated values
        """
        if not opponents or not scores or len(opponents) != len(scores):
            return player

        # Convert to Glicko-2 scale
        mu, phi = self.glicko_to_glicko2(player.rating, player.rating_deviation)
        sigma = player.volatility

        # Convert opponent ratings
        opponent_params = []
        for opp in opponents:
            opp_mu, opp_phi = self.glicko_to_glicko2(opp.rating, opp.rating_deviation)
            opponent_params.append((opp_mu, opp_phi))

        # Step 3: Compute variance
        v = self.calculate_variance(mu, opponent_params)

        # Step 4: Compute delta
        raw_sum = self.calculate_delta(mu, opponent_params, scores)
        delta = v * raw_sum

        # Step 5: Compute new volatility
        new_sigma = self.calculate_new_volatility(sigma, phi, v, delta)

        # Step 6: Update rating deviation
        phi_star = math.sqrt(phi * phi + new_sigma * new_sigma)
        new_phi = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)

        # Step 7: Update rating
        new_mu = mu + new_phi * new_phi * raw_sum

        # Convert back to Glicko scale
        new_rating, new_rd = self.glicko2_to_glicko(new_mu, new_phi)

        # Numerical stability checks
        if not math.isfinite(new_rating) or abs(new_rating) > 10000:
            new_rating = player.rating
        if not math.isfinite(new_rd) or new_rd < 30 or new_rd > 500:
            new_rd = player.rating_deviation
        if not math.isfinite(new_sigma):
            new_sigma = player.volatility

        return PlayerRating(
            player_id=player.player_id,
            rating=new_rating,
            rating_deviation=new_rd,
            volatility=new_sigma,
            games_played=player.games_played + len(opponents),
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

    def expected_score(self, player: PlayerRating, opponent: PlayerRating) -> float:
        """
        Calculate expected score for player against opponent.

        Args:
            player: The player
            opponent: The opponent

        Returns:
            Expected score (0.0 to 1.0)
        """
        mu, phi = self.glicko_to_glicko2(player.rating, player.rating_deviation)
        mu_j, phi_j = self.glicko_to_glicko2(opponent.rating, opponent.rating_deviation)
        return self.e_function(mu, mu_j, phi_j)

    def win_probability(self, player: PlayerRating, opponent: PlayerRating) -> float:
        """
        Calculate probability of player beating opponent.

        This is an approximation based on rating difference.

        Args:
            player: The player
            opponent: The opponent

        Returns:
            Win probability (0.0 to 1.0)
        """
        # Use expected score as approximation
        return self.expected_score(player, opponent)

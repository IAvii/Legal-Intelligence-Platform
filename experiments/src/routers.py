"""
experiments/src/routers.py
Phase 6: Router implementations and replay evaluation.

Routers (per EXPERIMENT_PLAN §6.2):
  - always_large / always_small / always_mid
  - random (averaged over 20 fixed seeds)
  - profile_mean(lambda) / profile_lcb(lambda)
  - cascade (FrugalGPT-style: try small, escalate to large on uncertainty)
  - oracle_skill (upper bound: best model per skill on test labels)

All routers share the same interface:
    router.select(skill: str, context: dict) -> str   (model alias)

Evaluation is by REPLAY: look up stored outcome of selected model.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────

class Router(ABC):
    name: str

    @abstractmethod
    def select(self, skill: str, row: dict | None = None) -> str:
        """Return model alias to use for this (skill, row) pair."""

    def __repr__(self) -> str:
        return self.name


# ─────────────────────────────────────────────────────────────────────────────
# Fixed routers
# ─────────────────────────────────────────────────────────────────────────────

class AlwaysRouter(Router):
    def __init__(self, alias: str) -> None:
        self.alias = alias
        self.name = f"always_{alias}"

    def select(self, skill: str, row: dict | None = None) -> str:
        return self.alias


# ─────────────────────────────────────────────────────────────────────────────
# Random router
# ─────────────────────────────────────────────────────────────────────────────

class RandomRouter(Router):
    def __init__(self, aliases: list[str], seed: int = 0) -> None:
        self.aliases = aliases
        self.rng = random.Random(seed)
        self.name = f"random_seed{seed}"

    def select(self, skill: str, row: dict | None = None) -> str:
        return self.rng.choice(self.aliases)


# ─────────────────────────────────────────────────────────────────────────────
# Profile-based routers
# ─────────────────────────────────────────────────────────────────────────────

def _cost_norm(profiles: dict, aliases: list[str]) -> dict[str, float]:
    """Normalize cost: cost_norm(m) = mean_cost(m) / max_mean_cost."""
    mean_costs: dict[str, float] = {}
    for alias in aliases:
        skill_data = profiles.get(alias, {})
        costs = [p["cost_per_call"] for p in skill_data.values() if p.get("cost_per_call") is not None]
        mean_costs[alias] = sum(costs) / len(costs) if costs else 0.0
    max_cost = max(mean_costs.values()) if mean_costs else 1.0
    return {alias: c / max_cost if max_cost > 0 else 0.0 for alias, c in mean_costs.items()}


class ProfileRouter(Router):
    """
    argmax_m  competence(m, s) - lambda * cost_norm(m)
    competence = competence_mean or competence_lcb depending on use_lcb.
    """

    def __init__(
        self,
        profiles: dict,
        aliases: list[str],
        lam: float,
        use_lcb: bool = False,
    ) -> None:
        self.profiles = profiles
        self.aliases = aliases
        self.lam = lam
        self.use_lcb = use_lcb
        comp_key = "competence_lcb" if use_lcb else "competence_mean"
        tag = "lcb" if use_lcb else "mean"
        self.name = f"profile_{tag}_lam{lam}"
        self._comp_key = comp_key
        self._cost_norms = _cost_norm(profiles, aliases)

    def select(self, skill: str, row: dict | None = None) -> str:
        best_alias = self.aliases[0]
        best_score = float("-inf")
        for alias in self.aliases:
            p = self.profiles.get(alias, {}).get(skill, {})
            comp = p.get(self._comp_key, 0.5)  # default to 0.5 if no profile
            cost = self._cost_norms.get(alias, 0.0)
            score = comp - self.lam * cost
            if score > best_score:
                best_score = score
                best_alias = alias
        return best_alias


# ─────────────────────────────────────────────────────────────────────────────
# Cascade router (FrugalGPT-style)
# ─────────────────────────────────────────────────────────────────────────────

class CascadeRouter(Router):
    """
    Try small model first. Escalate to large if:
      - confidence != "high"  OR
      - parse_ok == False     OR
      - present=True but quote_found_in_context=False (hallucinated evidence)
    
    Evaluation note: cost = small_cost + large_cost when escalated.
    This router requires the stored small-model row to decide escalation.
    The evaluate module passes the row for the first-attempt model.
    """

    def __init__(self, small_alias: str, large_alias: str) -> None:
        self.small = small_alias
        self.large = large_alias
        self.name = "cascade"

    def select(self, skill: str, row: dict | None = None) -> str:
        # First call is always small; evaluate() handles cascading logic
        return self.small

    def should_escalate(self, small_row: dict) -> bool:
        if not small_row.get("parse_ok", False):
            return True
        if small_row.get("confidence", "low") != "high":
            return True
        if small_row.get("pred_present") and not small_row.get("quote_found_in_context", False):
            return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Oracle router (upper bound — uses test labels; must be labeled clearly)
# ─────────────────────────────────────────────────────────────────────────────

class OracleSkillRouter(Router):
    """
    Picks the best model per skill on the TEST set.
    UPPER BOUND ONLY — not achievable in practice. Label clearly in plots.
    """

    def __init__(self, best_model_per_skill: dict[str, str]) -> None:
        self.best = best_model_per_skill
        self.name = "oracle_skill"

    def select(self, skill: str, row: dict | None = None) -> str:
        return self.best.get(skill, list(self.best.values())[0])


# ─────────────────────────────────────────────────────────────────────────────
# Factory: build all routers for evaluation
# ─────────────────────────────────────────────────────────────────────────────

def build_all_routers(
    profiles: dict,
    aliases: list[str],
    lambda_values: list[float],
    random_seeds: int = 20,
) -> list[Router]:
    routers: list[Router] = []

    # Fixed baselines
    for alias in aliases:
        routers.append(AlwaysRouter(alias))

    # Random (averaged over 20 seeds in evaluate.py)
    for seed in range(random_seeds):
        routers.append(RandomRouter(aliases, seed=seed))

    # Profile routers
    for lam in lambda_values:
        routers.append(ProfileRouter(profiles, aliases, lam, use_lcb=False))
        routers.append(ProfileRouter(profiles, aliases, lam, use_lcb=True))

    # Cascade (needs small and large aliases)
    if "small" in aliases and "large" in aliases:
        routers.append(CascadeRouter("small", "large"))

    # Oracle added separately in evaluate.py (needs test data)
    return routers

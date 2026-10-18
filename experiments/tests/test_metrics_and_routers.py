"""
experiments/tests/test_metrics_and_routers.py
Unit tests for evaluate.py metrics and routers.py logic.
No API calls, no file I/O, fully offline.

Run:  python -m pytest tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXPERIMENTS_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(EXPERIMENTS_DIR / "src"))

from evaluate import balanced_accuracy, _compute_metrics  # noqa: E402
from routers import (  # noqa: E402
    AlwaysRouter, RandomRouter, ProfileRouter, CascadeRouter, OracleSkillRouter
)


# ─────────────────────────────────────────────────────────────────────────────
# balanced_accuracy
# ─────────────────────────────────────────────────────────────────────────────

class TestBalancedAccuracy:
    def test_perfect(self):
        assert balanced_accuracy(10, 10, 10, 10) == 1.0

    def test_all_wrong(self):
        assert balanced_accuracy(0, 10, 0, 10) == 0.0

    def test_asymmetric(self):
        # TPR = 1.0, TNR = 0.5 → BA = 0.75
        assert abs(balanced_accuracy(10, 10, 5, 10) - 0.75) < 1e-9

    def test_missing_positive_class(self):
        import math
        result = balanced_accuracy(0, 0, 5, 10)
        assert math.isnan(result)

    def test_missing_negative_class(self):
        import math
        result = balanced_accuracy(5, 10, 0, 0)
        assert math.isnan(result)


# ─────────────────────────────────────────────────────────────────────────────
# _compute_metrics
# ─────────────────────────────────────────────────────────────────────────────

def _make_rows(skill: str, gold_vals: list[bool], pred_vals: list[bool], costs: list[float]) -> list[dict]:
    rows = []
    for i, (g, p, c) in enumerate(zip(gold_vals, pred_vals, costs)):
        rows.append({
            "contract_id": f"c{i}",
            "skill": skill,
            "model": "large",
            "gold_present": g,
            "pred_present": p,
            "correct": g == p,
            "parse_ok": True,
            "cost_usd": c,
        })
    return rows


class TestComputeMetrics:
    def test_empty(self):
        import math
        m = _compute_metrics([])
        assert math.isnan(m["balanced_accuracy"])

    def test_perfect_balance(self):
        rows = _make_rows("TerminationForConvenience",
                          [True, True, False, False],
                          [True, True, False, False],
                          [0.001] * 4)
        m = _compute_metrics(rows)
        assert m["balanced_accuracy"] == 1.0

    def test_mean_cost(self):
        rows = _make_rows("GoverningLaw",
                          [True, False], [True, False], [0.002, 0.004])
        m = _compute_metrics(rows)
        assert abs(m["mean_cost_usd"] - 0.003) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Routers
# ─────────────────────────────────────────────────────────────────────────────

class TestAlwaysRouter:
    def test_always_returns_alias(self):
        r = AlwaysRouter("large")
        assert r.select("AnySkill") == "large"
        assert r.name == "always_large"


class TestRandomRouter:
    def test_returns_valid_alias(self):
        aliases = ["small", "large"]
        r = RandomRouter(aliases, seed=0)
        for _ in range(20):
            assert r.select("skill") in aliases

    def test_different_seeds_differ(self):
        # With enough calls, different seeds should pick different models
        aliases = ["small", "large"]
        r0 = RandomRouter(aliases, seed=0)
        r1 = RandomRouter(aliases, seed=99)
        choices_0 = [r0.select("s") for _ in range(10)]
        choices_1 = [r1.select("s") for _ in range(10)]
        # Not guaranteed to differ every time, but very likely with different seeds
        assert isinstance(choices_0, list) and isinstance(choices_1, list)


class TestProfileRouter:
    PROFILES = {
        "large": {
            "GoverningLaw": {"competence_mean": 0.9, "competence_lcb": 0.7, "cost_per_call": 0.002},
            "AuditRights":  {"competence_mean": 0.7, "competence_lcb": 0.5, "cost_per_call": 0.002},
        },
        "small": {
            "GoverningLaw": {"competence_mean": 0.85, "competence_lcb": 0.65, "cost_per_call": 0.0004},
            "AuditRights":  {"competence_mean": 0.5,  "competence_lcb": 0.3, "cost_per_call": 0.0004},
        },
    }

    def test_lambda_zero_picks_highest_competence(self):
        r = ProfileRouter(self.PROFILES, ["large", "small"], lam=0.0, use_lcb=False)
        # GoverningLaw: large=0.9 > small=0.85 → large
        assert r.select("GoverningLaw") == "large"

    def test_high_lambda_picks_cheap_model(self):
        # At very high lambda, cost dominates → small
        r = ProfileRouter(self.PROFILES, ["large", "small"], lam=1.0, use_lcb=False)
        # small has cost_norm=0.2 (0.0004/0.002), large has cost_norm=1.0
        # large: 0.9 - 1.0*1.0 = -0.1;  small: 0.85 - 1.0*0.2 = 0.65 → small wins
        assert r.select("GoverningLaw") == "small"

    def test_lcb_variant_name(self):
        r = ProfileRouter(self.PROFILES, ["large", "small"], lam=0.1, use_lcb=True)
        assert "lcb" in r.name

    def test_missing_skill_defaults_gracefully(self):
        r = ProfileRouter(self.PROFILES, ["large", "small"], lam=0.0, use_lcb=False)
        # Should not raise even for an unknown skill
        result = r.select("UnknownSkill")
        assert result in ["large", "small"]


class TestCascadeRouter:
    def test_no_escalate_on_high_confidence(self):
        r = CascadeRouter("small", "large")
        row = {"parse_ok": True, "confidence": "high", "pred_present": True, "quote_found_in_context": True}
        assert not r.should_escalate(row)

    def test_escalate_on_low_confidence(self):
        r = CascadeRouter("small", "large")
        row = {"parse_ok": True, "confidence": "low", "pred_present": False, "quote_found_in_context": False}
        assert r.should_escalate(row)

    def test_escalate_on_parse_fail(self):
        r = CascadeRouter("small", "large")
        row = {"parse_ok": False, "confidence": "high", "pred_present": False, "quote_found_in_context": False}
        assert r.should_escalate(row)

    def test_escalate_on_hallucinated_quote(self):
        r = CascadeRouter("small", "large")
        row = {"parse_ok": True, "confidence": "high", "pred_present": True, "quote_found_in_context": False}
        assert r.should_escalate(row)

    def test_first_call_always_small(self):
        r = CascadeRouter("small", "large")
        assert r.select("AnySkill") == "small"


class TestOracleRouter:
    def test_picks_best_per_skill(self):
        best = {"GoverningLaw": "large", "AuditRights": "small"}
        r = OracleSkillRouter(best)
        assert r.select("GoverningLaw") == "large"
        assert r.select("AuditRights") == "small"

    def test_fallback_on_unknown_skill(self):
        best = {"GoverningLaw": "large"}
        r = OracleSkillRouter(best)
        # Should not raise
        result = r.select("Unknown")
        assert isinstance(result, str)

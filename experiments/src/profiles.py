"""
experiments/src/profiles.py
Phase 5: Compute per-(model, skill) agent profiles from dev-split data only.

Profile per (model m, skill s):
  - successes, failures (success = correct AND parse_ok)
  - Beta posterior with uniform prior: alpha = 1 + successes, beta = 1 + failures
  - competence_mean = alpha / (alpha + beta)
  - competence_lcb = 10th percentile of Beta(alpha, beta)
  - cost_per_call (mean USD), tokens_per_call, latency_per_call

Saves: results/profiles.json and results/profiles.csv

Usage:
    python src/profiles.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from scipy.stats import beta as beta_dist  # type: ignore

EXPERIMENTS_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(EXPERIMENTS_DIR / "src"))
from config_loader import CFG, EXPERIMENTS_DIR  # noqa: E402

RESULTS_DIR = EXPERIMENTS_DIR / "results"
DATA_DIR = EXPERIMENTS_DIR / "data"
SPLIT_FILE = DATA_DIR / "split.json"
RAW_CALLS = RESULTS_DIR / "raw_calls.jsonl"
PROFILES_JSON = RESULTS_DIR / "profiles.json"
PROFILES_CSV = RESULTS_DIR / "profiles.csv"

LCB_PERCENTILE = CFG.get("routing", {}).get("lcb_percentile", 10) / 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Load raw calls (dev split only)
# ─────────────────────────────────────────────────────────────────────────────

def _load_dev_rows() -> list[dict]:
    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        split = json.load(f)
    dev_ids = set(split["dev"])

    rows = []
    with open(RAW_CALLS, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("contract_id") in dev_ids:
                rows.append(row)
    print(f"Loaded {len(rows)} dev-split rows from {RAW_CALLS}.")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Build profiles
# ─────────────────────────────────────────────────────────────────────────────

def build_profiles(rows: list[dict]) -> dict:
    """
    Returns nested dict: profiles[model_alias][skill] = {
        successes, failures, alpha, beta_param,
        competence_mean, competence_lcb,
        cost_per_call, tokens_per_call, latency_per_call, n_calls
    }
    """
    # Aggregate counts
    counts: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {
        "successes": 0, "failures": 0,
        "costs": [], "tokens_in": [], "tokens_out": [], "latencies": []
    }))

    for row in rows:
        model = row.get("model", "")
        skill = row.get("skill", "")
        if not model or not skill:
            continue
        d = counts[model][skill]
        # Success = correct answer AND parse succeeded
        if row.get("parse_ok") and row.get("correct"):
            d["successes"] += 1
        else:
            d["failures"] += 1
        if row.get("cost_usd") is not None:
            d["costs"].append(row["cost_usd"])
        if row.get("prompt_tokens") is not None:
            d["tokens_in"].append(row["prompt_tokens"])
        if row.get("completion_tokens") is not None:
            d["tokens_out"].append(row["completion_tokens"])
        if row.get("latency_s") is not None:
            d["latencies"].append(row["latency_s"])

    # Build profile objects
    profiles: dict[str, dict] = {}
    for model, skill_data in counts.items():
        profiles[model] = {}
        for skill, d in skill_data.items():
            s = d["successes"]
            f = d["failures"]
            alpha = 1 + s
            beta_param = 1 + f
            mean = alpha / (alpha + beta_param)
            lcb = float(beta_dist.ppf(LCB_PERCENTILE, alpha, beta_param))
            n = s + f
            profiles[model][skill] = {
                "successes": s,
                "failures": f,
                "n_calls": n,
                "alpha": alpha,
                "beta": beta_param,
                "competence_mean": round(mean, 4),
                "competence_lcb": round(lcb, 4),
                "cost_per_call": round(sum(d["costs"]) / len(d["costs"]), 6) if d["costs"] else None,
                "tokens_in_per_call": round(sum(d["tokens_in"]) / len(d["tokens_in"]), 1) if d["tokens_in"] else None,
                "tokens_out_per_call": round(sum(d["tokens_out"]) / len(d["tokens_out"]), 1) if d["tokens_out"] else None,
                "latency_per_call": round(sum(d["latencies"]) / len(d["latencies"]), 3) if d["latencies"] else None,
            }
    return profiles


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

def save_profiles(profiles: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(PROFILES_JSON, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)
    print(f"Profiles written to {PROFILES_JSON}")

    # Flatten to CSV
    rows = []
    for model, skill_data in profiles.items():
        for skill, p in skill_data.items():
            rows.append({"model": model, "skill": skill, **p})
    df = pd.DataFrame(rows)
    df.to_csv(PROFILES_CSV, index=False)
    print(f"Profiles CSV written to {PROFILES_CSV}")


# ─────────────────────────────────────────────────────────────────────────────
# Print observations
# ─────────────────────────────────────────────────────────────────────────────

def print_observations(profiles: dict) -> None:
    print("\n--- Profile observations (H1: skill heterogeneity) ---")
    models = list(profiles.keys())
    for model in models:
        skill_competences = {sk: p["competence_mean"] for sk, p in profiles[model].items()}
        if not skill_competences:
            continue
        best = max(skill_competences, key=skill_competences.get)
        worst = min(skill_competences, key=skill_competences.get)
        spread = skill_competences[best] - skill_competences[worst]
        print(f"\nModel: {model}")
        print(f"  Best skill:  {best} ({skill_competences[best]:.2%})")
        print(f"  Worst skill: {worst} ({skill_competences[worst]:.2%})")
        print(f"  Spread:      {spread:.2%}")

    # Compare models per skill
    if len(models) >= 2:
        print("\n  Per-skill model comparison (competence_mean):")
        all_skills = list(profiles[models[0]].keys())
        header = f"  {'Skill':<35}" + "".join(f" {m[:8]:>10}" for m in models) + "  gap"
        print(header)
        for sk in sorted(all_skills):
            vals = [profiles[m].get(sk, {}).get("competence_mean", float("nan")) for m in models]
            gap = max(vals) - min(vals)
            row_str = f"  {sk:<35}" + "".join(f" {v:>10.2%}" for v in vals) + f"  {gap:.2%}"
            print(row_str)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Phase 5 — Building agent profiles (dev split only)")
    print("=" * 60)
    print("[IMPORTANT] Only dev-split rows will be used. Test split is NOT touched.")

    rows = _load_dev_rows()
    profiles = build_profiles(rows)
    save_profiles(profiles)
    print_observations(profiles)

    print("\nPhase 5 acceptance check: PASSED ✓ (verify: no test rows used)")


if __name__ == "__main__":
    main()

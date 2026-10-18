"""
experiments/src/plots.py
Phase 6: Pareto plot (accuracy vs. cost) and per-skill heatmap.

Outputs:
  - results/pareto.png
  - results/skill_heatmap.png

Usage:
    python src/plots.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

EXPERIMENTS_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(EXPERIMENTS_DIR / "src"))
from config_loader import EXPERIMENTS_DIR  # noqa: E402

RESULTS_DIR = EXPERIMENTS_DIR / "results"
PROFILES_JSON = RESULTS_DIR / "profiles.json"
ROUTER_EVAL_CSV = RESULTS_DIR / "router_eval.csv"
PARETO_PNG = RESULTS_DIR / "pareto.png"
HEATMAP_PNG = RESULTS_DIR / "skill_heatmap.png"


# ─────────────────────────────────────────────────────────────────────────────
# Pareto helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_pareto(costs: np.ndarray, accs: np.ndarray) -> np.ndarray:
    """Return bool mask of Pareto-efficient points (min cost, max acc)."""
    n = len(costs)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j has lower cost AND higher (or equal) accuracy
            if costs[j] <= costs[i] and accs[j] >= accs[i] and (costs[j] < costs[i] or accs[j] > accs[i]):
                dominated[i] = True
                break
    return ~dominated


# ─────────────────────────────────────────────────────────────────────────────
# Pareto plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_pareto(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_facecolor("#0f1117")
    fig.patch.set_facecolor("#0f1117")

    # Colour scheme
    COLOURS = {
        "always": "#ef4444",
        "random": "#6b7280",
        "profile_mean": "#3b82f6",
        "profile_lcb": "#8b5cf6",
        "cascade": "#f59e0b",
        "oracle_skill": "#10b981",
    }

    def _color(name: str) -> str:
        for k, c in COLOURS.items():
            if k in name:
                return c
        return "#9ca3af"

    def _marker(name: str) -> str:
        if "oracle" in name:
            return "D"
        if "always" in name:
            return "s"
        if "cascade" in name:
            return "^"
        if "random" in name:
            return "x"
        return "o"

    # Filter to test split, drop NaN
    plot_df = df[df["split"] == "test"].dropna(subset=["balanced_accuracy", "mean_cost_usd"])

    # Deduplicate random routers — average them
    random_rows = plot_df[plot_df["router"].str.startswith("random_")]
    non_random = plot_df[~plot_df["router"].str.startswith("random_")]
    if not random_rows.empty:
        random_avg = random_rows[["balanced_accuracy", "mean_cost_usd", "ba_ci_lo", "ba_ci_hi"]].mean()
        random_avg["router"] = "random (avg)"
        random_avg["split"] = "test"
        random_avg["n"] = random_rows["n"].mean()
        non_random = pd.concat([non_random, pd.DataFrame([random_avg])], ignore_index=True)
    plot_df = non_random

    costs = plot_df["mean_cost_usd"].values.astype(float)
    accs = plot_df["balanced_accuracy"].values.astype(float)
    pareto_mask = _is_pareto(costs, accs)

    # Draw Pareto frontier
    pareto_pts = plot_df[pareto_mask].sort_values("mean_cost_usd")
    if len(pareto_pts) >= 2:
        ax.plot(
            pareto_pts["mean_cost_usd"],
            pareto_pts["balanced_accuracy"],
            color="#ffffff", linewidth=1.2, linestyle="--", alpha=0.4, zorder=1
        )

    # Scatter all points
    for _, row in plot_df.iterrows():
        is_oracle = "oracle" in row["router"]
        color = _color(row["router"])
        marker = _marker(row["router"])
        alpha = 0.5 if is_oracle else 0.9
        zorder = 5 if is_oracle else 3
        ax.scatter(
            row["mean_cost_usd"], row["balanced_accuracy"],
            color=color, marker=marker, s=90, alpha=alpha,
            zorder=zorder, linewidths=0.5, edgecolors="#ffffff"
        )
        # Error bars for non-random, non-oracle
        if not is_oracle and "random" not in row["router"]:
            lo = row.get("ba_ci_lo", float("nan"))
            hi = row.get("ba_ci_hi", float("nan"))
            if lo == lo and hi == hi:
                ax.errorbar(
                    row["mean_cost_usd"], row["balanced_accuracy"],
                    yerr=[[row["balanced_accuracy"] - lo], [hi - row["balanced_accuracy"]]],
                    fmt="none", color=color, alpha=0.4, linewidth=1, capsize=3, zorder=2
                )
        # Label
        label = row["router"]
        if len(label) > 22:
            label = label[:22]
        ax.annotate(
            label,
            (row["mean_cost_usd"], row["balanced_accuracy"]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=6.5, color=color, alpha=0.85
        )

    # Legend patches
    patches = [mpatches.Patch(color=c, label=k) for k, c in COLOURS.items()]
    ax.legend(handles=patches, fontsize=8, framealpha=0.2,
              labelcolor="white", facecolor="#1e2030")

    ax.set_xlabel("Mean cost per call (USD)", color="#9ca3af", fontsize=11)
    ax.set_ylabel("Macro balanced accuracy (test)", color="#9ca3af", fontsize=11)
    ax.set_title(
        "Accuracy vs. Cost — Skill-Aware Routing on CUAD\n"
        "(oracle = dashed upper bound; not achievable in practice)",
        color="white", fontsize=12, pad=14
    )
    ax.tick_params(colors="#6b7280")
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")

    fig.tight_layout()
    fig.savefig(PARETO_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Pareto plot saved to {PARETO_PNG}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Skill heatmap (model × skill, competence_mean)
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap() -> None:
    with open(PROFILES_JSON, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    models = list(profiles.keys())
    skills = sorted({sk for m in models for sk in profiles[m].keys()})

    data = np.full((len(models), len(skills)), fill_value=np.nan)
    for i, model in enumerate(models):
        for j, skill in enumerate(skills):
            p = profiles[model].get(skill, {})
            v = p.get("competence_mean")
            if v is not None:
                data[i, j] = v

    fig, ax = plt.subplots(figsize=(max(10, len(skills) * 1.2), max(4, len(models) * 1.4)))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    im = ax.imshow(data, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Competence (mean)", color="#9ca3af")
    cbar.ax.yaxis.set_tick_params(color="#6b7280")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#9ca3af")

    ax.set_xticks(range(len(skills)))
    ax.set_xticklabels(skills, rotation=40, ha="right", color="#9ca3af", fontsize=8)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, color="#9ca3af", fontsize=9)

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(skills)):
            val = data[i, j]
            if val == val:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.5, color="black" if val > 0.5 else "white")

    ax.set_title("Model × Skill Competence (dev split)", color="white", fontsize=12, pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#374151")

    fig.tight_layout()
    fig.savefig(HEATMAP_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Heatmap saved to {HEATMAP_PNG}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Phase 6 — Generating plots")
    print("=" * 60)

    if not ROUTER_EVAL_CSV.exists():
        print(f"ERROR: {ROUTER_EVAL_CSV} not found. Run evaluate.py first.")
        sys.exit(1)

    df = pd.read_csv(ROUTER_EVAL_CSV)
    plot_pareto(df)

    if PROFILES_JSON.exists():
        plot_heatmap()
    else:
        print(f"[WARN] {PROFILES_JSON} not found. Skipping heatmap.")

    print("\nDone. Outputs:")
    print(f"  {PARETO_PNG}")
    print(f"  {HEATMAP_PNG}")


if __name__ == "__main__":
    main()

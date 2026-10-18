"""
experiments/src/load_cuad.py
Phase 1: Download CUAD via Hugging Face datasets, parse contracts and gold labels.

Outputs:
  - data/split.json  (committed to git)
  - Returns a list of Contract dicts for use by other modules

Usage:
    python src/load_cuad.py [--sample 200]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

EXPERIMENTS_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(EXPERIMENTS_DIR / "src"))
from config_loader import CFG, EXPERIMENTS_DIR  # noqa: E402

DATA_DIR = EXPERIMENTS_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SPLIT_FILE = DATA_DIR / "split.json"
SKILLS_FILE = DATA_DIR / "skills.yaml"

# ─────────────────────────────────────────────────────────────────────────────
# Download / load CUAD
# ─────────────────────────────────────────────────────────────────────────────

HF_DATASET = "theatticusproject/cuad-qa"


def _load_cuad_hf() -> Any:
    """Load CUAD from Hugging Face; cache locally to data/raw/."""
    from datasets import load_dataset  # type: ignore

    print(f"Loading CUAD from Hugging Face ({HF_DATASET}) — first run downloads ~300MB ...")
    ds = load_dataset(HF_DATASET, cache_dir=str(RAW_DIR))
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# Parse into contract-level dicts
# ─────────────────────────────────────────────────────────────────────────────

def _parse_contracts(ds: Any, skill_labels: list[str]) -> list[dict]:
    """
    Returns a list of:
      {
        "contract_id": str,
        "title": str,
        "text": str,
        "gold": {skill_label: {"present": bool, "spans": [str]}}
      }
    
    Groups rows by contract (CUAD has one row per question per contract).
    """
    # CUAD train split has all annotated examples
    split = ds["train"] if "train" in ds else list(ds.values())[0]

    contracts: dict[str, dict] = {}
    skipped_fields: set[str] = set()

    for row in tqdm(split, desc="Parsing CUAD rows"):
        # Field names to check (VERIFY against actual dataset)
        title = row.get("title", row.get("contract_title", ""))
        context = row.get("context", row.get("text", ""))
        question = row.get("question", "")
        answers = row.get("answers", {})

        if not title or not context:
            continue

        cid = title.strip()
        if cid not in contracts:
            contracts[cid] = {
                "contract_id": cid,
                "title": cid,
                "text": context,
                "gold": {},
            }

        # Match question to a known skill label (case-insensitive substring)
        matched_skill = None
        for sl in skill_labels:
            if sl.lower() in question.lower():
                matched_skill = sl
                break

        if matched_skill is None:
            skipped_fields.add(question[:60])
            continue

        # Gold: at least one non-empty answer span = present
        ans_texts: list[str] = []
        if isinstance(answers, dict):
            ans_texts = answers.get("text", [])
        elif isinstance(answers, list):
            ans_texts = [a.get("text", "") if isinstance(a, dict) else str(a) for a in answers]

        non_empty = [t for t in ans_texts if t and t.strip()]
        contracts[cid]["gold"][matched_skill] = {
            "present": len(non_empty) > 0,
            "spans": non_empty,
        }

    print(f"Parsed {len(contracts)} unique contracts.")
    if skipped_fields:
        print(f"  Questions not matched to any skill ({len(skipped_fields)} unique): printed to console for VERIFY.")
        for q in sorted(skipped_fields)[:20]:
            print(f"    - {q}")

    return list(contracts.values())


# ─────────────────────────────────────────────────────────────────────────────
# Prevalence analysis
# ─────────────────────────────────────────────────────────────────────────────

def _prevalence_table(contracts: list[dict], skill_labels: list[str]) -> dict[str, float]:
    counts: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    for c in contracts:
        for sl in skill_labels:
            if sl in c["gold"]:
                totals[sl] += 1
                if c["gold"][sl]["present"]:
                    counts[sl] += 1
    return {sl: counts[sl] / totals[sl] if totals[sl] else 0.0 for sl in skill_labels}


def print_prevalence(contracts: list[dict], skill_labels: list[str]) -> None:
    prev = _prevalence_table(contracts, skill_labels)
    print("\n  Prevalence per skill (on sampled contracts):")
    print(f"  {'Skill':<35} {'Prevalence':>12}  {'Band':>8}")
    print(f"  {'-'*35} {'-'*12}  {'-'*8}")
    for sl, p in sorted(prev.items(), key=lambda x: x[1]):
        band = "OK" if 0.15 <= p <= 0.85 else ("LOW" if p < 0.15 else "HIGH")
        print(f"  {sl:<35} {p:>11.1%}  {band:>8}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Stratified split
# ─────────────────────────────────────────────────────────────────────────────

def _stratified_split(
    contracts: list[dict],
    skill_labels: list[str],
    n_total: int,
    n_dev: int,
    n_pilot: int,
    seed: int,
) -> dict:
    """
    Sample n_total contracts, stratified so each skill has both classes in
    dev and test. Retry with a larger pool if stratification fails.
    Returns {"dev": [...ids], "test": [...ids], "pilot": [...ids]}
    """
    rng = random.Random(seed)

    # Only keep contracts that have gold labels for ALL chosen skills
    eligible = [c for c in contracts if all(s in c["gold"] for s in skill_labels)]
    print(f"  Eligible contracts (gold for all skills): {len(eligible)}")

    if len(eligible) < n_total:
        raise ValueError(
            f"Only {len(eligible)} contracts have gold labels for all chosen skills, "
            f"but n_total={n_total}. Either reduce n_total or choose fewer skills."
        )

    # Stratified: ensure each skill has positives in both splits
    best: dict | None = None
    for attempt in range(2000):
        sample = rng.sample(eligible, n_total)
        dev_ids = [c["contract_id"] for c in sample[:n_dev]]
        test_ids = [c["contract_id"] for c in sample[n_dev:]]
        pilot_ids = dev_ids[:n_pilot]

        dev_set = [c for c in sample if c["contract_id"] in dev_ids]
        test_set = [c for c in sample if c["contract_id"] in test_ids]

        ok = True
        for sl in skill_labels:
            dev_pos = sum(1 for c in dev_set if c["gold"][sl]["present"])
            dev_neg = n_dev - dev_pos
            tst_pos = sum(1 for c in test_set if c["gold"][sl]["present"])
            tst_neg = (n_total - n_dev) - tst_pos
            if dev_pos == 0 or dev_neg == 0 or tst_pos == 0 or tst_neg == 0:
                ok = False
                break

        if ok:
            best = {"dev": dev_ids, "test": test_ids, "pilot": pilot_ids}
            print(f"  Stratified split found on attempt {attempt + 1}.")
            break

    if best is None:
        raise ValueError(
            "Could not find a stratified split after 2000 attempts. "
            "Try adjusting skill selection or increasing n_total."
        )
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

# Candidate CUAD skill labels — VERIFY these names match the dataset's question text
# before finalising. These are the strings we search for in the 'question' field.
CANDIDATE_SKILLS = [
    "Governing Law",
    "Anti-Assignment",
    "Cap On Liability",
    "Termination For Convenience",
    "Audit Rights",
    "Insurance",
    "License Grant",
    "Non-Compete",
    "Exclusivity",
    "Liquidated Damages",
    "Renewal Term",
    "IP Ownership Assignment",
    "Change Of Control",
    "Revenue/Profit Sharing",
    "Warranty Duration",
    "Minimum Commitment",
    "Post-Termination Services",
    "Uncapped Liability",
]


def main(sample_size: int = 200) -> None:
    print("=" * 60)
    print("Phase 1 — CUAD data loading")
    print("=" * 60)

    ds = _load_cuad_hf()

    # Parse with all candidate skills to get full prevalence picture
    print("\nStep 1: Parsing contracts for all candidate skills ...")
    contracts_all = _parse_contracts(ds, CANDIDATE_SKILLS)

    # Show prevalence on up to sample_size contracts
    sample_for_prev = contracts_all[:sample_size] if len(contracts_all) > sample_size else contracts_all
    print(f"\nStep 2: Prevalence on {len(sample_for_prev)} contracts:")
    print_prevalence(sample_for_prev, CANDIDATE_SKILLS)

    # ── READ skills.yaml if it already exists, else print instructions ────────
    if not SKILLS_FILE.exists():
        print(
            "skills.yaml does not exist yet.\n"
            "  1. Review the prevalence table above.\n"
            "  2. Choose 8 skills in the 25-75% band.\n"
            "  3. Create experiments/data/skills.yaml (see template below).\n"
            "  4. Re-run this script.\n\n"
            "Template:\n"
            "  skills:\n"
            "    - label: 'Governing Law'\n"
            "      definition: 'Specifies which jurisdiction's laws govern the contract.'\n"
            "      retrieval_query: 'governing law jurisdiction'\n"
            "      cuad_label: 'Governing Law'\n"
        )
        sys.exit(0)

    with open(SKILLS_FILE, "r", encoding="utf-8") as f:
        skills_cfg = yaml.safe_load(f)
    chosen_skills: list[str] = [s["label"] for s in skills_cfg["skills"]]
    print(f"Chosen skills ({len(chosen_skills)}): {chosen_skills}")

    # Parse again with chosen skills only for speed
    contracts = _parse_contracts(ds, chosen_skills)

    # Split
    data_cfg = CFG.get("data", {})
    n_total = data_cfg.get("n_contracts", 60)
    n_dev = data_cfg.get("n_dev", 30)
    n_pilot = data_cfg.get("n_pilot", 8)
    seed = data_cfg.get("seed", 42)

    print(f"\nStep 3: Creating stratified split (n={n_total}, dev={n_dev}, test={n_total - n_dev}, pilot={n_pilot}, seed={seed}) ...")
    split = _stratified_split(contracts, chosen_skills, n_total, n_dev, n_pilot, seed)

    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_FILE, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    print(f"  Split written to {SPLIT_FILE} (commit this file).")

    # Print per-skill prevalence in each split
    id_to_contract = {c["contract_id"]: c for c in contracts}
    dev_contracts = [id_to_contract[cid] for cid in split["dev"] if cid in id_to_contract]
    test_contracts = [id_to_contract[cid] for cid in split["test"] if cid in id_to_contract]

    print("\nStep 4: Prevalence per skill per split:")
    print(f"  {'Skill':<35} {'Dev %':>8}  {'Test %':>8}")
    print(f"  {'-'*35} {'-'*8}  {'-'*8}")
    for sl in chosen_skills:
        dev_p = sum(1 for c in dev_contracts if c["gold"].get(sl, {}).get("present")) / len(dev_contracts)
        tst_p = sum(1 for c in test_contracts if c["gold"].get(sl, {}).get("present")) / len(test_contracts)
        print(f"  {sl:<35} {dev_p:>7.1%}  {tst_p:>7.1%}")

    print("\nPhase 1 acceptance check: PASSED ✓" if SPLIT_FILE.exists() else "\n[ERROR] split.json not written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=200, help="Number of contracts for prevalence analysis")
    args = parser.parse_args()
    main(args.sample)

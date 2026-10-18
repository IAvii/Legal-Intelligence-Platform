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

# CUAD primary source: CUADv1.json — official full dataset with all 510 contracts
# and ALL (contract, skill) pairs including negatives (is_impossible=True).
# Downloaded from https://github.com/The-Atticus-Project/cuad/raw/main/data.zip
# (data.zip -> CUADv1.json) on 2026-10-18. License: CC BY 4.0.
#
# chenghao/cuad_qa (HF parquet) omits negative rows, making prevalence unusable.
# We fall back to it only for the question-name VERIFY step.
CUAD_JSON = RAW_DIR / "cuad_extracted" / "CUADv1.json"
HF_DATASET = "chenghao/cuad_qa"   # used only for VERIFY step (listing all questions)


def _load_cuad_json() -> list[dict]:
    """
    Load CUADv1.json (SQuAD format) and return a flat list of
    {title, context, question, answers, is_impossible} dicts.
    """
    if not CUAD_JSON.exists():
        raise FileNotFoundError(
            f"CUADv1.json not found at {CUAD_JSON}.\n"
            "Run: Invoke-WebRequest -Uri https://github.com/The-Atticus-Project/cuad/raw/main/data.zip "
            "-OutFile data/raw/cuad_data.zip; Expand-Archive data/raw/cuad_data.zip data/raw/cuad_extracted"
        )
    print(f"Loading CUAD from {CUAD_JSON} ...")
    with open(CUAD_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    articles = data["data"]
    rows = []
    for article in articles:
        title = article["title"]
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                rows.append({
                    "title": title,
                    "context": context,
                    "question": qa["question"].strip(),
                    "answers": qa.get("answers", []),
                    "is_impossible": qa.get("is_impossible", len(qa.get("answers", [])) == 0),
                })
    contracts = len({r["title"] for r in rows})
    questions = len({r["question"] for r in rows})
    print(f"  {contracts} contracts x {questions} question types = {len(rows)} total rows")
    return rows


def _list_all_questions(rows: list[dict] | None = None) -> list[str]:
    """
    Extract all unique question strings from CUAD rows (or fall back to HF).
    VERIFY step: confirm which CUAD category names actually exist.
    """
    if rows:
        return sorted({r["question"] for r in rows if r.get("question")})
    if CUAD_JSON.exists():
        loaded = _load_cuad_json()
        return sorted({r["question"] for r in loaded if r.get("question")})
    try:
        import warnings
        warnings.filterwarnings("ignore")
        from datasets import load_dataset  # type: ignore
        ds = load_dataset(HF_DATASET)
        split = ds["train"] if "train" in ds else list(ds.values())[0]
        questions: set[str] = set()
        for row in split:
            q = row.get("question", "").strip()
            if q:
                questions.add(q)
        return sorted(questions)
    except Exception as e:
        print(f"  Note: HF verify fallback failed ({e}), using empty list.")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Parse into contract-level dicts
# ─────────────────────────────────────────────────────────────────────────────


def _parse_contracts(rows: list[dict], skills: list[dict]) -> list[dict]:
    """
    Parse flat CUADv1.json rows into contract-level dicts:
      {
        "contract_id": str,
        "text": str,
        "gold": {label: {"present": bool, "spans": [str]}}
      }

    skill["cuad_question"] is matched as a case-insensitive substring of
    the question text (the full CUAD question sentences contain the category
    name as a substring).
    """
    contracts: dict[str, dict] = {}
    skipped_questions: set[str] = set()

    for row in tqdm(rows, desc="Parsing CUAD rows"):
        title = row["title"].strip()
        context = row["context"]
        question = row["question"]
        answers = row["answers"]
        is_impossible = row["is_impossible"]

        if not title or not context:
            continue

        if title not in contracts:
            contracts[title] = {
                "contract_id": title,
                "text": context,
                "gold": {},
            }

        # Match by cuad_question substring (case-insensitive)
        matched_skill: dict | None = None
        for sk in skills:
            cq = sk["cuad_question"]
            if cq.lower() in question.lower():
                matched_skill = sk
                break

        if matched_skill is None:
            skipped_questions.add(question[:60])
            continue

        label = matched_skill["label"]
        # Gold: is_impossible=False AND at least one non-empty span
        non_empty = [a["text"] for a in answers if a.get("text", "").strip()] if answers else []
        present = (not is_impossible) and len(non_empty) > 0
        contracts[title]["gold"][label] = {
            "present": present,
            "spans": non_empty,
        }

    result = list(contracts.values())
    print(f"Parsed {len(result)} unique contracts.")
    n_unmatched = len(skipped_questions)
    if n_unmatched:
        print(f"  {n_unmatched} question types not matched to any skill (expected — we only use 8 of 41).")
    return result



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

def main(sample_size: int = 200) -> None:
    print("=" * 60)
    print("Phase 1 - CUAD data loading")
    print("=" * 60)

    # Step 1: Load CUADv1.json (includes negative rows)
    print("Step 1: Loading CUADv1.json (includes negative/absent rows) ...")
    rows = _load_cuad_json()

    # Step 0: VERIFY all category names (quick local lookup from CUADv1.json)
    print("\nStep 0 (VERIFY): All unique question/category names in CUAD:")
    all_questions = _list_all_questions(rows)
    for i, q in enumerate(all_questions, 1):
        print(f"  {i:>3}. {q}")
    print(f"\n  Total categories: {len(all_questions)}\n")

    # Step 2: Load skills.yaml if it exists, else print instructions and exit
    if not SKILLS_FILE.exists():
        print(
            "\nskills.yaml not found. Create experiments/data/skills.yaml with:\n"
            "  skills:\n"
            "    - label: 'TerminationForConvenience'\n"
            "      cuad_question: 'Termination For Convenience'\n"
            "      definition: 'A clause allowing one party to end the contract without cause.'\n"
            "      retrieval_query: 'termination convenience notice cancel'\n"
        )
        sys.exit(0)

    with open(SKILLS_FILE, "r", encoding="utf-8") as f:
        skills_cfg = yaml.safe_load(f)
    skills: list[dict] = skills_cfg["skills"]
    labels: list[str] = [s["label"] for s in skills]
    print(f"\nStep 2: Using {len(labels)} chosen skills from skills.yaml:")
    for s in skills:
        print(f"  - {s['label']} (matches: '{s['cuad_question']}')")

    # Step 3: Parse all contracts with chosen skills
    print("\nStep 3: Parsing contracts ...")
    contracts = _parse_contracts(rows, skills)

    # Step 4: Prevalence on sample
    sample = contracts[:sample_size] if len(contracts) > sample_size else contracts
    print(f"\nStep 4: Prevalence on {len(sample)} contracts:")
    print_prevalence(sample, labels)

    # Step 5: Stratified split
    data_cfg = CFG.get("data", {})
    n_total = data_cfg.get("n_contracts", 60)
    n_dev = data_cfg.get("n_dev", 30)
    n_pilot = data_cfg.get("n_pilot", 8)
    seed = data_cfg.get("seed", 42)

    print(f"\nStep 5: Creating stratified split (total={n_total}, dev={n_dev}, "
          f"test={n_total - n_dev}, pilot={n_pilot}, seed={seed}) ...")
    split = _stratified_split(contracts, labels, n_total, n_dev, n_pilot, seed)

    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLIT_FILE, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    print(f"  split.json written to {SPLIT_FILE}  [COMMIT THIS FILE]")

    # Step 6: Per-skill prevalence in each split
    id_to_contract = {c["contract_id"]: c for c in contracts}
    dev_cs = [id_to_contract[cid] for cid in split["dev"] if cid in id_to_contract]
    tst_cs = [id_to_contract[cid] for cid in split["test"] if cid in id_to_contract]

    print("\nStep 6: Per-skill prevalence per split:")
    print(f"  {'Skill':<35} {'Dev %':>8}  {'Test %':>8}  {'Both classes?':>14}")
    print(f"  {'-'*35} {'-'*8}  {'-'*8}  {'-'*14}")
    all_ok = True
    for sl in labels:
        dev_p = sum(1 for c in dev_cs if c["gold"].get(sl, {}).get("present")) / max(len(dev_cs), 1)
        tst_p = sum(1 for c in tst_cs if c["gold"].get(sl, {}).get("present")) / max(len(tst_cs), 1)
        dev_ok = 0 < dev_p < 1
        tst_ok = 0 < tst_p < 1
        ok = "OK" if (dev_ok and tst_ok) else "FAIL"
        if ok == "FAIL":
            all_ok = False
        print(f"  {sl:<35} {dev_p:>7.1%}  {tst_p:>7.1%}  {ok:>14}")

    if all_ok:
        print("\nPhase 1 acceptance check: PASSED [OK]")
        print("  Every skill has both classes in dev and test splits.")
    else:
        print("\n[WARN] Some skills are missing a class in dev or test - re-run or adjust n_contracts.")

    print(f"\n  split.json: dev={len(split['dev'])}, test={len(split['test'])}, "
          f"pilot={len(split.get('pilot', []))} contracts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=200, help="Number of contracts for prevalence analysis")
    args = parser.parse_args()
    main(args.sample)

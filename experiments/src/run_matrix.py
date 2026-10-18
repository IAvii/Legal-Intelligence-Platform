"""
experiments/src/run_matrix.py
Phase 2/4: Resumable runner — iterates over (contract, skill, model) triples,
calls the LLM, and appends results to results/raw_calls.jsonl.

Features:
  - Disk cache keyed by hash(model, contract_id, skill, prompt_version, k)
  - Resume: skips already-completed rows (checks cache, not the JSONL)
  - Writes one JSONL row per call atomically
  - Prints progress with tqdm
  - Respects --split flag: pilot / dev / test / all

Usage:
    python src/run_matrix.py --split pilot
    python src/run_matrix.py --split all
    python src/run_matrix.py --split dev
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import yaml
from tqdm import tqdm

EXPERIMENTS_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(EXPERIMENTS_DIR / "src"))
from config_loader import CFG, EXPERIMENTS_DIR  # noqa: E402
from groq_client import call_model  # noqa: E402
from windows import retrieve_windows, retrieval_recall  # noqa: E402

DATA_DIR = EXPERIMENTS_DIR / "data"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
CACHE_DIR = EXPERIMENTS_DIR / CFG.get("cache", {}).get("dir", "cache")
CALL_CACHE_DIR = CACHE_DIR / "calls"
CALL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RAW_CALLS_PATH = RESULTS_DIR / "raw_calls.jsonl"
SPLIT_FILE = DATA_DIR / "split.json"
SKILLS_FILE = DATA_DIR / "skills.yaml"

PROMPT_VERSION = CFG.get("llm", {}).get("prompt_version", "v1")
PROMPT_FILE = EXPERIMENTS_DIR / CFG.get("llm", {}).get("prompt_file", "prompts/skill_presence_v1.txt")
TOP_K = CFG.get("retrieval", {}).get("top_k", 3)


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _call_cache_key(model_alias: str, contract_id: str, skill_label: str, prompt_version: str, k: int) -> str:
    raw = f"{model_alias}|{contract_id}|{skill_label}|{prompt_version}|{k}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _cached_result(cache_key: str) -> dict | None:
    path = CALL_CACHE_DIR / f"{cache_key}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_cached_result(cache_key: str, row: dict) -> None:
    path = CALL_CACHE_DIR / f"{cache_key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(row, f)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_split() -> dict:
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"{SPLIT_FILE} not found. Run Phase 1 first: python src/load_cuad.py")
    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_skills() -> list[dict]:
    if not SKILLS_FILE.exists():
        raise FileNotFoundError(f"{SKILLS_FILE} not found. Create it during Phase 1.")
    with open(SKILLS_FILE, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["skills"]


def _contract_ids_for_split(split_cfg: dict, split_name: str) -> list[str]:
    if split_name == "pilot":
        return split_cfg["pilot"]
    elif split_name == "dev":
        return split_cfg["dev"]
    elif split_name == "test":
        return split_cfg["test"]
    elif split_name == "all":
        return split_cfg["dev"] + split_cfg["test"]
    else:
        raise ValueError(f"Unknown split: {split_name!r}. Use pilot/dev/test/all.")


def _load_contracts_from_hf(contract_ids: list[str], skill_labels: list[str]) -> list[dict]:
    """Load the subset of CUAD contracts needed for the run."""
    from load_cuad import _load_cuad_hf, _parse_contracts  # type: ignore
    ds = _load_cuad_hf()
    contracts = _parse_contracts(ds, skill_labels)
    id_map = {c["contract_id"]: c for c in contracts}
    missing = [cid for cid in contract_ids if cid not in id_map]
    if missing:
        print(f"[WARN] {len(missing)} contract IDs from split not found in CUAD. Check split.json matches this dataset version.")
    return [id_map[cid] for cid in contract_ids if cid in id_map]


# ─────────────────────────────────────────────────────────────────────────────
# Prompt rendering
# ─────────────────────────────────────────────────────────────────────────────

def _load_prompt_template() -> str:
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read()


_PROMPT_TEMPLATE: str | None = None


def _render_prompt(skill_name: str, skill_definition: str, context_text: str) -> str:
    global _PROMPT_TEMPLATE
    if _PROMPT_TEMPLATE is None:
        _PROMPT_TEMPLATE = _load_prompt_template()
    return _PROMPT_TEMPLATE.format(
        skill_name=skill_name,
        skill_definition=skill_definition,
        context=context_text,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _quote_found_in_context(quote: str | None, context_text: str) -> bool:
    if not quote:
        return False
    import re
    q_norm = re.sub(r"\s+", " ", quote).strip().lower()
    c_norm = re.sub(r"\s+", " ", context_text).strip().lower()
    return q_norm in c_norm if len(q_norm) >= 5 else False


# ─────────────────────────────────────────────────────────────────────────────
# Single call
# ─────────────────────────────────────────────────────────────────────────────

def _run_one(
    run_id: str,
    split_name: str,
    contract: dict,
    skill: dict,
    model_alias: str,
    top_k: int,
) -> dict:
    """Execute or replay one (contract, skill, model) call. Returns a log row."""
    cid = contract["contract_id"]
    skill_label = skill["label"]
    skill_def = skill["definition"]
    retrieval_query = skill.get("retrieval_query", skill_label)

    cache_key = _call_cache_key(model_alias, cid, skill_label, PROMPT_VERSION, top_k)
    cached = _cached_result(cache_key)
    if cached is not None:
        return cached

    # Retrieve windows
    windows = retrieve_windows(cid, contract["text"], retrieval_query, top_k=top_k)
    context_text = "\n\n---\n\n".join(w.text for w in windows)
    window_ids = [w.window_id for w in windows]

    # Gold
    gold_info = contract["gold"].get(skill_label, {"present": False, "spans": []})
    gold_present = gold_info["present"]
    gold_spans = gold_info.get("spans", [])

    # Retrieval recall
    ret_recall = retrieval_recall(windows, gold_spans) if gold_present else None

    # Prompt
    prompt_text = _render_prompt(skill_label, skill_def, context_text)
    system_msg = "You are a precise legal clause detector. Return only a JSON object."

    # Call model
    result = call_model(model_alias, system_msg, prompt_text)

    # Parse output fields
    parsed = result.parsed or {}
    pred_present = bool(parsed.get("present", False)) if result.parse_ok else False
    evidence_quote = parsed.get("evidence_quote") if result.parse_ok else None
    confidence = parsed.get("confidence", "low") if result.parse_ok else "low"
    correct = (pred_present == gold_present) if result.parse_ok else False
    quote_found = _quote_found_in_context(evidence_quote, context_text)

    row = {
        "run_id": run_id,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split": split_name,
        "contract_id": cid,
        "skill": skill_label,
        "model": model_alias,
        "model_id": result.model_id,
        "prompt_version": PROMPT_VERSION,
        "k": top_k,
        "retrieved_window_ids": window_ids,
        "gold_present": gold_present,
        "retrieval_recall": ret_recall,
        "pred_present": pred_present,
        "correct": correct,
        "parse_ok": result.parse_ok,
        "evidence_quote": evidence_quote,
        "quote_found_in_context": quote_found,
        "confidence": confidence,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "cost_usd": result.cost_usd,
        "latency_s": result.latency_s,
        "error": result.error,
    }

    _save_cached_result(cache_key, row)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(split_name: str = "pilot") -> None:
    import uuid

    run_id = str(uuid.uuid4())[:8]
    print(f"Run ID: {run_id}  |  Split: {split_name}  |  Prompt: {PROMPT_VERSION}")

    split_cfg = _load_split()
    skills = _load_skills()
    skill_labels = [s["label"] for s in skills]
    models_cfg = CFG.get("models", {})
    model_aliases = list(models_cfg.keys())

    contract_ids = _contract_ids_for_split(split_cfg, split_name)
    print(f"Contracts: {len(contract_ids)} | Skills: {len(skills)} | Models: {model_aliases}")
    print(f"Total calls (max): {len(contract_ids) * len(skills) * len(model_aliases)}")

    # Load contracts
    contracts = _load_contracts_from_hf(contract_ids, skill_labels)
    id_map = {c["contract_id"]: c for c in contracts}

    # Build job list
    jobs = [
        (cid, skill, alias)
        for cid in contract_ids
        for skill in skills
        for alias in model_aliases
        if cid in id_map
    ]

    # Check which are already cached
    cached_count = 0
    for cid, skill, alias in jobs:
        key = _call_cache_key(alias, cid, skill["label"], PROMPT_VERSION, TOP_K)
        if (CALL_CACHE_DIR / f"{key}.json").exists():
            cached_count += 1
    print(f"Already cached: {cached_count}/{len(jobs)}")

    # Run
    new_rows = 0
    errors = 0
    parse_fails = 0

    with open(RAW_CALLS_PATH, "a", encoding="utf-8") as out_f:
        for cid, skill, alias in tqdm(jobs, desc="Running calls"):
            key = _call_cache_key(alias, cid, skill["label"], PROMPT_VERSION, TOP_K)
            cached = _cached_result(key)
            if cached is not None:
                continue  # skip already-cached calls; don't re-write to JSONL

            contract = id_map[cid]
            row = _run_one(run_id, split_name, contract, skill, alias, TOP_K)
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            new_rows += 1

            if row.get("error"):
                errors += 1
            if not row.get("parse_ok"):
                parse_fails += 1

    print(f"\nDone. New rows written: {new_rows} | Errors: {errors} | Parse failures: {parse_fails}")
    parse_fail_rate = parse_fails / max(new_rows, 1)
    print(f"Parse failure rate (new calls): {parse_fail_rate:.1%}")

    if split_name == "pilot":
        print("\nPhase 2 acceptance check:")
        print("  Run this script again with --split pilot to verify cache hit (0 new API calls).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="pilot", choices=["pilot", "dev", "test", "all"])
    args = parser.parse_args()
    main(args.split)

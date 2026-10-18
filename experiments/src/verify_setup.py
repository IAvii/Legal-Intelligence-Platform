"""
experiments/src/verify_setup.py
Phase 0 acceptance check.

What it does:
  1. Verifies GROQ_API_KEY is set.
  2. Sends a 1-token test call to each configured model and prints the response + cost.
  3. Scrapes current pricing from the Groq pricing page and writes it into config.yaml.
  4. Prints a pass/fail summary.

Run:  python src/verify_setup.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import httpx
import yaml

# ── Bootstrap path so src.* imports work when run directly ───────────────────
EXPERIMENTS_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(EXPERIMENTS_DIR / "src"))

from config_loader import CFG, EXPERIMENTS_DIR, groq_api_key  # noqa: E402

try:
    from groq import Groq
except ImportError:
    print("ERROR: groq package not installed. Run:  pip install -r requirements.txt")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Price scraping
# ─────────────────────────────────────────────────────────────────────────────

GROQ_PRICING_URL = "https://groq.com/pricing/"

# Known patterns the page uses (verified 2026-10-18; re-verify if this breaks).
# Format on the page is typically something like:
#   "$0.05 / 1M tokens"  or  "Input: $0.05/M"  or rows in a table.
# We use a broad regex and match by model name proximity.
_PRICE_RE = re.compile(
    r"\$\s*([\d]+\.[\d]+)\s*/?\s*(?:1M|M|million)?\s*(?:input|output)?\s*tokens?",
    re.IGNORECASE,
)


def _scrape_prices() -> dict[str, dict[str, float | None]]:
    """
    Returns {model_id: {in: price_per_mtok, out: price_per_mtok}} or empty on failure.
    This is best-effort; if parsing fails we warn and continue with nulls.
    """
    results: dict[str, dict] = {}
    try:
        resp = httpx.get(GROQ_PRICING_URL, follow_redirects=True, timeout=15)
        text = resp.text

        models_cfg = CFG.get("models", {})
        for alias, mcfg in models_cfg.items():
            model_id: str = mcfg.get("id", "")
            # Search for a section of the page near the model ID
            idx = text.lower().find(model_id.lower().split("-versatile")[0].lower())
            if idx == -1:
                # try shorter prefix
                short = "-".join(model_id.split("-")[:3]).lower()
                idx = text.lower().find(short)
            if idx == -1:
                results[model_id] = {"in": None, "out": None}
                continue
            # Extract surrounding text ± 500 chars
            snippet = text[max(0, idx - 100) : idx + 600]
            prices = _PRICE_RE.findall(snippet)
            if len(prices) >= 2:
                results[model_id] = {"in": float(prices[0]), "out": float(prices[1])}
            elif len(prices) == 1:
                results[model_id] = {"in": float(prices[0]), "out": None}
            else:
                results[model_id] = {"in": None, "out": None}
    except Exception as exc:
        print(f"  [WARN] Price scraping failed: {exc}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. Model verification
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_MSG = "You are a helpful assistant."
USER_MSG = "Say the single word: VERIFIED"

_TOKENS_PER_MTOK = 1_000_000


def _cost_usd(in_tokens: int, out_tokens: int, price_in: float | None, price_out: float | None) -> float | None:
    if price_in is None or price_out is None:
        return None
    return (in_tokens * price_in + out_tokens * price_out) / _TOKENS_PER_MTOK


def verify_model(client: "Groq", model_id: str, alias: str, prices: dict) -> bool:
    print(f"\n  Testing model '{alias}' ({model_id}) ...")
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": USER_MSG},
            ],
            temperature=0,
            max_tokens=10,
        )
    except Exception as exc:
        print(f"  ✗ FAILED: {exc}")
        return False

    latency = time.perf_counter() - t0
    content = resp.choices[0].message.content
    in_tok = resp.usage.prompt_tokens
    out_tok = resp.usage.completion_tokens
    model_prices = prices.get(model_id, {})
    cost = _cost_usd(in_tok, out_tok, model_prices.get("in"), model_prices.get("out"))
    cost_str = f"${cost:.6f}" if cost is not None else "cost unknown (prices not scraped)"

    print(f"  ✓ Response : {content!r}")
    print(f"    Tokens   : {in_tok} in / {out_tok} out")
    print(f"    Latency  : {latency:.2f}s")
    print(f"    Est. cost: {cost_str}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 3. Write prices back to config.yaml
# ─────────────────────────────────────────────────────────────────────────────

def _update_config_prices(prices: dict[str, dict]) -> None:
    cfg_path = EXPERIMENTS_DIR / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    today = date.today().isoformat()
    updated = False
    for alias, mcfg in cfg.get("models", {}).items():
        model_id = mcfg.get("id", "")
        p = prices.get(model_id, {})
        if p.get("in") is not None:
            mcfg["price_in_per_mtok"] = p["in"]
            mcfg["price_out_per_mtok"] = p.get("out")
            mcfg["price_date"] = today
            updated = True

    if updated:
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"\n  config.yaml updated with prices (date: {today}).")
    else:
        print(
            "\n  [WARN] Could not parse prices from the Groq pricing page.\n"
            "  Visit https://groq.com/pricing/ and fill in config.yaml manually:\n"
            "    price_in_per_mtok: <USD per million input tokens>\n"
            "    price_out_per_mtok: <USD per million output tokens>\n"
            "    price_date: " + today
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Phase 0 — Setup verification")
    print("=" * 60)

    # API key
    try:
        api_key = groq_api_key()
        print(f"\n[1/3] GROQ_API_KEY found (len={len(api_key)}) ✓")
    except EnvironmentError as e:
        print(f"\n[1/3] ✗ {e}")
        sys.exit(1)

    # Scrape prices
    print("\n[2/3] Scraping live prices from groq.com/pricing/ ...")
    prices = _scrape_prices()
    for model_id, p in prices.items():
        pin = f"${p['in']}/Mtok" if p.get("in") else "?"
        pout = f"${p['out']}/Mtok" if p.get("out") else "?"
        print(f"  {model_id}: in={pin}, out={pout}")

    _update_config_prices(prices)

    # Verify each model
    print("\n[3/3] Sending test calls to each model ...")
    client = Groq(api_key=api_key)
    models_cfg = CFG.get("models", {})
    results: dict[str, bool] = {}
    for alias, mcfg in models_cfg.items():
        ok = verify_model(client, mcfg["id"], alias, prices)
        results[alias] = ok

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_ok = True
    for alias, ok in results.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {alias} ({models_cfg[alias]['id']})")
        if not ok:
            all_ok = False

    if all_ok:
        print("\nPhase 0 acceptance check: PASSED ✓")
        print("Prices recorded in config.yaml (verify manually if scraping failed).")
    else:
        print("\nPhase 0 acceptance check: FAILED ✗")
        print("Fix the failing models before proceeding to Phase 1.")
        sys.exit(1)


if __name__ == "__main__":
    main()

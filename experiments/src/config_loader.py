"""
experiments/src/config_loader.py
Loads config.yaml and .env once; imported by all other modules.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(dotenv_path: Path | str | None = None) -> None:  # type: ignore
        """Fallback .env reader if python-dotenv is not installed in the active environment."""
        if dotenv_path is None:
            return
        p = Path(dotenv_path)
        if not p.is_file():
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass

# ── Locate repo ────────────────────────────────────────────────────────────────
EXPERIMENTS_DIR = Path(__file__).parent.parent.resolve()
REPO_ROOT = EXPERIMENTS_DIR.parent

# ── Load .env (GROQ_API_KEY lives here) ───────────────────────────────────────
_env_path = EXPERIMENTS_DIR / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    # Fall back to the backend .env so dev doesn't need to duplicate the key
    load_dotenv(REPO_ROOT / "backend" / ".env")


def load_config() -> dict:
    cfg_path = EXPERIMENTS_DIR / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def groq_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key or key == "your_groq_api_key_here":
        raise EnvironmentError(
            "GROQ_API_KEY is not set or is still the placeholder value.\n"
            "Copy experiments/.env.example to experiments/.env and fill in your key."
        )
    return key


# Singleton — callers do: from src.config_loader import CFG
CFG = load_config()

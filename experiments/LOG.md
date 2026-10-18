# Experiment Decision Log

> Append-only. Format: `## YYYY-MM-DD — <what>` then bullet points.

---

## 2026-10-18 — Phase 0: Setup

- **Branch:** `experiment` (branched from `main`).
- **Working directory:** `experiments/` at repo root; no existing app code modified.
- **Goal of Phase 0:** scaffold directory skeleton, create venv, verify Groq key + model IDs, record live prices in `config.yaml`.
- **Python target:** 3.11 (matches backend `runtime.txt`).
- **Groq models to verify:** `llama-3.3-70b-versatile` (large), `llama-3.1-8b-instant` (small). Will also probe one mid-size candidate; decision on whether to include `mid` deferred until rate-limit data in Phase 4.
- **Embedding model:** keeping `sentence-transformers/all-MiniLM-L6-v2` (same as app config) for consistency; experiments use `fastembed` directly, not the full app stack.
- **Prices:** to be scraped from Groq pricing page and written into `config.yaml` with today's date. Not hardcoded from memory.
- **Random seed:** 42 throughout (splits, bootstrap).
- **Temperature:** 0 for all LLM calls.
- **Prompt version:** `v1` initially; bump to `v2` if prompt is changed post-pilot (cache key includes version).

### Phase 0 completion decisions (2026-10-18)
- **Model IDs changed (VERIFIED):** `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` are no longer available on this Groq account. Live model list: `allam-2-7b`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `openai/gpt-oss-safeguard-20b`, `qwen/qwen3.8-27b`, whisper models, canopylabs models.
- **Chosen models:**
  - `large` → `openai/gpt-oss-120b` (120B params, highest capability)
  - `small` → `openai/gpt-oss-20b` (20B params, fast/cheap)
- **Prices (manually verified 2026-10-18, source: console.groq.com/docs/models):** large: $0.15/$0.60 per Mtok (in/out); small: $0.075/$0.30 per Mtok.
- **httpx pinned to `>=0.27,<0.28`:** groq==0.11.0 passes `proxies=` kwarg removed in httpx 0.28+.
- **Groq pricing page is a React SPA** — automated scraping returns no prices; manual entry used and documented.
- **Phase 0 acceptance check:** PASSED. Both models returned a valid response. Unit tests: 22/22 passed.
- **Next:** Phase 1 — download CUAD, verify skill labels, create stratified split.


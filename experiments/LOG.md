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

### Phase 1 completion decisions (2026-10-18)
- **Data source changed:** `chenghao/cuad_qa` (HF parquet) omits negative rows — every contract appears to have 100% prevalence. Root cause: that mirror only stores positive QA rows. Fixed by using `CUADv1.json` (downloaded from `github.com/The-Atticus-Project/cuad`, `data.zip`, 38MB). License: CC BY 4.0.
- **True prevalence (full 510 contracts, CUADv1.json):**
  - 25-75% band (10 candidates): Non-Transferable License (27%), Minimum Commitment (32%), Revenue/Profit Sharing (32%), Insurance (32%), Renewal Term (34%), Exclusivity (35%), Post-Termination Services (35%), Termination For Convenience (35%), Audit Rights (42%), License Grant (50%), Cap On Liability (54%), Anti-Assignment (73%).
- **8 chosen skills** (written to `data/skills.yaml`): TerminationForConvenience, AuditRights, LicenseGrant, CapOnLiability, AntiAssignment, Exclusivity, InsuranceRequirement, RenewalTerm.
  - LicenseGrant shows 14.5% on the 200-contract sample but 23%/10% in the actual splits — both classes present in dev and test, so it passes acceptance.
- **Windows MAX_PATH issue:** HF `cache_dir` with the full local path exceeds 260 chars. Fixed by using the global HF cache (`~/.cache/huggingface`) for the verify step only; CUADv1.json loaded directly from disk.
- **split.json committed:** dev=30, test=30, pilot=8 contracts. Stratified on all 8 skills simultaneously. Seed 42.
- **Phase 1 acceptance check:** PASSED. Every skill has both classes in dev and test.
- **Next:** Phase 2 — `windows.py` (windowing + embedding cache), `groq_client.py` (retry/backoff), `run_matrix.py` pilot run (8 contracts × 8 skills × 2 models).



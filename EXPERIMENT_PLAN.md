# Experiment Plan: Skill-Aware Model Routing on Legal Contract Analysis

> **Audience:** any AI coding agent (or human) picking this up cold.
> **Status:** PLAN ONLY. Nothing in this document has been run yet. Do not report results that were not actually produced.
> **Last updated:** 17 Oct 2026

---

## 0. Instructions for the agent executing this plan

1. **Read sections 1-5 fully before writing code.** They explain *why* and fix the key design decisions.
2. **Work only inside a new `experiments/` directory** at the repo root until Phase 8. Do not change existing app behaviour before then.
3. **Never fabricate results, numbers, or citations.** If a step cannot be run (no API key, rate limit, dataset unreachable), stop, say so, and propose the smallest workaround.
4. **Keep a decision log** at `experiments/LOG.md`: date, what you did, what you decided, why. Append only.
5. **Secrets:** API keys live in `.env` only. Never commit them. Never print them.
6. **Verify anything marked `VERIFY`** against the live source (dataset page, Groq docs/pricing) before relying on it. Prices, model IDs and dataset field names change.
7. **Every phase ends with an acceptance check** (listed per phase). Do not start the next phase until it passes. Report the outcome to the user.
8. **Be reproducible:** fixed random seeds, `temperature=0`, cached API calls, pinned prompt versions.
9. Prefer simple and correct over clever. This is a small, honest experiment, not a paper reproduction.

---

## 1. Background and goal

### 1.1 The person
A Master's student applying for a Research Assistant (RA) position with a professor who co-authored the paper **"SkillOrchestra: Learning to Route Agents via Skill Transfer"** (arXiv 2602.19672, Feb 2026; Wang, Ming, Ke, Joty, Albarghouthi, Sala).

The student has built an existing project, the **Legal Intelligence Platform** (this repo): a LangGraph multi-agent pipeline for contract analysis. The goal of this plan is to turn that *engineering project* into a *small research testbed* that speaks directly to the paper's ideas, so the student can show up with **evidence**, not only a demo.

### 1.2 The paper in 8 lines (what you need to know)
- Multi-agent / multi-model systems need an **orchestrator** deciding *what to do* (mode: search/code/answer) and *which model/agent does it*.
- Query-level routers decide once per query; RL-trained orchestrators (Router-R1, ToolOrchestra) are expensive and suffer **routing collapse**: they call one big, costly model almost always (e.g., 98% of calls to Llama-3.1-70B).
- **SkillOrchestra's idea:** learn a **Skill Handbook** from execution traces instead of training a policy end-to-end. It has: (i) mode-level insights, (ii) a registry of fine-grained **skills**, (iii) **agent profiles** = per-skill success estimates + cost.
- Success per (agent, skill) is modelled as a **Beta distribution**: `alpha += successes`, `beta += failures`; competence = `alpha / (alpha + beta)`.
- At runtime: identify the active skills, then pick the agent maximizing
  `sum_skill( weight * competence ) - lambda * cost`.
- Handbook **granularity** should match orchestrator capacity (weaker orchestrators do better with coarser skills).
- Results are reported as **accuracy vs. cost Pareto plots**.
- Evaluated on QA, math and FRAMES. **Not on legal text.** That is the gap this experiment touches.

### 1.3 The existing project (facts verified from the code)
Repo root contains `backend/`, `frontend/`, `database/`. Relevant facts:

| Fact | Location |
|---|---|
| Pipeline is a fixed LangGraph graph; only branch is `single` vs `comparison` | `backend/app/graph.py` |
| **All** LLM calls go through one function using one model setting | `backend/app/utils/llm_client.py` (`chat_completion`, `chat_completion_json`) |
| Default model: `llama-3.3-70b-versatile` (Groq) | `backend/app/config.py`, `.env.example` |
| Document Intelligence agent extracts 7+ things in **one** prompt | `backend/app/agents/document_intelligence_agent.py`, prompts in `backend/app/utils/prompts.py` |
| Document text is **truncated to 12,000 characters** before the LLM call | `MAX_DOCUMENT_CHARS = 12000` in the agent above |
| Compliance score is deterministic Python, depends on LLM `present` flags | `backend/app/agents/compliance_agent.py` |
| Overall risk = max severity across all findings (conservative, may over-escalate) | `backend/app/agents/risk_agent.py` |
| RAG: pgvector, `fastembed`, chunk size 800 chars / overlap 120, top-k 5 | `backend/app/rag/*`, `config.py` |
| Python deps incl. `groq==0.11.0`, `tenacity`, `langgraph==0.2.28` | `backend/requirements.txt` |
| README says "9-agent" but code has 8 agent modules | `README.md` |

**Honest framing:** the current system never *decides* which model to use. It is the "always call the big model" baseline by construction. That is exactly what makes it a good testbed.

---

## 2. Research question and hypotheses

**RQ.** On legal contract clause-detection tasks, can per-skill competence/cost profiles (a miniature of SkillOrchestra's agent profiles) route requests between a large and a small LLM with **lower cost and comparable accuracy** versus always using the large model?

**Hypotheses (all falsifiable):**
- **H1 (skill heterogeneity):** the small model's accuracy gap to the large model varies substantially across clause types (small is fine on some, bad on others).
- **H2 (profile-based routing helps):** a router built from dev-set profiles achieves a better accuracy-cost trade-off than the best single fixed model on held-out contracts.
- **H3 (granularity/uncertainty matters):** routing on skill profiles with small samples is noisy; using a *lower credible bound* (conservative) rather than the mean improves reliability.
- **H4 (retrieval confound):** some "model failures" are actually retrieval failures. Measuring retrieval recall separately changes the conclusion for some skills.

**Any outcome is publishable in the meeting** as long as it is analysed honestly (see section 12).

---

## 3. Scope

### In scope
- Clause **presence detection** (is clause type X present in this contract?) for ~8 clause types ("skills").
- Two or three Groq-hosted models of different sizes.
- Per-(model, skill) profiles with Beta estimates, cost and latency.
- A profile-based router + baselines, evaluated on **held-out contracts** by replaying stored outputs.
- Accuracy-vs-cost Pareto plot and a short written report.
- (Late phase) integrate the router into the app's `llm_client`.

### Out of scope (do NOT do these)
- Reimplementing SkillOrchestra's automatic skill discovery/refinement/handbook-selection.
- RL training of any kind.
- Claiming results "match" or "reproduce" the paper.
- Fine-tuning models.
- Production hardening of the app.

### Claims policy (very important)
Allowed: "I tested the core idea (per-skill competence-and-cost routing) at small scale on legal clause detection."
**Not allowed:** "I implemented SkillOrchestra", "production-grade", "outperforms X" without the numbers and CIs to back it.

---

## 4. Data: CUAD

**CUAD** (Contract Understanding Atticus Dataset, The Atticus Project): ~510 commercial contracts annotated by lawyers with 41 clause categories. `VERIFY` current size, license (believed CC BY 4.0) and download location (Atticus Project site, Zenodo, or Hugging Face e.g. `theatticusproject/cuad-qa`; check which format you get and its field names).

### 4.1 IMPORTANT correction: CUAD does not label everything the app extracts
To the best of current knowledge, CUAD has **no** categories for *Confidentiality*, *Indemnification* or *Payment terms* (the app extracts these). `VERIFY` by listing the dataset's category names in Phase 1. Do **not** build the evaluation around clause types that lack gold labels.

### 4.2 Candidate skills (CUAD-native)
Pick **8** whose prevalence in the sample is between roughly **25% and 75%** (so accuracy is not dominated by class imbalance). Candidates (names `VERIFY`):

`Governing Law` (usually near-always present, may be a ceiling case), `Anti-Assignment`, `Cap On Liability`, `Uncapped Liability` (often rare), `Termination For Convenience`, `Audit Rights`, `Insurance`, `License Grant`, `Non-Compete`, `Exclusivity`, `Liquidated Damages`, `Renewal Term`, `IP Ownership Assignment`, `Change Of Control`, `Post-Termination Services`, `Minimum Commitment`, `Revenue/Profit Sharing`, `Warranty Duration`.

### 4.3 Mapping to the app's own fields (for the integration phase)

| App field | CUAD label | Match quality |
|---|---|---|
| `governing_law` | Governing Law | direct |
| `liability_clause` | Cap On Liability / Uncapped Liability | approximate (either present) |
| `termination_clause` | Termination For Convenience | partial (CUAD is narrower) |
| `confidentiality_clause`, `indemnity_clause`, `payment_terms` | none | no gold labels, exclude from scored eval |

### 4.4 Gold label definition
For contract *c* and skill *s*: `gold_present = 1` if CUAD has at least one non-empty annotated span for *s* in *c*, else `0`. Store the gold span text(s) too (needed for the retrieval-recall diagnostic).

### 4.5 Splits (split by CONTRACT, never by row)
- Sample **60 contracts** (stratified so each chosen skill has both classes in both splits). Seed = 42.
- `dev` = 30 contracts (used to build profiles / choose lambda).
- `test` = 30 contracts (used only for final router evaluation; never touch during development).
- Plus a `pilot` = 8 contracts taken from `dev` for debugging the harness cheaply.
- Commit the split file `experiments/data/split.json` so it is fixed.

If budget/rate limits bite, shrink to 40 contracts (20/20) rather than dropping the dev/test separation.

---

## 5. Experimental design

### 5.1 Models (the "agent pool")
Use Groq since the app already depends on it. `VERIFY` model IDs and availability in the Groq docs before starting.

| Alias | Model ID (verify) | Role |
|---|---|---|
| `large` | `llama-3.3-70b-versatile` | current app default |
| `small` | `llama-3.1-8b-instant` | cheap candidate |
| `mid` (optional) | any mid-size Groq model that is currently available | adds a third option; drop if rate limits are tight |

Pricing: read current per-token prices from Groq's pricing page and store them in `experiments/config.yaml` (`price_in_per_mtok`, `price_out_per_mtok` per model) with the retrieval date. **Do not hardcode prices from memory.** Also record raw tokens and latency, so conclusions survive price changes.

### 5.2 Task format (one "skill call")
One call = (model, contract, skill). The model sees retrieved context from the contract plus the skill definition and must return JSON:

```json
{"present": true, "evidence_quote": "<=25 words copied from the context, or null", "confidence": "low|medium|high"}
```

Prompt rules (version it as `prompt_v1`, store in `experiments/prompts/`):
- Define the skill in 1-2 plain sentences (write your own short definitions; do not copy dataset text verbatim).
- "Answer only from the provided context. If the context does not contain such a clause, return present=false and evidence_quote=null."
- `temperature=0`, JSON mode, identical prompt for every model.

### 5.3 Context construction (why retrieval)
Full CUAD contracts are far longer than the app's 12,000-character truncation, and sending them whole to every call is too expensive for rate limits. So:
1. Split each contract into windows of ~3,000 characters with ~300 overlap.
2. Embed windows with the app's existing embedding model (`fastembed`, same one as `config.py`), cached on disk.
3. For each skill, embed a short skill query and retrieve **top-k = 3** windows (k configurable).
4. Send those windows as the context.

Retrieval is shared by all models, so it is a common-mode factor; measure it separately (5.6, H4).

*Fallback (simplest):* replicate the app's current behaviour (first 12,000 chars only). Report clearly that this lowers recall for clauses appearing late in a contract. Use the fallback only if retrieval blocks progress.

### 5.4 Run matrix and call budget
`calls = contracts x skills x models` = 60 x 8 x 3 = **1,440** (or 960 with two models). Each call is roughly 1-2k tokens. Groq free-tier rate limits are tight (`VERIFY` at the Groq rate-limits docs): implement a global rate limiter, exponential backoff (`tenacity` is already a dependency), and **resumable runs with a disk cache** keyed by `hash(model, contract_id, skill, prompt_version, k)`.

### 5.5 What to log for every call (JSONL or CSV, one row per call)
`run_id, ts, split, contract_id, skill, model, prompt_version, k, retrieved_window_ids, gold_present, pred_present, correct, parse_ok, evidence_quote, quote_found_in_context, confidence, prompt_tokens, completion_tokens, cost_usd, latency_s, error`

- `parse_ok=false` (bad JSON) counts as **incorrect** and is reported separately (reliability differs by model size).
- `quote_found_in_context` = the quote is a substring of the retrieved text after whitespace normalization. This measures **evidence hallucination**, a key concern for the app's "evidence-backed" claim.

### 5.6 Diagnostics
- **Retrieval recall (only for `gold_present=1`):** does any retrieved window overlap the gold span (fuzzy substring match)? Report per skill. If retrieval recall is low for a skill, model comparisons on that skill are not meaningful; flag it.
- **Prevalence per skill per split** (print it; sanity-check class balance).

### 5.7 Metrics
Primary: **balanced accuracy** per (model, skill), plus macro-average across skills. Secondary: accuracy, F1 (positive class), parse-failure rate, quote-found rate, mean tokens, mean cost, mean latency.
Uncertainty: **Wilson 95% intervals** for proportions and **bootstrap over contracts** (1,000 resamples) for aggregate metrics. With ~30 test contracts the intervals will be wide; **report them and do not over-interpret small differences.**

---

## 6. Profiles and routing

### 6.1 Agent profile (miniature of the paper's Definition 4.2)
Computed on **dev only**, per (model m, skill s):
- `successes`, `failures` (success = `correct`)
- Beta posterior with a uniform prior: `alpha = 1 + successes`, `beta = 1 + failures`
- `competence_mean = alpha / (alpha + beta)`
- `competence_lcb = 10th percentile of Beta(alpha, beta)` (use `scipy.stats.beta.ppf(0.10, alpha, beta)`)
- `cost_per_call` (mean USD), `tokens_per_call`, `latency_per_call`

Save as `experiments/results/profiles.json` and a human-readable `profiles.csv`. This table is a key artifact for the meeting.

### 6.2 Routers to evaluate (all choose a model *per skill*)
Let `cost_norm(m) = mean cost per call of m / max over models`.

| Router | Rule |
|---|---|
| `always_large` | baseline |
| `always_small` | baseline |
| `random` | uniform random model per call, seed fixed (average over 20 seeds) |
| `profile_mean(lambda)` | `argmax_m  competence_mean(m,s) - lambda * cost_norm(m)` |
| `profile_lcb(lambda)` | same but uses `competence_lcb` (H3) |
| `cascade` (FrugalGPT-style, the paper compares against FrugalGPT) | run `small`; if `confidence != high` or `parse_ok=false` or `present=true` with `quote_found_in_context=false`, escalate to `large`; cost = sum of both calls when escalated |
| `oracle_skill` (upper bound, uses **test** labels) | best model per skill on test; label clearly as an unattainable bound |

Sweep `lambda in {0, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0}`. Choose any "recommended lambda" using **dev only**.

### 6.3 Evaluation by replay (no new API calls)
Because every model was run on every test (contract, skill), evaluating a router = look up the stored outcome of the model it would have picked. Compute test balanced accuracy and total/mean cost. This makes the lambda sweep and all baselines free.

### 6.4 Pareto plot
x = mean cost per call (or total USD on test), y = macro balanced accuracy on test. Plot: always_large, always_small, (always_mid), random, profile_mean sweep, profile_lcb sweep, cascade, oracle_skill (dashed/grey). Draw the Pareto frontier. Add bootstrap error bars for the main points. Output `experiments/results/pareto.png` (and `.csv` of plotted points).

---

## 7. Repository layout to create

```
experiments/
  README.md                 # how to run, in 10 lines
  LOG.md                    # decision log (append only)
  config.yaml               # models, prices (+date), k, seeds, skills, paths
  requirements.txt          # experiment-only deps (scipy, pandas, matplotlib, tqdm, pyyaml, datasets?)
  data/
    raw/                    # downloaded CUAD (gitignored)
    split.json              # committed
    skills.yaml             # skill name, short definition, retrieval query, CUAD label
  prompts/
    skill_presence_v1.txt
  src/
    load_cuad.py            # parse CUAD -> contracts + gold per skill
    windows.py              # windowing + embeddings + retrieval (cached)
    groq_client.py          # model-parameterized call, retry, rate limit, cost calc
    run_matrix.py           # resumable runner with disk cache
    profiles.py             # Beta posteriors, LCB, cost
    routers.py              # all routers in 6.2
    evaluate.py             # replay + metrics + bootstrap
    plots.py                # Pareto + per-skill heatmap
  results/
    raw_calls.jsonl         # every call
    profiles.json / .csv
    router_eval.csv
    pareto.png
    skill_heatmap.png       # model x skill balanced accuracy
  REPORT.md                 # final write-up (Phase 7)
  tests/                    # unit tests for metrics/routers (small)
```

---

## 8. Phases, tasks, acceptance checks

### Phase 0: Setup (0.5-1 day)
- [ ] Create `experiments/` skeleton and venv; install deps (`groq`, `pandas`, `scipy`, `matplotlib`, `pyyaml`, `tqdm`, `tenacity`, `fastembed`, `datasets` if using HF).
- [ ] Confirm Groq key works with a 1-token call for each model; confirm model IDs.
- [ ] Record prices (+ date) into `config.yaml`.
- **Accept:** a script prints one successful JSON response from each model and its computed cost.

### Phase 1: Data (1 day)
- [ ] Download CUAD; document source URL/version/license in `LOG.md`.
- [ ] List all category names; confirm which candidate skills exist (section 4).
- [ ] Build `load_cuad.py`: contracts, text, `gold_present`, gold spans per skill.
- [ ] Compute prevalence per skill on a 200-contract sample; choose 8 skills (25-75% band); write `skills.yaml` with short self-written definitions.
- [ ] Create the 60-contract stratified sample and `split.json` (30 dev / 30 test, plus 8-contract pilot subset of dev).
- **Accept:** a printed table of prevalence per skill per split; every skill has both classes in dev and test; `split.json` committed.

### Phase 2: Harness (1-2 days)
- [ ] `windows.py`: windowing, cached embeddings, top-k retrieval; log retrieval recall on gold spans.
- [ ] `groq_client.py`: `call(model, system, user)` with JSON mode, retry/backoff, rate limiter, token counting, cost, latency.
- [ ] `prompts/skill_presence_v1.txt` + robust JSON parsing (never crash; set `parse_ok=false`).
- [ ] `run_matrix.py`: resumable, cached, writes `raw_calls.jsonl`.
- **Accept:** running the pilot (8 contracts x 8 skills x all models) twice yields identical results the second time from cache with zero new API calls; no unhandled exceptions; parse failure rate printed.

### Phase 3: Pilot sanity review (0.5 day)
- [ ] Manually inspect ~20 pilot rows (mix of correct/incorrect). Are labels sensible? Are retrieval windows plausible? Are quotes real?
- [ ] Fix prompt or retrieval **only** using pilot/dev data; bump `prompt_version` if the prompt changes (invalidates cache by design).
- **Accept:** user is shown 5 example rows plus aggregate pilot accuracy per model and agrees to proceed.

### Phase 4: Full run (1-2 days, mostly waiting on rate limits)
- [ ] Run the full matrix (dev + test, all models).
- [ ] Verify row counts = contracts x skills x models; list any missing/errored rows and re-run them.
- **Accept:** completeness report shows 0 missing rows (or lists exceptions with reasons).

### Phase 5: Profiles (0.5 day)
- [ ] Build profiles from **dev only** (6.1).
- [ ] Produce `skill_heatmap.png` (model x skill competence) and `profiles.csv`.
- [ ] Write 3-5 bullet observations, each backed by a number (tests H1).
- **Accept:** table + heatmap exist; no test-split rows were used.

### Phase 6: Routers and evaluation (1 day)
- [ ] Implement routers (6.2), replay evaluation (6.3), bootstrap CIs.
- [ ] Choose "recommended lambda" from dev only; then report test results.
- [ ] Produce `pareto.png` and `router_eval.csv`.
- **Accept:** unit tests for metric and router code pass; plot shows all baselines; CIs reported.

### Phase 7: Analysis and report (1 day)
- [ ] Write `REPORT.md` (structure in section 11) with honest limitations.
- [ ] Evaluate each hypothesis H1-H4: supported / not supported / inconclusive, with evidence.
- **Accept:** every number in the report traces to a file in `results/`.

### Phase 8: Integrate into the app (optional, 1-2 days; only after Phase 7)
- [ ] Add optional `model: str | None = None` and `skill: str | None = None` parameters to `chat_completion_json` in `backend/app/utils/llm_client.py` (default behaviour unchanged).
- [ ] Add `backend/app/routing/router.py` that loads `profiles.json` and returns a model for a given skill, with a feature flag `ENABLE_SKILL_ROUTING=false` by default.
- [ ] Note the design constraint: the Document Intelligence agent extracts many fields in **one** call, so per-skill routing needs that call split into per-skill calls. Implement as an **opt-in alternate path**, not a rewrite.
- [ ] Log the chosen model per call into the pipeline state (`errors`-style soft-fail; never crash on missing profiles, fall back to the default model).
- [ ] Add tests; confirm all existing 38 tests still pass.
- **Accept:** with flag off, behaviour and tests are unchanged; with flag on, a demo contract shows the per-skill model choices in the report/state.

### Phase 9: Repo hygiene and meeting materials (1 day)
- [ ] README: fix "9-agent" wording (8 agent modules), remove "production-grade", add the screenshots (currently placeholders), add a **"Limitations and Research Directions"** section, add a link to `experiments/REPORT.md`.
- [ ] Prepare a 1-page summary and 3-4 slides: (1) the app, (2) the gap, (3) the experiment + Pareto plot, (4) findings, limitations and questions for the professor.
- [ ] Draft a short email to the professor (3-4 sentences, link to repo).
- **Accept:** the user can explain every figure and every number without notes.

---

## 9. Suggested timeline (about 2 weeks, part-time)

| Days | Work |
|---|---|
| 1-2 | Phase 0-1 |
| 3-4 | Phase 2-3 |
| 5-7 | Phase 4 (runs) + start Phase 5 |
| 8-9 | Phase 5-6 |
| 10-11 | Phase 7 |
| 12-14 | Phase 8 (optional) + Phase 9 |

If time is short: do Phases 0-7 only. Phase 8 is a nice-to-have.

---

## 10. Risks and pitfalls

| Risk | Mitigation |
|---|---|
| Rate limits / free-tier throttling | cache, backoff, resumable runs, smaller sample (40 contracts) |
| Class imbalance inflates accuracy | pick skills in the 25-75% band; use balanced accuracy |
| Data leakage (tuning on test) | split by contract; test untouched until Phase 6; lambda chosen on dev |
| Retrieval, not the model, is the bottleneck | measure retrieval recall (5.6); flag low-recall skills |
| Tiny samples, wide CIs | report intervals; avoid strong claims; use LCB router |
| Prompt changes silently alter results | version prompts; cache key includes prompt version |
| Price changes | log tokens and latency too; store price date |
| Models deprecated mid-project | pin IDs in config; note date; re-verify at start |
| Skills lacking gold labels | only use CUAD-native categories (section 4) |
| Over-claiming | follow the claims policy (section 3) |
| Dataset license / redistribution | do not commit raw CUAD; commit only split IDs and code |

---

## 11. REPORT.md outline

1. Question and motivation (link to SkillOrchestra, 5 lines)
2. Setup: data, skills, models, retrieval, prompts, metrics (with versions/dates)
3. Results: profile heatmap; table of balanced accuracy +/- CI per model per skill; parse-failure and quote-found rates
4. Routing results: Pareto plot, router table, chosen lambda
5. Hypotheses H1-H4: verdicts with evidence
6. Limitations (small N, one dataset, presence-only task, shared retrieval, only Llama-family models, no automatic skill discovery, price snapshot)
7. What would come next (research directions)
8. Reproducibility: commands, seeds, file list

---

## 12. How to interpret outcomes (and what to say)

| Outcome | Honest takeaway |
|---|---|
| Router matches `always_large` accuracy at clearly lower cost | Supports the core idea in a new domain at small scale; mention CIs and N |
| Small model is competitive on some skills only | Strong H1 evidence; the per-skill heterogeneity is exactly the paper's premise |
| Small model fails nearly everywhere | Negative but useful: in high-stakes legal text, cost-only routing is risky; motivates a **risk-aware** router constraint |
| Router no better than best fixed model | Likely small-sample noise or low skill heterogeneity; discuss profile estimation error (link to the paper's granularity argument) |
| Retrieval recall is low for some skills | The bottleneck is retrieval, not model choice; important systems insight |
| Cascade beats profile routing | Report it honestly; it is the FrugalGPT-style baseline the paper also compares against |

---

## 13. Honest talking points for the professor meeting

- "My app is the *always-big-model* baseline by design. I wanted to test your question in a new domain."
- "I measured per-skill competence and cost for two or three models on CUAD clause detection and built a small profile-based router. This is a miniature of your agent profiles, not a reimplementation."
- Show: heatmap, Pareto plot, one failure analysis (including a retrieval-vs-model failure case).
- Be ready to critique the paper politely: skill discovery needs strong models (cost), exploratory traces are an upfront cost, evaluation is on QA/math/FRAMES, "700x/300x" refers to *learning* cost.
- Good questions to ask: "How does the handbook behave when a model's behaviour drifts after an update?", "Would risk-aware constraints fit the framework for high-stakes domains?", "Does handbook transfer across domains like legal?"

---

## 14. Final deliverables checklist

- [ ] `experiments/results/profiles.csv` and `profiles.json`
- [ ] `experiments/results/skill_heatmap.png`
- [ ] `experiments/results/pareto.png` and `router_eval.csv`
- [ ] `experiments/REPORT.md` (every number traceable)
- [ ] `experiments/LOG.md` (decisions)
- [ ] Updated top-level `README.md` (accurate claims, limitations section, screenshots)
- [ ] (Optional) feature-flagged router integration with passing tests
- [ ] 1-page summary + short slides + draft email

---

## 15. Glossary

- **Orchestrator:** the controller deciding what to do and which agent/model does it.
- **Mode:** type of action (search, code, answer). In this project: pipeline stages.
- **Skill:** a fine-grained capability. Here: detecting one clause type.
- **Agent profile:** per-skill success estimates plus cost for a given model.
- **Routing collapse:** a router that almost always picks one (big, costly) model.
- **Beta posterior:** distribution over a success probability; mean = alpha / (alpha + beta).
- **LCB:** lower credible bound, a conservative competence estimate.
- **Pareto frontier:** points where accuracy cannot improve without raising cost.
- **Cascade:** try the cheap model first and escalate only when uncertain.
- **Replay evaluation:** evaluating a router by looking up stored model outputs instead of making new calls.
- **CUAD:** lawyer-annotated contract dataset with 41 clause categories.
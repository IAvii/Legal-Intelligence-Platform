how to run the experiment, in 10 lines
--------------------------------------

1. Create venv:  python -m venv .venv && .venv\Scripts\activate
2. Install deps:  pip install -r requirements.txt
3. Copy env:      copy .env.example .env  (fill in GROQ_API_KEY)
4. Phase 0 check: python src/verify_setup.py
5. Phase 1 data:  python src/load_cuad.py        (downloads + processes CUAD)
6. Phase 2 pilot: python src/run_matrix.py --split pilot
7. Phase 4 full:  python src/run_matrix.py --split all
8. Phase 5:       python src/profiles.py
9. Phase 6:       python src/evaluate.py && python src/plots.py
10. Report:        see results/  and  REPORT.md

All API calls are cached; re-running is safe and free after the first run.
Seeds are fixed (42). Temperature is 0. Results are fully reproducible.

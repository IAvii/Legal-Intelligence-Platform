"""
Scratch: compute true CUAD prevalence from CUADv1.json (includes negative rows).
Run from experiments/ directory.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

# CUADv1.json is the original SQuAD-format file with all (contract, question) pairs
CUAD_JSON = Path("data/raw/cuad_extracted/CUADv1.json")

with open(CUAD_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

articles = data["data"]
print(f"Total contracts (articles): {len(articles)}")

skill_stats = defaultdict(lambda: {"present": 0, "absent": 0})

for article in articles:
    title = article["title"]
    for para in article["paragraphs"]:
        for qa in para["qas"]:
            question = qa["question"].strip()
            is_impossible = qa.get("is_impossible", False)
            answers = qa.get("answers", [])
            has_answer = not is_impossible and len(answers) > 0

            if has_answer:
                skill_stats[question]["present"] += 1
            else:
                skill_stats[question]["absent"] += 1

total_contracts = len(articles)
print(f"Total unique questions: {len(skill_stats)}")
print()
print(f"{'Skill':<45} {'Present':>8} {'Absent':>8} {'Prevalence':>12}  Band")
print("-" * 88)

rows = []
for skill, counts in skill_stats.items():
    p = counts["present"]
    a = counts["absent"]
    tot = p + a
    pct = p / tot * 100 if tot else 0
    band = "OK" if 25 <= pct <= 75 else ("HIGH" if pct > 75 else "LOW")
    rows.append((pct, skill, p, a, band))

rows.sort(key=lambda x: x[0])
for pct, skill, p, a, band in rows:
    marker = "  <-- GOOD" if band == "OK" else ""
    print(f"{skill:<45} {p:>8} {a:>8} {pct:>11.1f}%  {band}{marker}")

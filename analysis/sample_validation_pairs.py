#!/usr/bin/env python3
"""Sample question-passage pairs for the expert validation of the LLM relevance judge (amendment 6).

70 distinct pairs, at most one per question, stratified by LLM grade (0-3) and alternating question sets; 10 of them
are shared by both experts, 30 more go to each (40 per expert, 80 judgements). Experts never see the LLM grade.
Writes data/hemophilia-clin/judging_validation/sample.json.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import openpyxl

SEED = "validation-20261109"
PER_GRADE = {0: 18, 1: 17, 2: 18, 3: 17}          # 70 pairs
SHARED_PER_GRADE = {0: 3, 1: 2, 2: 3, 3: 2}       # 10 shared pairs
D = Path("data/hemophilia-clin")


def h(s: str) -> str:
    return hashlib.sha256((SEED + s).encode()).hexdigest()


def llm_grades() -> dict:
    key = json.loads((D / "judging" / "key.json").read_text())
    grades = {}
    for ini in ("MC", "ME"):
        ws = openpyxl.load_workbook(D / "judging" / "returned_20261108" / f"notation_pertinence_{ini}.xlsx")["Notation"]
        for r in ws.iter_rows(min_row=2, values_only=True):
            if r[3]:
                k = key[f"{ini}:{r[3]}"]
                grades[(k["qid"], k["chunk_id"])] = int(r[6])
    return grades


def main():
    grades = llm_grades()
    by_grade = defaultdict(list)
    for (q, c), g in grades.items():
        by_grade[g].append((q, c))
    used_q, chosen = set(), []
    for g in sorted(PER_GRADE):                                  # rarest grades first would also work; order fixed
        cands = sorted(by_grade[g], key=lambda p: h(p[0] + p[1]))
        n, pick = PER_GRADE[g], []
        for q, c in cands:
            if q not in used_q and sum(1 for x in pick if x[0][1] == q[1]) < (n + 1) // 2:   # balance sets A/B
                pick.append((q, c))
                used_q.add(q)
            if len(pick) == n:
                break
        assert len(pick) == n, (g, len(pick))
        chosen += [{"qid": q, "chunk_id": c, "llm_grade": g} for q, c in pick]
    shared, rest = [], []
    for g in sorted(PER_GRADE):
        items = sorted([x for x in chosen if x["llm_grade"] == g], key=lambda x: h("s" + x["qid"]))
        shared += items[: SHARED_PER_GRADE[g]]
        rest += items[SHARED_PER_GRADE[g]:]
    rest.sort(key=lambda x: h("r" + x["qid"]))
    assign = {"MC": shared + rest[0::2], "ME": shared + rest[1::2]}
    for ini in assign:
        assign[ini] = sorted(assign[ini], key=lambda x: h(ini + x["qid"]))   # shuffled order, grades hidden
    out = D / "judging_validation"
    out.mkdir(exist_ok=True)
    (out / "sample.json").write_text(json.dumps({"seed": SEED, "pairs": chosen, "shared": [x["qid"] for x in shared],
                                                "assignment": {k: [x["qid"] + "|" + x["chunk_id"] for x in v] for k, v in assign.items()}}, indent=1))
    print("pairs", len(chosen), "| per expert", {k: len(v) for k, v in assign.items()}, "| shared", len(shared))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""LLM controllers (B1, B2) and LLM-augmented learned controllers (A3+L, C+L) — DEVELOPMENT evaluation.

Same development protocol as controllers.py (locked test split never read; development pool re-split R times into
fit/calibration/evaluation 50/25/25 with the same random generator, hence the same splits as controllers.py).
LLM features come from llm_features.py (frozen prompts 2026-11-16-v1, qwen2.5:7b).

  B1      zero-shot: lexical_weight returned by the model from the query alone, snapped to the nearest action
          (rrf: w_lex; cvx: alpha = 1 - lexical_weight)
  B2      zero-shot: mean over valid samples of w_lex_L (rrf) or alpha_L (cvx), from the graded top-3 of each list,
          snapped to the nearest action; a* if no valid sample
  DAT     baseline of Hsu & Tzeng (2025), dat_features.py: alpha (dense weight) from LLM scores of the top-1 passage of
          each retriever (1 call); cvx: alpha; rrf: w_lex = 1 - alpha, snapped; a* if the reply is invalid (optional --dat)
  A3, C   as in controllers.py (query/retrieval signals only), recomputed here for a paired comparison
  A3+L    A3 with the LLM features appended to the signals
  C+L     C(eps) on A3+L (conformal risk control on calibration, same lambda rule)
No fitting or snapping uses evaluation queries. LLM calls per query: B1 1, B2 3, A3+L and C+L 4 (both prompts).

    python llm_controllers.py results/scifact ../llm_mac/scifact-llm/llm_features.csv [--family rrf] [--dev-only]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from controllers import EPSILONS, SEED, crc_lambda, fit_a3, load, locked_test, scores

LLM_FEATS = ["b1_wlex", "b1_needs_exact", "b1_mode_lexical", "b1_mode_semantic", "b2_alpha_mean", "b2_alpha_sd",
             "b2_wlex_mean", "b2_wlex_sd", "b2_both_zero_share", "b2_valid"]


def num(x, default):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def load_llm(path: Path, qids: list[str]):
    rows = {r["qid"]: r for r in csv.DictReader(open(path))}
    F, b1, b2, b2ok = [], [], [], []
    for q in qids:
        r = rows.get(q, {})
        w1 = num(r.get("b1_wlex"), np.nan)
        ok = num(r.get("b2_valid_samples"), 0) > 0
        F.append([num(w1, 0.5), 1.0 if str(r.get("b1_needs_exact")) == "True" else 0.0,
                  1.0 if r.get("b1_mode") == "lexical" else 0.0, 1.0 if r.get("b1_mode") == "semantic" else 0.0,
                  num(r.get("b2_alpha_mean"), 0.5), num(r.get("b2_alpha_sd"), 0.0), num(r.get("b2_wlex_mean"), 0.5),
                  num(r.get("b2_wlex_sd"), 0.0), num(r.get("b2_both_zero_share"), 1.0), 1.0 if ok else 0.0])
        b1.append(w1)
        b2.append((num(r.get("b2_wlex_mean"), np.nan), num(r.get("b2_alpha_mean"), np.nan)) if ok else (np.nan, np.nan))
    missing = sum(q not in rows for q in qids)
    return np.array(F), np.array(b1), np.array(b2), missing


def snap(target: np.ndarray, vals: np.ndarray, fallback: int) -> np.ndarray:
    a = np.abs(np.nan_to_num(target, nan=-9)[:, None] - vals[None, :]).argmin(1)
    return np.where(np.isnan(target), fallback, a)


def load_dat(path: Path, qids: list[str]) -> np.ndarray:
    rows = {r["qid"]: r for r in csv.DictReader(open(path))}
    return np.array([num(rows.get(q, {}).get("dat_alpha"), np.nan) for q in qids])


def run(res: Path, llm: Path, family: str, dev_only: bool, reps: int, dat: Path | None = None) -> dict:
    qids, acts, M, X, feats = load(res, family)
    F, b1, b2, missing = load_llm(llm, qids)
    da = load_dat(dat, qids) if dat else None
    t_dat = None if da is None else (1 - da if family == "rrf" else da)
    XL = np.hstack([X, F])
    vals = np.array([float(a.split("_")[1][1:]) for a in acts])
    t_b1 = b1 if family == "rrf" else 1 - b1
    t_b2 = b2[:, 0] if family == "rrf" else b2[:, 1]
    pool = np.arange(len(qids)) if dev_only else np.where(~locked_test(qids))[0]
    rng = np.random.default_rng(int(SEED))
    out, lambdas = defaultdict(list), defaultdict(list)
    for _ in range(reps):
        p = rng.permutation(pool)
        n_fit, n_cal = len(p) // 2, len(p) // 4
        fit, cal, ev = p[:n_fit], p[n_fit:n_fit + n_cal], p[n_fit + n_cal:]
        astar = int(M[np.concatenate([fit, cal])].mean(0).argmax())
        out["FIXED"].append(scores(M[ev], np.full(len(ev), astar), astar))
        oracle = np.where(M[ev, astar] >= M[ev].max(1) - 1e-12, astar, M[ev].argmax(1))
        out["ORACLE"].append(scores(M[ev], oracle, astar))
        out["B1"].append(scores(M[ev], snap(t_b1[ev], vals, astar), astar))
        out["B2"].append(scores(M[ev], snap(t_b2[ev], vals, astar), astar))
        if t_dat is not None:
            out["DAT"].append(scores(M[ev], snap(t_dat[ev], vals, astar), astar))
        for tag, Z in (("", X), ("+L", XL)):
            a3 = fit_a3(Z[fit], M[fit], astar)
            a_ev, g_ev, _ = a3(Z[ev])
            out["A3" + tag].append(scores(M[ev], np.where(g_ev > 0, a_ev, astar), astar))
            a_cal, g_cal, _ = a3(Z[cal])
            for eps in EPSILONS:
                lam = crc_lambda(g_cal, a_cal, M[cal], astar, eps)
                lambdas[f"C{tag}(eps={eps})"].append(lam)
                out[f"C{tag}(eps={eps})"].append(scores(M[ev], np.where(g_ev > lam, a_ev, astar), astar))
    # zero-shot B1/B2 on the whole development pool (no fitting involved), paired with every fixed action
    zs = {}
    for name, t in (("B1", t_b1), ("B2", t_b2)) + ((("DAT", t_dat),) if t_dat is not None else ()):
        ch = snap(t[pool], vals, -1)
        valid = ch >= 0
        zs[name] = {"valid_share": float(valid.mean()), "ndcg10": float(M[pool][valid, ch[valid]].mean()),
                    "best_fixed_same_queries": float(M[pool][valid].mean(0).max()),
                    "action_shares": {acts[j]: float((ch[valid] == j).mean()) for j in range(len(acts))}}
    summary = {"corpus": res.name, "family": family, "llm_features": str(llm), "queries_total": len(qids),
               "queries_without_llm_features": missing, "dev_pool": len(pool), "locked_test_untouched": not dev_only,
               "reps": reps, "split_sizes": [len(pool) // 2, len(pool) // 4, len(pool) - len(pool) // 2 - len(pool) // 4],
               "llm_calls_per_query": {"B1": 1, "B2": 3, "A3+L": 4, "C+L": 4, "DAT": 1}, "dat_features": str(dat) if dat else None,
               "dat_invalid": int(np.isnan(da).sum()) if da is not None else None,
               "lambda_hat_inf_share": {k: float(np.mean(np.isinf(v))) for k, v in lambdas.items()},
               "zero_shot_dev_pool": zs, "controllers": {}}
    for name, runs in out.items():
        summary["controllers"][name] = {k: {"mean": float(np.mean([r[k] for r in runs])),
                                            "p2.5": float(np.percentile([r[k] for r in runs], 2.5)),
                                            "p97.5": float(np.percentile([r[k] for r in runs], 97.5))} for k in runs[0]}
    return summary


def show(s: dict) -> None:
    print(f"\n=== {s['corpus']} | {s['family']} | pool {s['dev_pool']}/{s['queries_total']} | fit/cal/eval {s['split_sizes']} "
          f"| {s['reps']} reps | test untouched: {s['locked_test_untouched']} | no LLM features: {s['queries_without_llm_features']}")
    print(f"lambda_hat=+inf share: {s['lambda_hat_inf_share']}")
    print(f"{'controller':<16}{'nDCG@10':>9}{'gain':>9}{'  [95% reps]':>18}{'URisk':>9}{'win%':>7}{'loss%':>7}{'dev%':>7}{'meanLoss':>10}")
    for name, m in s["controllers"].items():
        g = m["gain"]
        print(f"{name:<16}{m['ndcg10']['mean']:>9.4f}{g['mean']:>+9.4f}   [{g['p2.5']:+.3f},{g['p97.5']:+.3f}]"
              f"{m['urisk']['mean']:>+9.4f}{100*m['win']['mean']:>7.1f}{100*m['loss']['mean']:>7.1f}"
              f"{100*m['dev']['mean']:>7.1f}{m['mean_loss']['mean']:>10.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("llm_features")
    ap.add_argument("--family", default="rrf", choices=["rrf", "cvx"])
    ap.add_argument("--dev-only", action="store_true")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--out", help="output directory (default: results)")
    ap.add_argument("--dat", help="dat_features.csv of dat_features.py (adds the DAT baseline)")
    a = ap.parse_args()
    s = run(Path(a.results), Path(a.llm_features), a.family, a.dev_only, a.reps, Path(a.dat) if a.dat else None)
    o = Path(a.out or a.results)
    o.mkdir(parents=True, exist_ok=True)
    (o / f"llm_controllers_dev_{a.family}.json").write_text(json.dumps(s, indent=1))
    show(s)

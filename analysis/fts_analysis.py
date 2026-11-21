#!/usr/bin/env python3
"""FTS lexical retriever and production rule A0 — DEVELOPMENT evaluation (weighted RRF family).

Inputs : server results (utility_matrix_fts.py run with the PostgreSQL FTS re-implementation, amendment 11) in
         ../server_results/<run>/results/<corpus>-fts, and the BM25 runs of the same queries.
For each corpus:
  * in-sample headroom with FTS and with BM25 (same definitions as make_article_figures.headroom);
  * development re-splits identical to controllers.py (same seed, same pool: locked test never read):
    FIXED a*_FTS, A0 (production rule: w_lex = 1 if the conjunctive stage matched, else 0.5), A3, C(eps), ORACLE,
    and FIXED a*_BM25 evaluated on the same evaluation folds (difference between lexical retrievers).
Outputs: results/<corpus>-fts/controllers_dev_rrf.json, results/fts_dev.json, results/fts_dev.txt

    python fts_analysis.py ../server_results/20261117_2021/results
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import controllers as K

HERE = Path(__file__).resolve().parent
W = ["rrf_w0.0", "rrf_w0.25", "rrf_w0.5", "rrf_w0.75", "rrf_w1.0"]
# key, corpus dir, BM25 results dir (relative to HERE/results or to the server results), dev-only
CORPORA = [("Hemo", "hemophilia-dev", "local", True), ("SciFact", "scifact", "local", False),
           ("NFCorpus", "nfcorpus", "local", False), ("TrecCovid", "trec-covid", "server", False),
           ("FiQA", "fiqa", "server", False)]


def matrix(path: Path):
    U = defaultdict(dict)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            U[r["qid"]][r["action"]] = float(r["ndcg10"])
    return U


def headroom(U, qids):
    M = np.array([[U[q][a] for a in W] for q in qids])
    b = int(M.mean(0).argmax())
    flat = M.max(1) - M.min(1) <= 1e-9
    opt = (M[:, b] >= M.max(1) - 1e-9) & ~flat
    return {"n": len(qids), "dense": float(np.mean([U[q]["dense_only"] for q in qids])),
            "lex": float(np.mean([U[q]["lex_only"] for q in qids])), "best_w": float(W[b].split("w")[1]),
            "fixed": float(M[:, b].mean()), "oracle": float(M.max(1).mean()),
            "rel": float((M.max(1).mean() / M[:, b].mean() - 1) * 100), "flat": float(flat.mean() * 100),
            "gain_share": float((~flat & ~opt).mean() * 100)}


def summarise(runs):
    return {k: {"mean": float(np.mean([r[k] for r in runs])), "p2.5": float(np.percentile([r[k] for r in runs], 2.5)),
                "p97.5": float(np.percentile([r[k] for r in runs], 97.5))} for k in runs[0]}


def main():
    srv = Path(sys.argv[1])
    allres, lines = {}, []
    for key, d, bsrc, dev_only in CORPORA:
        fts_dir = srv / f"{d}-fts"
        bm_dir = (HERE / "results" / d) if bsrc == "local" else (srv / d)
        Uf, Ub = matrix(fts_dir / "utility_matrix.csv"), matrix(bm_dir / "utility_matrix.csv")
        common = sorted(set(Uf) & set(Ub))
        # controllers on FTS, exactly as controllers.py (same seed, same pool)
        s = K.run(fts_dir, "rrf", dev_only, 50)
        out_dir = HERE / "results" / f"{d}-fts"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "controllers_dev_rrf.json").write_text(json.dumps(s, indent=1))
        # A0 and BM25 a* on the same re-splits
        qids, acts, M, _, _ = K.load(fts_dir, "rrf")
        a0 = np.array([Uf[q]["A0_production"] for q in qids])
        strict = {r["qid"]: int(float(r["strict_match"])) for r in csv.DictReader(open(fts_dir / "signals.csv"))}
        st = np.array([strict[q] for q in qids])
        Mb = np.array([[Ub[q][a] for a in acts] if q in Ub else [np.nan] * len(acts) for q in qids])
        pool = np.arange(len(qids)) if dev_only else np.where(~K.locked_test(qids))[0]
        rng = np.random.default_rng(int(K.SEED))
        a0_runs, bm_runs, w1 = [], [], acts.index("rrf_w1.0")
        for _ in range(50):
            p = rng.permutation(pool)
            n_fit, n_cal = len(p) // 2, len(p) // 4
            fc, ev = p[:n_fit + n_cal], p[n_fit + n_cal:]
            astar = int(M[fc].mean(0).argmax())
            d0 = a0[ev] - M[ev, astar]
            a0_runs.append({"ndcg10": a0[ev].mean(), "gain": d0.mean(), "win": (d0 > 1e-12).mean(),
                            "loss": (d0 < -1e-12).mean(), "mean_loss": np.maximum(0, -d0).mean(),
                            "gain_vs_w1": (a0[ev] - M[ev, w1]).mean()})
            ok = ~np.isnan(Mb[ev, 0])
            ab = int(np.nanmean(Mb[fc], 0).argmax())
            bm_runs.append({"ndcg10_bm25_fixed": Mb[ev][ok, ab].mean(), "ndcg10_fts_fixed": M[ev][ok, astar].mean(),
                            "diff": (Mb[ev][ok, ab] - M[ev][ok, astar]).mean(), "covered": ok.mean()})
        r = {"corpus": d, "dev_only": dev_only, "dev_pool": len(pool), "split_sizes": s["split_sizes"],
             "strict_match_share_all": float(st.mean()),
             "headroom_fts": headroom(Uf, common), "headroom_bm25": headroom(Ub, common),
             "a_star_choices_fts": s["a_star_choices"], "lambda_hat_inf_share": s["lambda_hat_inf_share"],
             "controllers_fts": s["controllers"], "A0": summarise(a0_runs), "bm25_vs_fts_fixed": summarise(bm_runs)}
        allres[key] = r
        c = s["controllers"]
        lines.append(f"=== {d} | pool {len(pool)} | strict match {100 * st.mean():.0f}% | a*_FTS {s['a_star_choices']}")
        for lab, h in (("FTS ", r["headroom_fts"]), ("BM25", r["headroom_bm25"])):
            lines.append(f"  {lab} in-sample: dense {h['dense']:.3f} lex {h['lex']:.3f} w*={h['best_w']} fixed {h['fixed']:.3f} "
                         f"oracle {h['oracle']:.3f} (+{h['rel']:.1f}%) flat {h['flat']:.0f}% gain {h['gain_share']:.0f}%")
        for name in ("FIXED", "A3", "C(eps=0.02)", "C(eps=0.05)", "ORACLE"):
            g = c[name]["gain"]
            lines.append(f"  {name:<12} nDCG {c[name]['ndcg10']['mean']:.4f} gain {g['mean']:+.4f} [{g['p2.5']:+.3f},{g['p97.5']:+.3f}] "
                         f"loss {100 * c[name]['loss']['mean']:.1f}% dev {100 * c[name]['dev']['mean']:.1f}% meanLoss {c[name]['mean_loss']['mean']:.4f}")
        A = r["A0"]
        lines.append(f"  A0           nDCG {A['ndcg10']['mean']:.4f} gain {A['gain']['mean']:+.4f} [{A['gain']['p2.5']:+.3f},{A['gain']['p97.5']:+.3f}] "
                     f"loss {100 * A['loss']['mean']:.1f}% meanLoss {A['mean_loss']['mean']:.4f}  vs rrf_w1.0 {A['gain_vs_w1']['mean']:+.4f}")
        B = r["bm25_vs_fts_fixed"]
        lines.append(f"  fixed BM25 - fixed FTS (same eval folds): {B['diff']['mean']:+.4f} [{B['diff']['p2.5']:+.3f},{B['diff']['p97.5']:+.3f}]")
    (HERE / "results" / "fts_dev.json").write_text(json.dumps(allres, indent=1))
    (HERE / "results" / "fts_dev.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

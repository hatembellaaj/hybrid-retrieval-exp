#!/usr/bin/env python3
"""Locked-test evaluation, exactly as frozen in TEST_PLAN.md. Executed ONCE with --run.

    python locked_test.py --dry-run            # code check on development data only (default; no test row is used)
    python locked_test.py --run --confirm "open the locked test"

--dry-run never uses a locked-test query of a public corpus nor any clinical relevance grade:
  * public corpora: the locked test rows are dropped first; a pseudo-test (last 25 % of the development pool by
    sha256(seed + "dry" + qid)) plays the test, the rest is split fit/calibration as in the plan;
  * clinical branch: the 60 haemophilia development questions play the clinical set (utility recomputed from their
    rankings and silver qrels, and checked against results/hemophilia-dev/utility_matrix.csv); numbers are meaningless.
--run refuses to start if results/locked_test/run/ already exists.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import controllers as K
from llm_controllers import load_dat, load_llm, snap
from utility_matrix_beir import evaluate

HERE = Path(__file__).resolve().parent
SRV = HERE.parent / "server_results" / "20261117_2021" / "results"
LLM = HERE.parent / "llm_mac"
CLIN = HERE / "data" / "hemophilia-clin"
STAT_SEED = 20261121
NBOOT = NPERM = 10_000
EPS = K.EPSILONS
# corpus -> (BM25 results dir, FTS results dir, LLM features dir or None)
PUBLIC = {"scifact": (HERE / "results" / "scifact", SRV / "scifact-fts", "scifact"),
          "nfcorpus": (HERE / "results" / "nfcorpus", SRV / "nfcorpus-fts", "nfcorpus"),
          "trec-covid": (SRV / "trec-covid", SRV / "trec-covid-fts", "trec-covid"),
          "fiqa": (SRV / "fiqa", SRV / "fiqa-fts", None)}
CONFIGS = [("bm25", "rrf"), ("bm25", "cvx"), ("fts", "rrf")]


def h(tag: str, q: str) -> str:
    return hashlib.sha256((K.SEED + tag + q).encode()).hexdigest()


# ------------------------------------------------------------------ data -----
def read_matrix(path: Path) -> dict:
    U = defaultdict(dict)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            U[r["qid"]][r["action"]] = float(r["ndcg10"])
    return U


def read_signals(path: Path) -> dict:
    with open(path) as fh:
        return {r["qid"]: r for r in csv.DictReader(fh)}


class Set:
    """Utility matrix of one action family + signals (+ LLM outputs) for a list of queries."""

    def __init__(self, U: dict, sig: dict, qids: list[str], family: str, feats: list[str] | None = None,
                 llm: str | None = None):
        self.qids = qids
        self.acts = sorted((a for a in U[qids[0]] if a.startswith(family + "_")), key=lambda a: float(a.split("_")[1][1:]))
        self.vals = np.array([float(a.split("_")[1][1:]) for a in self.acts])
        self.M = np.array([[U[q][a] for a in self.acts] for q in qids])
        self.a0 = np.array([U[q]["A0_production"] for q in qids]) if "A0_production" in U[qids[0]] else None
        self.feats = feats or [k for k in sig[qids[0]] if k not in K.EXCLUDE]
        self.X = np.array([[float(sig[q][k]) for k in self.feats] for q in qids])
        self.family = family
        self.has_llm = llm is not None
        if llm:
            F, b1, b2, missing = load_llm(LLM / f"{llm}-llm" / "llm_features.csv", qids)
            assert missing == 0, f"{missing} queries without LLM features"
            self.XL = np.hstack([self.X, F])
            self.t_b1 = b1 if family == "rrf" else 1 - b1
            self.t_b2 = b2[:, 0] if family == "rrf" else b2[:, 1]
            da = load_dat(LLM / f"{llm}-dat" / "dat_features.csv", qids)
            self.t_dat = 1 - da if family == "rrf" else da

    def sub(self, idx):
        s = object.__new__(Set)
        s.__dict__ = {k: (v[idx] if isinstance(v, np.ndarray) and k != "vals" else v) for k, v in self.__dict__.items()}
        s.qids = [self.qids[i] for i in idx]
        return s


def clinical_utility(rankings: Path, rel: dict, actions_prefix=("rrf_", "cvx_", "dense_only", "lex_only", "A0_")):
    """qid -> action -> nDCG@10 from saved top lists; also the share of unjudged passages in each top-10."""
    U, unj = defaultdict(dict), defaultdict(list)
    with gzip.open(rankings, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["qid"] in rel and r["action"].startswith(actions_prefix):
                U[r["qid"]][r["action"]] = evaluate(r["top"], rel[r["qid"]])["ndcg10"]
                unj[r["action"]].append(np.mean([d not in rel["__judged__"].get(r["qid"], set()) for d in r["top"][:10]])
                                        if "__judged__" in rel else 0.0)
    return U, {a: float(np.mean(v)) for a, v in unj.items()}


# ----------------------------------------------------------- controllers -----
def crossfit_gains(tr: Set, astar: int, folds: int = 5):
    order = sorted(range(len(tr.qids)), key=lambda i: h("cf", tr.qids[i]))
    a_cal, g_cal = np.zeros(len(order), int), np.zeros(len(order))
    for k in range(folds):
        hold = np.array(order[k::folds])
        keep = np.setdiff1d(np.arange(len(order)), hold)
        a, g, _ = K.fit_a3(tr.X[keep], tr.M[keep], astar)(tr.X[hold])
        a_cal[hold], g_cal[hold] = a, g
    return a_cal, g_cal


def crossfit_gains_L(tr: Set, astar: int, folds: int = 5):
    order = sorted(range(len(tr.qids)), key=lambda i: h("cf", tr.qids[i]))
    a_cal, g_cal = np.zeros(len(order), int), np.zeros(len(order))
    for k in range(folds):
        hold = np.array(order[k::folds])
        keep = np.setdiff1d(np.arange(len(order)), hold)
        a, g, _ = K.fit_a3(tr.XL[keep], tr.M[keep], astar)(tr.XL[hold])
        a_cal[hold], g_cal[hold] = a, g
    return a_cal, g_cal


def choose(fit: Set, te: Set, astar: int, cal: Set | None, crossfit: bool) -> tuple[dict, dict]:
    """Chosen action per test query for every controller, and the lambda_hat values."""
    n = len(te.qids)
    ch, lam = {"FIXED": np.full(n, astar)}, {}
    ch["ORACLE"] = np.where(te.M[:, astar] >= te.M.max(1) - 1e-12, astar, te.M.argmax(1))
    ch["A1"] = K.fit_a1(fit.X, fit.M, astar)(te.X)
    ch["A2"] = K.fit_a2(fit.X, fit.M, fit.feats, fit.acts, fit.family)(te.X)
    variants = [("", "X")] + ([("+L", "XL")] if fit.has_llm and te.has_llm else [])
    for tag, attr in variants:
        a3 = K.fit_a3(getattr(fit, attr), fit.M, astar)
        a_te, g_te, _ = a3(getattr(te, attr))
        ch["A3" + tag] = np.where(g_te > 0, a_te, astar)
        if crossfit:
            a_c, g_c = (crossfit_gains if tag == "" else crossfit_gains_L)(fit, astar)
            Mc = fit.M
        else:
            a_c, g_c, _ = a3(getattr(cal, attr))
            Mc = cal.M
        for eps in EPS:
            l = K.crc_lambda(g_c, a_c, Mc, astar, eps)
            lam[f"C{tag}(eps={eps})"] = l
            ch[f"C{tag}(eps={eps})"] = np.where(g_te > l, a_te, astar)
    if te.has_llm:
        ch["B1"] = snap(te.t_b1, te.vals, astar)
        ch["B2"] = snap(te.t_b2, te.vals, astar)
        ch["DAT"] = snap(te.t_dat, te.vals, astar)
    return ch, lam


# ---------------------------------------------------------------- stats ------
def holm(p: dict) -> dict:
    keys = sorted(p, key=p.get)
    m, adj, run = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * p[k]))
        adj[k] = run
    return adj


def stats(te: Set, ch: dict, astar: int, extra: dict | None = None) -> dict:
    rng = np.random.default_rng(STAT_SEED)
    n = len(te.qids)
    B = rng.integers(0, n, (NBOOT, n))
    S = rng.choice([-1.0, 1.0], (NPERM, n))
    base = te.M[:, astar]
    out, pvals = {}, {}
    cols = {k: te.M[np.arange(n), v] for k, v in ch.items()}
    if extra:
        cols.update(extra)
    for name, u in cols.items():
        d = u - base
        loss = np.maximum(0, -d)
        r = {"ndcg10": float(u.mean()), "gain": float(d.mean()),
             "gain_ci95": [float(np.percentile(d[B].mean(1), 2.5)), float(np.percentile(d[B].mean(1), 97.5))],
             "win": float((d > 1e-12).mean()), "loss": float((d < -1e-12).mean()),
             "mean_loss": float(loss.mean()), "mean_loss_ub95": float(np.percentile(loss[B].mean(1), 95))}
        if name in ch:
            r["dev"] = float((ch[name] != astar).mean())
        if name not in ("FIXED", "ORACLE", "FIXED_INSAMPLE") and np.any(d != 0):
            r["p_randomisation"] = float(((np.abs((S * d).mean(1)) >= abs(d.mean()) - 1e-15).sum() + 1) / (NPERM + 1))
            pvals[name] = r["p_randomisation"]
        elif name not in ("FIXED", "ORACLE", "FIXED_INSAMPLE"):
            r["p_randomisation"] = 1.0
            pvals[name] = 1.0
        out[name] = r
    for k, v in holm(pvals).items():
        out[k]["p_holm"] = v
    return out


# --------------------------------------------------------------- public ------
def public(run: bool, log: list) -> dict:
    res = {}
    for corpus, (bm_dir, fts_dir, llm) in PUBLIC.items():
        for lex, fam in CONFIGS:
            d = bm_dir if lex == "bm25" else fts_dir
            U, sig = read_matrix(d / "utility_matrix.csv"), read_signals(d / "signals.csv")
            qids = sorted(q for q in U if q in sig)
            locked = K.locked_test(qids)
            dev = [q for q, t in zip(qids, locked) if not t]
            if run:
                test = [q for q, t in zip(qids, locked) if t]
            else:                                    # pseudo-test inside the development pool
                o = sorted(dev, key=lambda q: h("dry", q))
                test, dev = o[-len(o) // 4:], o[: -len(o) // 4]
            o = sorted(dev, key=lambda q: h("fitcal", q))
            fit_q, cal_q = o[: 2 * len(o) // 3], o[2 * len(o) // 3:]
            use_llm = llm if lex == "bm25" else None
            fit, cal, te = (Set(U, sig, qs, fam, llm=use_llm) for qs in (fit_q, cal_q, test))
            astar = int(np.concatenate([fit.M, cal.M]).mean(0).argmax())
            ch, lam = choose(fit, te, astar, cal, crossfit=False)
            extra = {"A0": te.a0} if lex == "fts" else None
            key = f"{corpus}|{lex}|{fam}"
            res[key] = {"n_fit": len(fit_q), "n_cal": len(cal_q), "n_test": len(test), "a_star": fit.acts[astar],
                        "lambda_hat": {k: (v if np.isfinite(v) else "inf") for k, v in lam.items()},
                        "controllers": stats(te, ch, astar, extra)}
            log.append(f"{key}: fit {len(fit_q)} cal {len(cal_q)} test {len(test)} a* {fit.acts[astar]}")
    return res


# ------------------------------------------------------------- clinical -----
def load_llm_grades() -> dict:
    """(qid, chunk_id) -> LLM grade of the 2,209 pooled clinical pairs (same reader as sample_validation_pairs.py)."""
    import sample_validation_pairs as svp
    svp.D = CLIN
    return svp.llm_grades()


def clinical_relevance(run: bool) -> dict:
    """variant -> {qid: {chunk: grade}, '__judged__': {qid: set(chunks)}}."""
    if not run:   # dry run: haemophilia development questions with silver qrels (grade 1) play the clinical set
        rel = defaultdict(dict)
        with open(HERE / "data" / "hemophilia-dev" / "qrels" / "test.tsv") as fh:
            next(fh)
            for line in fh:
                q, c, g = line.rstrip("\n").split("\t")
                rel[q][c] = int(g)
        grades = {(q, c): g for q, cs in rel.items() for c, g in cs.items()}
    else:
        grades = load_llm_grades()
    agree = json.loads((CLIN / "judging_validation" / "agreement.json").read_text())
    C = np.array(agree["per_expert"]["MC"]["confusion_llm_rows_expert_cols"], float)
    mapped = {g: float((C[g] * np.arange(4)).sum() / C[g].sum()) for g in range(4)}
    judged = defaultdict(set)
    for (q, c) in grades:
        judged[q].add(c)
    out = {}
    for v, f in (("a_raw", lambda g: g), ("b_mapped", lambda g: mapped[g]), ("c_grade3", lambda g: 1 if g == 3 else 0)):
        rel = defaultdict(dict)
        for (q, c), g in grades.items():
            if f(g) > 0:
                rel[q][c] = f(g)
        out[v] = dict(rel)
        out[v]["__judged__"] = judged
    out["_mapping_b"] = mapped
    return out


def clinical(run: bool, log: list) -> dict:
    rels = clinical_relevance(run)
    res = {"mapping_b": rels.pop("_mapping_b")}
    dev_dirs = {"bm25": HERE / "results" / "hemophilia-dev", "fts": SRV / "hemophilia-dev-fts"}
    if run:
        clin = {"bm25": (SRV / "hemophilia-clin" / "rankings.jsonl.gz", SRV / "hemophilia-clin" / "signals.csv"),
                "fts": (SRV / "hemophilia-clin-fts" / "rankings.jsonl.gz", SRV / "hemophilia-clin-fts" / "signals.csv")}
        clin_llm = "hemophilia-clin"
    else:
        clin = {"bm25": (LLM / "rankings" / "hemophilia-dev" / "rankings.jsonl.gz", dev_dirs["bm25"] / "signals.csv"),
                "fts": (SRV / "hemophilia-dev-fts" / "rankings.jsonl.gz", dev_dirs["fts"] / "signals.csv")}
        clin_llm = "hemophilia-dev"
    for lex, fam in CONFIGS:
        Ud, sd = read_matrix(dev_dirs[lex] / "utility_matrix.csv"), read_signals(dev_dirs[lex] / "signals.csv")
        dq = sorted(q for q in Ud if q in sd)
        use_llm = "hemophilia-dev" if lex == "bm25" else None
        tr = Set(Ud, sd, dq, fam, llm=use_llm)
        astar_dev = int(tr.M.mean(0).argmax())
        for v, rel in rels.items():
            Uc, unj = clinical_utility(clin[lex][0], rel)
            if not run and v == "a_raw" and lex == "bm25" and fam == "rrf":   # correctness check of the utility code
                diff = max(abs(Uc[q][a] - Ud[q][a]) for q in Uc for a in Ud[q] if a in Uc[q])
                log.append(f"dry-run check: utility recomputed from rankings vs utility_matrix.csv, max |diff| = {diff:.2e}")
                assert diff < 1e-9
            sc = read_signals(clin[lex][1])
            cq = sorted(q for q in Uc if q in sc and q != "__judged__")
            n_total = len(sorted(q for q in sc))
            key = f"hemophilia-clin|{lex}|{fam}|{v}"
            if len(cq) < 10:
                res[key] = {"n_questions_with_relevant": len(cq), "n_questions": n_total, "skipped": "fewer than 10 questions"}
                log.append(f"{key}: skipped, {len(cq)} questions with a relevant passage")
                continue
            te = Set(Uc, sc, cq, fam, feats=tr.feats, llm=clin_llm if lex == "bm25" else None)
            key = f"hemophilia-clin|{lex}|{fam}|{v}"
            # primary: everything chosen on the development questions
            ch, lam = choose(tr, te, astar_dev, None, crossfit=True)
            ins = int(te.M.mean(0).argmax())
            extra = {"FIXED_INSAMPLE": te.M[:, ins]}
            if lex == "fts":
                extra["A0"] = te.a0
            prim = stats(te, ch, astar_dev, extra)
            # secondary: a* and lambda_hat re-chosen on 50 clinical calibration questions
            o = sorted(range(len(cq)), key=lambda i: h("clincal", cq[i]))
            ci, ei = np.array(sorted(o[: len(o) // 2])), np.array(sorted(o[len(o) // 2:]))
            tcal, tev = te.sub(ci), te.sub(ei)
            astar_c = int(tcal.M.mean(0).argmax())
            ch2, lam2 = choose(tr, tev, astar_c, tcal, crossfit=False)
            sec = stats(tev, ch2, astar_c, {"A0": tev.a0} if lex == "fts" else None)
            res[key] = {"n_questions_with_relevant": len(cq), "n_questions": n_total, "a_star_dev": tr.acts[astar_dev],
                        "a_star_insample": te.acts[ins], "unjudged_share_top10": unj,
                        "lambda_hat_primary": {k: (x if np.isfinite(x) else "inf") for k, x in lam.items()},
                        "primary": prim,
                        "secondary_recalibrated": {"n_cal": len(ci), "n_eval": len(ei), "a_star": te.acts[astar_c],
                                                   "lambda_hat": {k: (x if np.isfinite(x) else "inf") for k, x in lam2.items()},
                                                   "controllers": sec}}
            log.append(f"{key}: {len(cq)}/{n_total} questions with a relevant passage, a*_dev {tr.acts[astar_dev]}, "
                       f"a*_insample {te.acts[ins]}")
    return res


def report(R: dict) -> str:
    L = []
    for key, r in R["public"].items():
        L.append(f"\n=== {key} | test {r['n_test']} | a* {r['a_star']}")
        L += row_lines(r["controllers"])
    for key, r in R["clinical"].items():
        if key == "mapping_b" or "skipped" in r:
            continue
        L.append(f"\n=== {key} | {r['n_questions_with_relevant']}/{r['n_questions']} q | a*_dev {r['a_star_dev']} "
                 f"| a*_insample {r['a_star_insample']} | primary")
        L += row_lines(r["primary"])
        s = r["secondary_recalibrated"]
        L.append(f"  -- secondary (recalibrated on {s['n_cal']}, eval {s['n_eval']}, a* {s['a_star']})")
        L += row_lines(s["controllers"])
    return "\n".join(L) + "\n"


def row_lines(C: dict) -> list[str]:
    out = []
    for name, m in C.items():
        p = f"p={m['p_randomisation']:.3f} holm={m['p_holm']:.3f}" if "p_holm" in m else ""
        out.append(f"  {name:<16}{m['ndcg10']:.4f} {m['gain']:+.4f} [{m['gain_ci95'][0]:+.3f},{m['gain_ci95'][1]:+.3f}] "
                   f"loss {100 * m['loss']:5.1f}% meanLoss {m['mean_loss']:.4f} (ub {m['mean_loss_ub95']:.4f}) {p}")
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--run", action="store_true")
    ap.add_argument("--confirm", default="")
    a = ap.parse_args()
    run = a.run
    out = HERE / "results" / "locked_test" / ("run" if run else "dry_run")
    if run:
        if a.confirm != "open the locked test":
            sys.exit('refused: pass --confirm "open the locked test"')
        if out.exists():
            sys.exit(f"refused: {out} exists; the locked test is run once")
    plan = (HERE / "TEST_PLAN.md").read_bytes()
    if run:   # every attempt is recorded, including one that would crash before writing results
        with open(HERE / "results" / "locked_test" / "run_attempts.log", "a") as fh:
            fh.write(f"{datetime.now(timezone.utc).isoformat()} start, plan sha256 {hashlib.sha256(plan).hexdigest()}\n")
    log = [f"mode: {'RUN' if run else 'dry-run'}  started {datetime.now(timezone.utc).isoformat()}",
           f"TEST_PLAN.md sha256 {hashlib.sha256(plan).hexdigest()}",
           f"locked_test.py sha256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}"]
    R = {"mode": "run" if run else "dry_run", "public": public(run, log), "clinical": clinical(run, log)}
    R["log"] = log
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(R, indent=1))
    txt = "\n".join(log) + "\n" + report(R)
    (out / "report.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()

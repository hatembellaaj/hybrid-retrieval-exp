#!/usr/bin/env python3
"""Non-LLM retrieval controllers (A1-A3) and the risk-controlled deviation rule (C) — DEVELOPMENT evaluation.

The locked test split of PROTOCOL.md (25 % of each corpus, chosen by sha256(seed + qid)) is NEVER read here. The other
75 % (the development pool) are re-split R times at random into fit (50 %) / calibration (25 %) / evaluation (25 %), and
results are averaged over repetitions. Corpora marked dev-only (LLM-drafted haemophilia questions) use all their queries.

Controllers (action family: rrf = rrf_w{0,.25,.5,.75,1}; cvx = cvx_a{0,.1,...,1}):
  FIXED   a*: best fixed action on fit ∪ calibration (amendment 1 of PROTOCOL.md)
  A1      depth-2 decision tree over the signals, predicting the per-query best action (a* when a* is optimal)
  A2      alpha(q) = sigmoid(w0 + w1 [z(den_gap12) - z(lex_gap12)] + w2 z(idf_max)), grid-searched on fit, snapped
  A3      per-action ensembles (5 gradient-boosted regressors) of the gain Delta(q,a) = U(q,a) - U(q,a*); argmax
  C(eps)  A3, but deviates only if the predicted gain exceeds lambda_hat, set by conformal risk control on calibration:
          lambda_hat = inf{ lambda : n/(n+1) * mean_i L_i(lambda) + 1/(n+1) <= eps },  L_i = max(0, -Delta(q_i, pi_lambda))
          (L in [0,1], non-increasing in lambda; if no lambda qualifies, lambda_hat = +inf: never deviate)
  ORACLE  per-query best action (upper bound)
Metric: nDCG@10. Reported on evaluation folds: mean U, mean gain vs a*, URisk (alpha=1), win/loss %, deviation %, mean
loss (to compare with eps for C).

    python controllers.py results/scifact [--family rrf] [--dev-only] [--reps 50]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier

SEED = "20261109"
EPSILONS = (0.02, 0.05)
LAMBDAS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, float("inf")]   # sweep for the risk curve
METRIC = "ndcg10"
EXCLUDE = {"qid", "n_relevant"}        # n_relevant comes from the qrels: never a controller input


def load(res: Path, family: str):
    U = defaultdict(dict)
    with open(res / "utility_matrix.csv") as fh:
        for r in csv.DictReader(fh):
            if r["action"].startswith(family + "_"):
                U[r["qid"]][r["action"]] = float(r[METRIC])
    sig = {}
    with open(res / "signals.csv") as fh:
        rows = list(csv.DictReader(fh))
    feats = [k for k in rows[0] if k not in EXCLUDE]
    for r in rows:
        sig[r["qid"]] = [float(r[k]) for k in feats]
    qids = sorted(q for q in U if q in sig)
    acts = sorted(U[qids[0]], key=lambda a: float(a.split("_")[1][1:]))
    M = np.array([[U[q][a] for a in acts] for q in qids])
    X = np.array([sig[q] for q in qids])
    return qids, acts, M, X, feats


def locked_test(qids: list[str]) -> np.ndarray:
    order = sorted(range(len(qids)), key=lambda i: hashlib.sha256((SEED + qids[i]).encode()).hexdigest())
    test = np.zeros(len(qids), bool)
    test[order[: len(qids) // 4]] = True
    return test


# ------------------------------------------------------------- controllers ---
def fit_a1(X, M, astar):
    best = M.max(1, keepdims=True)
    y = np.where(M[:, astar] >= best[:, 0] - 1e-12, astar, M.argmax(1))
    clf = DecisionTreeClassifier(max_depth=2, min_samples_leaf=max(5, len(y) // 10), random_state=0).fit(X, y)
    return lambda Xe: clf.predict(Xe)


def fit_a2(X, M, feats, acts, family):
    ix = [feats.index(f) for f in ("den_gap12", "lex_gap12", "idf_max")]
    mu, sd = X[:, ix].mean(0), X[:, ix].std(0) + 1e-9
    grid = np.linspace(-3, 3, 13)
    vals = np.array([float(a.split("_")[1][1:]) for a in acts])        # w_lex (rrf) or alpha = dense weight (cvx)

    def act_index(Xs, w):
        Z = (Xs[:, ix] - mu) / sd
        alpha = 1 / (1 + np.exp(-(w[0] + w[1] * (Z[:, 0] - Z[:, 1]) + w[2] * Z[:, 2])))
        target = (1 - alpha) if family == "rrf" else alpha
        return np.abs(target[:, None] - vals[None, :]).argmin(1)

    best_w, best_u = None, -1.0
    for w0 in grid:
        for w1 in grid:
            for w2 in grid:
                u = M[np.arange(len(M)), act_index(X, (w0, w1, w2))].mean()
                if u > best_u + 1e-12:
                    best_w, best_u = (w0, w1, w2), u
    return lambda Xe: act_index(Xe, best_w)


def fit_a3(X, M, astar, n_models=5):
    D = M - M[:, [astar]]
    models = {}
    for j in range(M.shape[1]):
        if j == astar:
            continue
        models[j] = [GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8,
                                               random_state=b).fit(X, D[:, j]) for b in range(n_models)]

    def predict(Xe):
        G = np.zeros((len(Xe), M.shape[1]))
        U = np.zeros_like(G)
        for j, ms in models.items():
            P = np.stack([m.predict(Xe) for m in ms])
            G[:, j], U[:, j] = P.mean(0), P.std(0)
        a = G.argmax(1)
        return a, G[np.arange(len(Xe)), a], U[np.arange(len(Xe)), a]
    return predict


def crc_lambda(gain_hat, dev_action, M, astar, eps):
    """Smallest lambda whose CRC-adjusted calibration risk is <= eps (loss bound B = 1)."""
    n = len(M)
    loss_dev = np.maximum(0.0, -(M[np.arange(n), dev_action] - M[:, astar]))
    for lam in np.concatenate(([0.0], np.sort(np.unique(gain_hat[gain_hat > 0])))):
        L = np.where(gain_hat > lam, loss_dev, 0.0).mean()
        if n / (n + 1) * L + 1 / (n + 1) <= eps:
            return float(lam)
    return float("inf")


# -------------------------------------------------------------- evaluation ---
def scores(M, chosen, astar):
    n = len(M)
    u = M[np.arange(n), chosen]
    d = u - M[:, astar]
    return {"ndcg10": u.mean(), "gain": d.mean(), "urisk": d[d > 0].sum() / n + 2 * d[d < 0].sum() / n,
            "win": (d > 1e-12).mean(), "loss": (d < -1e-12).mean(), "dev": (chosen != astar).mean(),
            "mean_loss": np.maximum(0, -d).mean()}


def run(res: Path, family: str, dev_only: bool, reps: int) -> dict:
    qids, acts, M, X, feats = load(res, family)
    pool = np.arange(len(qids)) if dev_only else np.where(~locked_test(qids))[0]
    rng = np.random.default_rng(int(SEED))
    out = defaultdict(list)
    lambdas = defaultdict(list)
    sweep = defaultdict(list)
    astars = []
    for _ in range(reps):
        p = rng.permutation(pool)
        n_fit, n_cal = len(p) // 2, len(p) // 4
        fit, cal, ev = p[:n_fit], p[n_fit:n_fit + n_cal], p[n_fit + n_cal:]
        astar = int(M[np.concatenate([fit, cal])].mean(0).argmax())
        astars.append(acts[astar])
        out["FIXED"].append(scores(M[ev], np.full(len(ev), astar), astar))
        oracle = np.where(M[ev, astar] >= M[ev].max(1) - 1e-12, astar, M[ev].argmax(1))   # keep a* on ties
        out["ORACLE"].append(scores(M[ev], oracle, astar))
        out["A1"].append(scores(M[ev], fit_a1(X[fit], M[fit], astar)(X[ev]), astar))
        out["A2"].append(scores(M[ev], fit_a2(X[fit], M[fit], feats, acts, family)(X[ev]), astar))
        a3 = fit_a3(X[fit], M[fit], astar)
        a_ev, g_ev, _ = a3(X[ev])
        out["A3"].append(scores(M[ev], np.where(g_ev > 0, a_ev, astar), astar))
        for lam in LAMBDAS:
            sweep[lam].append(scores(M[ev], np.where(g_ev > lam, a_ev, astar), astar))
        a_cal, g_cal, _ = a3(X[cal])
        for eps in EPSILONS:
            lam = crc_lambda(g_cal, a_cal, M[cal], astar, eps)
            lambdas[eps].append(lam)
            out[f"C(eps={eps})"].append(scores(M[ev], np.where(g_ev > lam, a_ev, astar), astar))
    summary = {"corpus": res.name, "family": family, "queries_total": len(qids), "dev_pool": len(pool),
               "locked_test_untouched": not dev_only, "reps": reps, "split_sizes": [len(pool) // 2, len(pool) // 4, len(pool) - len(pool) // 2 - len(pool) // 4],
               "a_star_choices": {a: astars.count(a) for a in sorted(set(astars))},
               "lambda_hat_inf_share": {str(e): float(np.mean(np.isinf(v))) for e, v in lambdas.items()},
               "lambda_hat_mean_finite": {str(e): (float(np.mean([x for x in v if np.isfinite(x)])) if any(np.isfinite(v)) else None) for e, v in lambdas.items()},
               "lambda_sweep": [{"lambda": (lam if np.isfinite(lam) else "inf"), **{k: float(np.mean([r[k] for r in runs])) for k in runs[0]}}
                                for lam, runs in sweep.items()],
               "controllers": {}}
    for name, runs in out.items():
        keys = runs[0].keys()
        summary["controllers"][name] = {k: {"mean": float(np.mean([r[k] for r in runs])),
                                            "p2.5": float(np.percentile([r[k] for r in runs], 2.5)),
                                            "p97.5": float(np.percentile([r[k] for r in runs], 97.5))} for k in keys}
    return summary


def show(s: dict) -> None:
    print(f"\n=== {s['corpus']} | {s['family']} | pool {s['dev_pool']}/{s['queries_total']} | fit/cal/eval {s['split_sizes']} "
          f"| {s['reps']} reps | test untouched: {s['locked_test_untouched']}")
    print(f"a* chosen: {s['a_star_choices']} | lambda_hat=+inf share: {s['lambda_hat_inf_share']}")
    print(f"{'controller':<14}{'nDCG@10':>9}{'gain':>9}{'  [95% reps]':>18}{'URisk':>9}{'win%':>7}{'loss%':>7}{'dev%':>7}{'meanLoss':>10}")
    for name, m in s["controllers"].items():
        g = m["gain"]
        print(f"{name:<14}{m['ndcg10']['mean']:>9.4f}{g['mean']:>+9.4f}   [{g['p2.5']:+.3f},{g['p97.5']:+.3f}]"
              f"{m['urisk']['mean']:>+9.4f}{100*m['win']['mean']:>7.1f}{100*m['loss']['mean']:>7.1f}"
              f"{100*m['dev']['mean']:>7.1f}{m['mean_loss']['mean']:>10.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--family", default="rrf", choices=["rrf", "cvx"])
    ap.add_argument("--dev-only", action="store_true", help="corpus whose queries are all development queries")
    ap.add_argument("--reps", type=int, default=50)
    a = ap.parse_args()
    s = run(Path(a.results), a.family, a.dev_only, a.reps)
    Path(a.results, f"controllers_dev_{a.family}.json").write_text(json.dumps(s, indent=1))
    show(s)

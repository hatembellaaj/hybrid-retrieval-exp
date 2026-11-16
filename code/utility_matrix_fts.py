#!/usr/bin/env python3
"""Per-query rankings (and, when qrels exist, utility matrix) with the DATABASE FULL-TEXT lexical retriever
(fts_pg.PgFts: PostgreSQL, ts_rank_cd without IDF, conjunctive stage then disjunctive fallback, field weights).

For every query of a BEIR-format dataset:
  dense  : exact cosine over snowflake-arctic-embed-s vectors (cached by utility_matrix_beir.embed), top-50
  lexical: PgFts.search(query, k=50), English stemming
  actions: dense_only, lex_only, rrf_w{0,.25,.5,.75,1} (k=60), cvx_a{0,...,1} (min-max per list, absent -> 0),
           A0_production (w_lex = 1 if the conjunctive stage matched, else 0.5)
Index fields: file name = the corpus "doc_id" field when present (weight A), section headings = "title" (B), body = "text" (C).
Outputs in <out>/:
  rankings.jsonl.gz   {"qid", "action", "top": [doc ids, top-20]} for every query and action (evaluate later with any qrels)
  signals.csv         per-query signals (same names as utility_matrix_beir.py; lex_* from FTS scores) + strict_match, fts_hits
  utility_matrix.csv  only if <data>/qrels/<split>.tsv exists
  manifest.json
The collection is loaded into the experiment database and deleted at the end (also on error).

    python utility_matrix_fts.py --data data/scifact --out results/scifact-fts --dsn postgresql://postgres:postgres@127.0.0.1:18508/fts
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np

from fts_pg import PgFts
from fusion import fuse_rrf, fuse_weighted_linear, normalize_scores_minmax
from utility_matrix_beir import (_ACRONYM, _NUMBER, _TOKEN, _UNIT, ALPHAS, MODEL, POOL, RRF_K, RRF_WEIGHTS, BM25Index,
                                 embed, evaluate, load_beir, ranked, rbo, top_stats)
from utility_matrix_fts_compat import load_queries_only

OR_FALLBACK_WEIGHT = 0.5      # production rule A0: lexical RRF weight when only the disjunctive stage matched
TOP_SAVED = 20


def fts_rows(data: Path) -> list[dict]:
    rows = []
    with open(data / "corpus.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            rows.append({"doc_id": str(d["_id"]), "file_name": d.get("doc_id") or "", "headings": d.get("title") or "",
                         "body": d.get("text") or ""})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--config", default="english")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("THREADS", "4")))
    a = ap.parse_args()
    t0 = time.time()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    has_qrels = (data / "qrels" / f"{a.split}.tsv").exists()
    if has_qrels:
        corpus, queries, qrels = load_beir(data, a.split)
        qids = [q for q in qrels if q in queries and any(r > 0 for r in qrels[q].values())]
    else:
        corpus, queries = load_queries_only(data)
        qrels, qids = {}, sorted(queries)
    ids = list(corpus)
    texts = [corpus[i] for i in ids]
    D = embed(texts, False, data / ".cache", a.threads)
    Q = embed([queries[q] for q in qids], True, data / ".cache", a.threads)
    bm25 = BM25Index(ids, texts)             # IDF / OOV signals only; the lexical ranking is FTS
    idf = bm25.model.idf
    centroid = D.mean(axis=0)
    centroid /= np.linalg.norm(centroid)

    fts = PgFts(a.dsn, a.config)
    collection = f"exp_{uuid.uuid4().hex[:8]}"
    t_load = time.time()
    fts.load(collection, fts_rows(data))
    load_s = time.time() - t_load
    um_rows, sig_rows, search_s = [], [], []
    try:
        with gzip.open(out / "rankings.jsonl.gz", "wt", encoding="utf-8") as rk:
            for qi, qid in enumerate(qids):
                text = queries[qid]
                dense_all = D @ Q[qi]
                d_top = np.argsort(-dense_all, kind="stable")[:POOL]
                dense = {ids[i]: float(dense_all[i]) for i in d_top}
                ts = time.time()
                outcome = fts.search(collection, text, k=POOL)
                search_s.append(time.time() - ts)
                lex = outcome.scores
                acts = {"dense_only": ranked(dense), "lex_only": ranked(lex)}
                for w in RRF_WEIGHTS:
                    acts[f"rrf_w{w}"] = ranked(fuse_rrf(dense, lex, k=RRF_K, sparse_weight=w))
                d_n, l_n = normalize_scores_minmax(dense), normalize_scores_minmax(lex)
                for al in ALPHAS:
                    acts[f"cvx_a{al}"] = ranked(fuse_weighted_linear(d_n, l_n, dense_weight=al))
                w0 = 1.0 if outcome.strict_match else OR_FALLBACK_WEIGHT
                acts["A0_production"] = ranked(fuse_rrf(dense, lex, k=RRF_K, sparse_weight=w0))
                for act, ranking in acts.items():
                    rk.write(json.dumps({"qid": qid, "action": act, "top": ranking[:TOP_SAVED]}) + "\n")
                    if has_qrels:
                        um_rows.append({"qid": qid, "action": act, **evaluate(ranking, qrels[qid])})
                toks = bm25.tok(text)
                q_idf = [idf.get(t, 0.0) for t in toks] or [0.0]
                rd, rl = acts["dense_only"], acts["lex_only"]
                lex_vals = np.array(sorted(lex.values(), reverse=True)) if lex else np.zeros(0)
                if len(lex_vals) < 2:
                    lex_vals = np.concatenate([lex_vals, np.zeros(2 - len(lex_vals))])
                ds, ls = top_stats(dense_all[d_top]), top_stats(lex_vals)
                sig_rows.append({
                    "qid": qid, "query_len": len(_TOKEN.findall(text)), "has_number": int(bool(_NUMBER.search(text))),
                    "has_unit": int(bool(_UNIT.search(text))), "n_acronyms": len(_ACRONYM.findall(text)),
                    "idf_max": max(q_idf), "idf_mean": float(np.mean(q_idf)), "idf_min": min(q_idf),
                    "oov_rate": sum(t not in bm25.vocab for t in toks) / max(1, len(toks)),
                    "centroid_sim": float(Q[qi] @ centroid),
                    **{f"den_{k}": v for k, v in ds.items()}, **{f"lex_{k}": v for k, v in ls.items()},
                    "lex_hits": len(lex),
                    "jaccard10": len(set(rd[:10]) & set(rl[:10])) / max(1, len(set(rd[:10]) | set(rl[:10]))),
                    "rbo10": rbo(rd, rl),
                    "den_top1_rank_in_lex": (rl.index(rd[0]) + 1) if rd and rd[0] in rl else POOL + 1,
                    "lex_top1_rank_in_den": (rd.index(rl[0]) + 1) if rl and rl[0] in rd else POOL + 1,
                    "strict_match": int(outcome.strict_match), "fts_hits": len(lex),
                    **({"n_relevant": sum(1 for r in qrels[qid].values() if r > 0)} if has_qrels else {}),
                })
                if (qi + 1) % 100 == 0:
                    print(f"  {qi + 1}/{len(qids)} queries", flush=True)
    finally:
        fts.drop(collection)
        fts.close()
    for name, rows in (("signals.csv", sig_rows), ("utility_matrix.csv", um_rows)):
        if rows:
            with open(out / name, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    manifest = {"data": str(data), "split": a.split, "model": MODEL,
                "lexical": f"PostgreSQL FTS (fts_pg.PgFts, ts_rank_cd norm 32, AND then OR fallback, config {a.config})",
                "pool": POOL, "rrf_k": RRF_K, "a0_or_weight": OR_FALLBACK_WEIGHT, "documents": len(ids), "queries": len(qids),
                "qrels": has_qrels, "strict_match_share": float(np.mean([r["strict_match"] for r in sig_rows])) if sig_rows else None,
                "index_load_seconds": round(load_s, 1), "fts_search_ms_p50": round(1000 * float(np.median(search_s)), 1) if search_s else None,
                "fts_search_ms_p95": round(1000 * float(np.percentile(search_s, 95)), 1) if search_s else None,
                "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()

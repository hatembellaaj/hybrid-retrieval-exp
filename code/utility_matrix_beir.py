#!/usr/bin/env python3
"""Per-query utility matrix U(q, a) and observable signals for a BEIR-format dataset (lexical retriever: BM25).

Outputs in <out>/:
  utility_matrix.csv   qid, action, ndcg10, hit10, recall10, mrr10   (one row per query and fusion action)
  signals.csv          one row per query: pre-retrieval, post-retrieval and agreement signals (PROTOCOL.md section 3)
  manifest.json        dataset, model, parameters, sizes, timing
Graded qrels give graded nDCG (gain 2^rel - 1); queries without a relevant document in the qrels are skipped and counted.

    python utility_matrix_beir.py --data data/scifact --split test --out results/scifact
Embeddings are cached in <data>/.cache/ (one .npy per model and text set).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import snowballstemmer
from rank_bm25 import BM25Okapi

from fusion import fuse_rrf, fuse_weighted_linear, normalize_scores_minmax

K_EVAL = 10
POOL = 50
RRF_K = 60
RRF_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
ALPHAS = [round(0.1 * i, 1) for i in range(11)]          # weight of the DENSE channel
MODEL = "snowflake/snowflake-arctic-embed-s"
PREFIX = {MODEL: ("Represent this sentence for searching relevant passages: ", "")}

_TOKEN = re.compile(r"\w+", re.UNICODE)
_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9\-]{1,}\b")
_NUMBER = re.compile(r"\d")
_UNIT = re.compile(r"\d\s?(%|mg|kg|iu|ui|dl|ml|bu|ub|g/l|mmol|h|days?|weeks?)\b", re.I)


# ------------------------------------------------------------------ data ---
def load_beir(data: Path, split: str):
    corpus = {}
    with open(data / "corpus.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            corpus[str(d["_id"])] = ((d.get("title") or "") + " " + (d.get("text") or "")).strip()
    queries = {}
    with open(data / "queries.jsonl", encoding="utf-8") as fh:
        for line in fh:
            q = json.loads(line)
            queries[str(q["_id"])] = q["text"]
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(data / "qrels" / f"{split}.tsv", encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            qid, cid, score = line.rstrip("\n").split("\t")[:3]
            qrels[qid][cid] = int(float(score))
    return corpus, queries, qrels


# ------------------------------------------------------------- retrievers ---
class BM25Index:
    def __init__(self, ids: list[str], texts: list[str], lang: str = "english"):
        self.stem = snowballstemmer.stemmer(lang)
        self.ids = ids
        self.model = BM25Okapi([self.tok(t) for t in texts], k1=1.2, b=0.75)
        self.vocab = set(self.model.idf)

    def tok(self, text: str) -> list[str]:
        return self.stem.stemWords(_TOKEN.findall(text.lower()))

    def scores(self, query: str) -> np.ndarray:
        return self.model.get_scores(self.tok(query))


def embed(texts: list[str], query: bool, cache: Path, threads: int) -> np.ndarray:
    key = hashlib.sha256(("Q" if query else "D").encode() + MODEL.encode() + "\x00".join(texts).encode()).hexdigest()[:16]
    path = cache / f"{key}.npy"
    if path.exists():
        return np.load(path)
    from fastembed import TextEmbedding

    qp, dp = PREFIX.get(MODEL, ("", ""))
    model = TextEmbedding(model_name=MODEL, cache_dir=os.environ.get("FASTEMBED_CACHE"), threads=threads)
    E = np.array(list(model.embed([(qp if query else dp) + t for t in texts], batch_size=32)), dtype=np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    cache.mkdir(parents=True, exist_ok=True)
    np.save(path, E)
    return E


# ---------------------------------------------------------------- metrics ---
def ranked(scores: dict[str, float]) -> list[str]:
    return [d for d, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def evaluate(ranking: list[str], rel: dict[str, int], k: int = K_EVAL) -> dict[str, float]:
    top = ranking[:k]
    gains = [(2 ** rel.get(d, 0) - 1) for d in top]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted((2 ** r - 1 for r in rel.values() if r > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    hits = [rel.get(d, 0) > 0 for d in top]
    n_rel = sum(1 for r in rel.values() if r > 0)
    return {"ndcg10": dcg / idcg if idcg else 0.0, "hit10": float(any(hits)),
            "recall10": sum(hits) / n_rel if n_rel else 0.0,
            "mrr10": next((1.0 / (i + 1) for i, h in enumerate(hits) if h), 0.0)}


def rbo(a: list[str], b: list[str], p: float = 0.9, depth: int = 10) -> float:
    s, A, B = 0.0, set(), set()
    for d in range(1, depth + 1):
        if d <= len(a):
            A.add(a[d - 1])
        if d <= len(b):
            B.add(b[d - 1])
        s += p ** (d - 1) * len(A & B) / d
    return (1 - p) * s


def top_stats(vals: np.ndarray) -> dict[str, float]:
    v = np.sort(vals)[::-1]
    v10 = v[:10]
    pos = v10 - v10.min()
    probs = pos / pos.sum() if pos.sum() > 0 else np.full(len(v10), 1 / len(v10))
    return {"top1": float(v[0]), "gap12": float(v[0] - v[1]) if len(v) > 1 else 0.0,
            "nqc": float(np.std(v10) / (abs(np.mean(v10)) + 1e-9)),
            "entropy10": float(-(probs * np.log(probs + 1e-12)).sum())}


# ------------------------------------------------------------------- main ---
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="english")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("THREADS", "4")))
    args = ap.parse_args()
    t0 = time.time()
    data, out = Path(args.data), Path(args.out)
    has_qrels = (data / "qrels" / f"{args.split}.tsv").exists()
    if has_qrels:
        corpus, queries, qrels = load_beir(data, args.split)
        qids = [q for q in qrels if q in queries and any(r > 0 for r in qrels[q].values())]
        skipped = len([q for q in qrels if q not in qids])
    else:                                    # no relevance judgements (e.g. clinical set judged later): rankings only
        from utility_matrix_fts_compat import load_queries_only
        corpus, queries = load_queries_only(data)
        qrels, qids, skipped = {}, sorted(queries), 0
    ids = list(corpus)
    texts = [corpus[i] for i in ids]
    out.mkdir(parents=True, exist_ok=True)
    rk = gzip.open(out / "rankings.jsonl.gz", "wt", encoding="utf-8")

    bm25 = BM25Index(ids, texts, args.lang)
    D = embed(texts, False, data / ".cache", args.threads)
    Q = embed([queries[q] for q in qids], True, data / ".cache", args.threads)
    centroid = D.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    idf = bm25.model.idf

    um_rows, sig_rows = [], []
    for qi, qid in enumerate(qids):
        text, rel = queries[qid], qrels.get(qid, {})
        dense_all = D @ Q[qi]
        lex_all = bm25.scores(text)
        d_top = np.argsort(-dense_all, kind="stable")[:POOL]
        l_top = [i for i in np.argsort(-lex_all, kind="stable")[:POOL] if lex_all[i] > 0]
        dense = {ids[i]: float(dense_all[i]) for i in d_top}
        lex = {ids[i]: float(lex_all[i]) for i in l_top}
        union = sorted(set(d_top) | set(l_top))
        # convex fusion: exact scores of both channels for every candidate (no imputation)
        d_u = normalize_scores_minmax({ids[i]: float(dense_all[i]) for i in union})
        l_u = normalize_scores_minmax({ids[i]: float(lex_all[i]) for i in union})
        acts = {"dense_only": ranked(dense), "lex_only": ranked(lex)}
        for w in RRF_WEIGHTS:
            acts[f"rrf_w{w}"] = ranked(fuse_rrf(dense, lex, k=RRF_K, sparse_weight=w))
        for a in ALPHAS:
            acts[f"cvx_a{a}"] = ranked(fuse_weighted_linear(d_u, l_u, dense_weight=a))
        for act, ranking in acts.items():
            rk.write(json.dumps({"qid": qid, "action": act, "top": ranking[:20]}) + "\n")
            if has_qrels:
                um_rows.append({"qid": qid, "action": act, **evaluate(ranking, rel)})

        toks = bm25.tok(text)
        q_idf = [idf.get(t, 0.0) for t in toks] or [0.0]
        rd, rl = acts["dense_only"], acts["lex_only"]
        ds, ls = top_stats(dense_all[d_top]), top_stats(lex_all[l_top] if l_top else np.zeros(2))
        sig_rows.append({
            "qid": qid, "query_len": len(_TOKEN.findall(text)), "has_number": int(bool(_NUMBER.search(text))),
            "has_unit": int(bool(_UNIT.search(text))), "n_acronyms": len(_ACRONYM.findall(text)),
            "idf_max": max(q_idf), "idf_mean": float(np.mean(q_idf)), "idf_min": min(q_idf),
            "oov_rate": sum(t not in bm25.vocab for t in toks) / max(1, len(toks)),
            "centroid_sim": float(Q[qi] @ centroid),
            **{f"den_{k}": v for k, v in ds.items()}, **{f"lex_{k}": v for k, v in ls.items()},
            "lex_hits": len(l_top),
            "jaccard10": len(set(rd[:10]) & set(rl[:10])) / max(1, len(set(rd[:10]) | set(rl[:10]))),
            "rbo10": rbo(rd, rl),
            "den_top1_rank_in_lex": (rl.index(rd[0]) + 1) if rd and rd[0] in rl else POOL + 1,
            "lex_top1_rank_in_den": (rd.index(rl[0]) + 1) if rl and rl[0] in rd else POOL + 1,
            **({"n_relevant": sum(1 for r in rel.values() if r > 0)} if has_qrels else {}),
        })

    rk.close()
    for name, rows in (("utility_matrix.csv", um_rows), ("signals.csv", sig_rows)):
        if not rows:
            continue
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    manifest = {"data": str(data), "split": args.split, "model": MODEL, "lexical": "BM25 (k1=1.2, b=0.75, Snowball " + args.lang + ")",
                "pool": POOL, "rrf_k": RRF_K, "rrf_weights": RRF_WEIGHTS, "alphas": ALPHAS, "documents": len(ids),
                "queries": len(qids), "qrels": has_qrels, "queries_skipped_no_relevant": skipped, "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()

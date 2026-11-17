#!/usr/bin/env python3
"""LLM controller features B1 (query only) and B2 (query + top-m passages of each retriever), via a local Ollama server.

B1: one call (temperature 0) -> query_type, needs_exact_match, key_terms, mode, lexical_weight.
B2: the top-3 passages of the dense list and of the BM25 list (union, deduplicated, order shuffled with a fixed seed so
    the model cannot tell which retriever proposed them) are graded 0-3 in one call; B2_SAMPLES calls at temperature 0.7
    (self-consistency). For each sample, g_den and g_lex are the DCG of the grades of each list's top-3 in rank order;
    alpha_L = g_den / (g_den + g_lex) (dense weight) and w_lex_L = g_lex / max(g_den, g_lex); both 0.5 if g_den = g_lex = 0.
Every call is cached in <out>/llm_cache.jsonl (the run resumes after an interruption) with latency and token counts.
Outputs: <out>/llm_features.csv, <out>/manifest.json.

    python llm_features.py --data data/scifact --out results/scifact-llm --model qwen2.5:7b
With --rankings <rankings.jsonl.gz> (written by utility_matrix_beir.py), the top-3 dense and BM25 lists are read from
that file (actions dense_only and lex_only) instead of being recomputed: same lists, no embeddings needed.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import time
import urllib.request
from pathlib import Path

import numpy as np

import prompts as P
from utility_matrix_beir import BM25Index, embed
from utility_matrix_fts_compat import load_queries_only


class Ollama:
    def __init__(self, host: str, model: str, cache: Path):
        self.host, self.model, self.cache_path = host.rstrip("/"), model, cache
        self.cache = {}
        if cache.exists():
            for line in cache.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.cache[r["key"]] = r
        self.fh = open(cache, "a", encoding="utf-8")

    def chat(self, system: str, user: str, schema: dict, temperature: float, seed: int) -> dict:
        key = hashlib.sha256(json.dumps([self.model, P.PROMPT_VERSION, system, user, temperature, seed]).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        body = {"model": self.model, "stream": False, "format": schema,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"temperature": temperature, "seed": seed, "num_ctx": 4096}}
        t0 = time.time()
        req = urllib.request.Request(self.host + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as resp:
            r = json.loads(resp.read())
        rec = {"key": key, "content": r["message"]["content"], "prompt_tokens": r.get("prompt_eval_count", 0),
               "completion_tokens": r.get("eval_count", 0), "seconds": round(time.time() - t0, 3)}
        self.cache[key] = rec
        self.fh.write(json.dumps(rec) + "\n")
        self.fh.flush()
        return rec


def dcg(grades: list[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def snippet(text: str, n: int) -> str:
    w = text.split()
    return " ".join(w[:n]) + (" …" if len(w) > n else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "qwen2.5:7b"))
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434"))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("THREADS", "4")))
    ap.add_argument("--limit", type=int, default=0, help="first N queries only (smoke test)")
    ap.add_argument("--rankings", help="rankings.jsonl.gz of utility_matrix_beir.py (BM25 run) to reuse")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    corpus, queries = load_queries_only(data)
    qids = sorted(queries)[: a.limit or None]
    ids = list(corpus)
    tops = {}
    if a.rankings:
        with gzip.open(a.rankings, "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["action"] in ("dense_only", "lex_only"):
                    tops.setdefault(r["qid"], {})[r["action"]] = r["top"]
        qids = [q for q in qids if q in tops]
    else:
        texts = [corpus[i] for i in ids]
        D = embed(texts, False, data / ".cache", a.threads)
        Q = embed([queries[q] for q in qids], True, data / ".cache", a.threads)
        bm25 = BM25Index(ids, texts)
    llm = Ollama(a.host, a.model, out / "llm_cache.jsonl")
    rows, t0 = [], time.time()
    for qi, qid in enumerate(qids):
        q = queries[qid]
        calls = []
        # ---- B1
        r1 = llm.chat(P.B1_SYSTEM, P.B1_USER.format(query=q), P.B1_SCHEMA, 0.0, 0)
        calls.append(r1)
        try:
            b1 = json.loads(r1["content"])
        except json.JSONDecodeError:
            b1 = {}
        # ---- B2
        if a.rankings:
            d_rank, l_rank = tops[qid]["dense_only"][: P.B2_TOP_M], tops[qid]["lex_only"][: P.B2_TOP_M]
        else:
            d_rank = [ids[i] for i in np.argsort(-(D @ Q[qi]), kind="stable")[: P.B2_TOP_M]]
            ls = bm25.scores(q)
            l_rank = [ids[i] for i in np.argsort(-ls, kind="stable")[: P.B2_TOP_M] if ls[i] > 0]
        pool = list(dict.fromkeys(d_rank + l_rank))
        random.Random(f"{P.PROMPT_VERSION}|{qid}").shuffle(pool)
        number = {doc: n + 1 for n, doc in enumerate(pool)}
        passages = "\n\n".join(f"[{number[d]}] {snippet(corpus[d], P.B2_SNIPPET_WORDS)}" for d in pool)
        samples = []
        for s in range(P.B2_SAMPLES):
            r2 = llm.chat(P.B2_SYSTEM, P.B2_USER.format(query=q, passages=passages), P.B2_SCHEMA, P.B2_TEMPERATURE, s + 1)
            calls.append(r2)
            try:
                g = {int(x["id"]): int(x["grade"]) for x in json.loads(r2["content"])["grades"]}
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                g = {}
            if not all(number[d] in g and 0 <= g[number[d]] <= 3 for d in pool):
                samples.append(None)                         # malformed output: counted, not imputed
                continue
            gd, gl = dcg([g[number[d]] for d in d_rank]), dcg([g[number[d]] for d in l_rank])
            both0 = gd == 0 and gl == 0
            samples.append({"grades": {d: g[number[d]] for d in pool}, "alpha": 0.5 if both0 else gd / (gd + gl),
                            "wlex": 0.5 if both0 else gl / max(gd, gl), "both_zero": both0})
        ok = [x for x in samples if x]
        rows.append({
            "qid": qid, "b1_query_type": b1.get("query_type"), "b1_needs_exact": b1.get("needs_exact_match"),
            "b1_mode": b1.get("mode"), "b1_wlex": b1.get("lexical_weight"), "b1_key_terms": "|".join(b1.get("key_terms") or []),
            "b2_valid_samples": len(ok), "b2_alpha_mean": float(np.mean([x["alpha"] for x in ok])) if ok else "",
            "b2_alpha_sd": float(np.std([x["alpha"] for x in ok])) if ok else "",
            "b2_wlex_mean": float(np.mean([x["wlex"] for x in ok])) if ok else "",
            "b2_wlex_sd": float(np.std([x["wlex"] for x in ok])) if ok else "",
            "b2_both_zero_share": float(np.mean([x["both_zero"] for x in ok])) if ok else "",
            "b2_grades": json.dumps([x["grades"] if x else None for x in samples]),
            "calls": len(calls), "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
            "completion_tokens": sum(c["completion_tokens"] for c in calls), "llm_seconds": round(sum(c["seconds"] for c in calls), 2),
        })
        if (qi + 1) % 10 == 0:
            print(f"  {qi + 1}/{len(qids)} queries, {time.time() - t0:.0f} s", flush=True)
    with open(out / "llm_features.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import prompts as _p
    manifest = {"data": str(data), "model": a.model, "rankings": a.rankings, "ollama_host": a.host, "prompt_version": P.PROMPT_VERSION,
                "prompts_sha256": hashlib.sha256(Path(_p.__file__).read_bytes()).hexdigest(), "queries": len(rows),
                "b2_samples": P.B2_SAMPLES, "b2_top_m": P.B2_TOP_M, "invalid_b2_samples": sum(P.B2_SAMPLES - r["b2_valid_samples"] for r in rows),
                "total_calls": sum(r["calls"] for r in rows), "total_llm_seconds": round(sum(r["llm_seconds"] for r in rows), 1),
                "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()

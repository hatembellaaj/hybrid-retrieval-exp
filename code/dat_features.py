#!/usr/bin/env python3
"""DAT baseline (Hsu & Tzeng 2025, "DAT: Dynamic Alpha Tuning for Hybrid Retrieval in RAG", arXiv 2503.23013),
re-implemented as published, with our retrievers and our local LLM.

Per query, ONE call: the LLM sees the query, the top-1 passage of the dense list and the top-1 passage of the BM25 list,
and returns two integers 0-5 (dense first, then BM25) on the published rubric (5 direct hit; 3-4 conceptually close;
1-2 loosely related; 0 off-track). The prompt follows the structure of the published prompt (role, task, criteria,
data, output "two integers separated by a space"), in our own wording. alpha (weight of the DENSE score) is
    0.5 if Sv = Sb = 0;  1.0 if Sv = 5 and Sb != 5;  0.0 if Sb = 5 and Sv != 5;  Sv / (Sv + Sb) otherwise,
rounded to one decimal (half up). Unspecified in the paper and fixed here: temperature 0, passages truncated to the
first DAT_WORDS words, output parsed as the first two integers in the reply (otherwise the call is invalid and the
controller falls back to the fixed action).

Every call is cached in <out>/dat_cache.jsonl (resumable). Outputs: <out>/dat_features.csv, <out>/manifest.json.

    python dat_features.py --data data/scifact --out ../llm_mac/scifact-dat --rankings rankings.jsonl.gz
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
import urllib.error
import urllib.request
from pathlib import Path

from utility_matrix_fts_compat import load_queries_only

DAT_VERSION = "2026-11-19-v1"
DAT_WORDS = 300
DAT_TEMPERATURE = 0.0
DAT_PROMPT = """You are an evaluator who judges how effective two retrieval methods are for a question: dense retrieval (embedding similarity) and BM25 (keyword matching). For each method you are shown only its top-ranked passage.

## Task
Score each passage separately from 0 to 5 for how well it helps answer the question.

## Scoring criteria
- 5: direct hit; the passage answers the question.
- 3-4: conceptually close; 4 if very close to the answer, 3 if somewhat close.
- 1-2: loosely related but likely misleading; 2 if the answer may be nearby, 1 if unlikely.
- 0: completely off-track.

## Data
Question: {question}

Dense retrieval top-1 passage: {dense}

BM25 top-1 passage: {bm25}

## Output format
Return two integers separated by a space: first the score of the dense passage, then the score of the BM25 passage (for example: 3 4). Do not output any other text."""


def snippet(text: str, n: int) -> str:
    w = text.split()
    return " ".join(w[:n]) + (" …" if len(w) > n else "")


def dat_alpha(sv: int, sb: int) -> float:
    if sv == 0 and sb == 0:
        a = 0.5
    elif sv == 5 and sb != 5:
        a = 1.0
    elif sb == 5 and sv != 5:
        a = 0.0
    else:
        a = sv / (sv + sb)
    return math.floor(a * 10 + 0.5) / 10


def parse(content: str):
    m = re.findall(r"\d+", content or "")
    if len(m) < 2:
        return None
    sv, sb = int(m[0]), int(m[1])
    return (sv, sb) if 0 <= sv <= 5 and 0 <= sb <= 5 else None


class Ollama:
    def __init__(self, host: str, model: str, cache: Path):
        self.host, self.model = host.rstrip("/"), model
        self.cache = {}
        if cache.exists():
            for line in cache.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.cache[r["key"]] = r
        self.fh = open(cache, "a", encoding="utf-8")

    def chat(self, user: str) -> dict:
        key = hashlib.sha256(json.dumps([self.model, DAT_VERSION, user, DAT_TEMPERATURE]).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        body = {"model": self.model, "stream": False, "messages": [{"role": "user", "content": user}],
                "options": {"temperature": DAT_TEMPERATURE, "seed": 0, "num_ctx": 4096}}
        req = urllib.request.Request(self.host + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0, msg = time.time(), ""
        for attempt in range(1, 4):
            try:
                with urllib.request.urlopen(req, timeout=900) as resp:
                    r = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")[:300]
                print(f"  ollama HTTP {e.code} (attempt {attempt}/3): {msg}", flush=True)
                time.sleep(10 * attempt)
        else:
            return {"key": key, "content": "", "prompt_tokens": 0, "completion_tokens": 0,
                    "seconds": round(time.time() - t0, 3), "error": msg}
        rec = {"key": key, "content": r["message"]["content"], "prompt_tokens": r.get("prompt_eval_count", 0),
               "completion_tokens": r.get("eval_count", 0), "seconds": round(time.time() - t0, 3)}
        self.cache[key] = rec
        self.fh.write(json.dumps(rec) + "\n")
        self.fh.flush()
        return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rankings", required=True, help="rankings.jsonl.gz of the BM25 run (actions dense_only, lex_only)")
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "qwen2.5:7b"))
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    corpus, queries = load_queries_only(data)
    tops = {}
    with gzip.open(a.rankings, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["action"] in ("dense_only", "lex_only"):
                tops.setdefault(r["qid"], {})[r["action"]] = r["top"]
    qids = [q for q in sorted(queries) if q in tops]
    llm = Ollama(a.host, a.model, out / "dat_cache.jsonl")
    rows, t0 = [], time.time()
    for qi, qid in enumerate(qids):
        d1, l1 = tops[qid]["dense_only"][0], (tops[qid]["lex_only"] or [None])[0]
        user = DAT_PROMPT.format(question=queries[qid], dense=snippet(corpus[d1], DAT_WORDS),
                                 bm25=snippet(corpus[l1], DAT_WORDS) if l1 else "(no passage retrieved)")
        r = llm.chat(user)
        s = parse(r["content"])
        rows.append({"qid": qid, "dat_valid": int(s is not None), "dat_sv": s[0] if s else "", "dat_sb": s[1] if s else "",
                     "dat_alpha": dat_alpha(*s) if s else "", "same_top1": int(d1 == l1), "reply": r["content"][:80],
                     "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                     "llm_seconds": r["seconds"]})
        if (qi + 1) % 10 == 0:
            print(f"  {qi + 1}/{len(qids)} queries, {time.time() - t0:.0f} s", flush=True)
    with open(out / "dat_features.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    manifest = {"data": str(data), "model": a.model, "rankings": a.rankings, "ollama_host": a.host,
                "dat_version": DAT_VERSION, "dat_words": DAT_WORDS, "temperature": DAT_TEMPERATURE,
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "queries": len(rows),
                "invalid": sum(1 - r["dat_valid"] for r in rows), "total_calls": len(rows),
                "total_llm_seconds": round(sum(r["llm_seconds"] for r in rows), 1), "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()

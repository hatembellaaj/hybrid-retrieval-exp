"""Shared helper: load a BEIR-format corpus and queries without qrels."""
import json
from pathlib import Path


def load_queries_only(data: Path):
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
    return corpus, queries

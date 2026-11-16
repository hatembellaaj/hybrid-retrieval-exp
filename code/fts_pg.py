"""Database full-text retriever on plain PostgreSQL (own implementation, paper Section 3.1).

Design (as described in the paper for the production platform):
  * index: tsvector with field weights, document identifier / file name = A, section headings = B, body = C,
    language-specific stemming (configurable, 'english' here); GIN index;
  * query, stage 1 (conjunctive): websearch_to_tsquery(<cfg>, q) OR-ed with websearch_to_tsquery('simple', q), so that
    a term surviving in either configuration can match (every term of the query required within a configuration);
  * stage 2 (disjunctive fallback, only if fewer than k hits): any single query lexeme matches;
  * ranking: ts_rank_cd (cover density, no collection-level IDF) with normalisation 32 (rank / (rank + 1), in [0, 1));
    every stage-1 hit is placed above every stage-2 hit by adding 1 to stage-1 scores, so scores are ordinal and bimodal;
  * strict_match = at least one stage-1 hit (used by the production rule A0).
"""
from __future__ import annotations

from dataclasses import dataclass

import psycopg

SCHEMA = """
CREATE TABLE IF NOT EXISTS fts_passage (
    collection text NOT NULL,
    doc_id     text NOT NULL,
    file_name  text,
    headings   text,
    body       text NOT NULL,
    tsv        tsvector NOT NULL,
    PRIMARY KEY (collection, doc_id)
);
CREATE INDEX IF NOT EXISTS fts_passage_tsv ON fts_passage USING gin (tsv);
"""

INSERT = """
INSERT INTO fts_passage (collection, doc_id, file_name, headings, body, tsv)
VALUES (%(c)s, %(id)s, %(f)s, %(h)s, %(b)s,
        setweight(to_tsvector(%(cfg)s::regconfig, coalesce(%(f)s, '')), 'A') ||
        setweight(to_tsvector(%(cfg)s::regconfig, coalesce(%(h)s, '')), 'B') ||
        setweight(to_tsvector(%(cfg)s::regconfig, %(b)s), 'C'))
"""

SEARCH = """
WITH q AS (
    SELECT websearch_to_tsquery(%(cfg)s::regconfig, %(q)s) || websearch_to_tsquery('simple', %(q)s) AS and_q,
           (SELECT string_agg(quote_literal(lex), ' | ')
              FROM unnest(tsvector_to_array(to_tsvector(%(cfg)s::regconfig, %(q)s)
                                            || to_tsvector('simple', %(q)s))) AS lex) AS or_text
),
strict AS (
    SELECT p.doc_id, 1.0 + ts_rank_cd(p.tsv, q.and_q, 32) AS score, true AS stage1
      FROM fts_passage p, q
     WHERE p.collection = %(c)s AND numnode(q.and_q) > 0 AND p.tsv @@ q.and_q
     ORDER BY score DESC, p.doc_id
     LIMIT %(k)s
),
loose AS (
    SELECT p.doc_id, ts_rank_cd(p.tsv, q.or_text::tsquery, 32) AS score, false AS stage1
      FROM fts_passage p, q
     WHERE p.collection = %(c)s AND q.or_text IS NOT NULL
       AND (SELECT count(*) FROM strict) < %(k)s
       AND p.tsv @@ q.or_text::tsquery
       AND p.doc_id NOT IN (SELECT doc_id FROM strict)
     ORDER BY score DESC, p.doc_id
     LIMIT %(k)s
)
SELECT doc_id, score, stage1 FROM (SELECT * FROM strict UNION ALL SELECT * FROM loose) u
 ORDER BY score DESC, doc_id
 LIMIT %(k)s
"""


@dataclass
class Outcome:
    scores: dict[str, float]
    strict_match: bool


class PgFts:
    def __init__(self, dsn: str, config: str = "english"):
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.config = config
        self.conn.execute(SCHEMA)

    def load(self, collection: str, rows: list[dict], batch: int = 2000) -> None:
        """rows: {doc_id, file_name, headings, body}."""
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM fts_passage WHERE collection = %s", (collection,))
            for i in range(0, len(rows), batch):
                cur.executemany(INSERT, [{"c": collection, "id": r["doc_id"], "f": r.get("file_name"), "h": r.get("headings"),
                                          "b": r["body"], "cfg": self.config} for r in rows[i:i + batch]])
            cur.execute("ANALYZE fts_passage")

    def search(self, collection: str, query: str, k: int = 50) -> Outcome:
        if not query.strip():
            return Outcome({}, False)
        rows = self.conn.execute(SEARCH, {"cfg": self.config, "q": query, "c": collection, "k": k}).fetchall()
        return Outcome({r[0]: float(r[1]) for r in rows}, any(r[2] for r in rows))

    def drop(self, collection: str) -> None:
        self.conn.execute("DELETE FROM fts_passage WHERE collection = %s", (collection,))

    def close(self) -> None:
        self.conn.close()

"""Frozen prompts and JSON schemas of the LLM controllers B1 and B2 (paper, Appendix A).

FROZEN on 2026-11-16, before any locked test split was evaluated. Any change must bump PROMPT_VERSION and be recorded
as an amendment in PROTOCOL.md.
"""
PROMPT_VERSION = "2026-11-16-v1"

B1_SYSTEM = (
    "You assist a hybrid search engine that combines a lexical retriever (exact word matching, good for drug names, "
    "numbers, units, codes and rare terms) and a semantic retriever (meaning matching, good for paraphrases, lay wording "
    "and descriptive questions). Given a user query, characterise it and choose how much weight the lexical retriever "
    "should get. Answer with JSON only."
)
B1_USER = (
    "Query: {query}\n\n"
    "Return a JSON object with:\n"
    "- query_type: one of \"exact_term\" (the answer depends on a precise name, number, dose, threshold or code), "
    "\"clinical_vignette\" (a described situation in everyday words), \"conceptual\" (asks why/what/difference), \"other\";\n"
    "- needs_exact_match: true if exact matching of some term is essential;\n"
    "- key_terms: up to 5 terms of the query that must be matched exactly (may be empty);\n"
    "- mode: \"lexical\", \"semantic\" or \"hybrid\";\n"
    "- lexical_weight: weight of the lexical retriever relative to the semantic one, one of 0, 0.25, 0.5, 0.75, 1."
)
B1_SCHEMA = {
    "type": "object",
    "properties": {
        "query_type": {"type": "string", "enum": ["exact_term", "clinical_vignette", "conceptual", "other"]},
        "needs_exact_match": {"type": "boolean"},
        "key_terms": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "mode": {"type": "string", "enum": ["lexical", "semantic", "hybrid"]},
        "lexical_weight": {"type": "number", "enum": [0, 0.25, 0.5, 0.75, 1]},
    },
    "required": ["query_type", "needs_exact_match", "key_terms", "mode", "lexical_weight"],
}

B2_SYSTEM = (
    "You grade search results. For each numbered passage, judge how useful it is to answer the query. "
    "Use 3 = answers the query directly, 2 = partially answers it, 1 = same topic but does not really help, "
    "0 = not relevant. Judge each passage on its own. Answer with JSON only."
)
B2_USER = "Query: {query}\n\nPassages:\n{passages}\n\nReturn {{\"grades\": [{{\"id\": <passage number>, \"grade\": <0-3>}}, ...]}} with one entry per passage."
B2_SCHEMA = {
    "type": "object",
    "properties": {"grades": {"type": "array", "items": {
        "type": "object", "properties": {"id": {"type": "integer"}, "grade": {"type": "integer", "minimum": 0, "maximum": 3}},
        "required": ["id", "grade"]}}},
    "required": ["grades"],
}
B2_TOP_M = 3              # passages per retriever
B2_SNIPPET_WORDS = 150
B2_SAMPLES = 3            # self-consistency samples
B2_TEMPERATURE = 0.7

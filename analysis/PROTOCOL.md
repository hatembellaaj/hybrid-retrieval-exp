# Protocol — risk-controlled hybrid retrieval (paper for *Autonomous Intelligent Systems*)

Status of this file: **frozen before any controller is fitted or any test split is opened.** Changes after that point
are appended in the "Amendments" section with a date and a reason; nothing above it is edited.

## 1. Research questions
- **RQ1 (headroom).** How large is the per-query oracle gain over the best fixed fusion, how much of it is flat
  (all actions tie), and on which query types does it concentrate? Per corpus, per lexical retriever.
- **RQ2 (controllers).** Do signal-based controllers (A1–A3) and LLM controllers (B1–B2) realise part of that gain, and
  do they beat the production rule A0?
- **RQ3 (risk).** Does the risk-controlled cascade C keep the expected degradation w.r.t. the fixed blend below ε
  (empirical check of the conformal guarantee), and at what cost in mean gain and LLM calls?
- **RQ4 (answers).** Do retrieval gains translate into answer faithfulness? (last, only on the haemophilia corpus) — **withdrawn, amendment 8**

## 2. Corpora and queries
| Corpus | Language | Role | Queries | Relevance |
|---|---|---|---|---|
| RAISE pilot fixture | fr | pilot (already run) | 186 | derived from reference answers |
| Haemophilia (hemorag, E3 extraction, structure chunking, 200 words) | en | clinical target | 60 answerable items of benchmark v0.1 | **silver**: passages containing a verbatim gold quote |
| NFCorpus, SciFact, TREC-COVID (BEIR test splits) | en | biomedical, power | 323 / 300 / 50 | official qrels |
| FiQA (BEIR test) | en | non-medical | 648 | official qrels |

**Known weakness, to be fixed before the haemophilia results are reported:** the 60 haemophilia questions were drafted
by an LLM from the passages that contain the gold quotes. Such questions over-share vocabulary with their target passage
and are expected to favour lexical retrieval. They are used for development only. The reported haemophilia set will be
**questions written by the clinician co-authors without looking at the corpus**, judged on pools (section 4).

## 3. Retrieval and action space
- Dense: `snowflake/snowflake-arctic-embed-s` (fastembed/ONNX, CPU), exact cosine; same model on every corpus.
- Lexical: (i) corpus-wide BM25 (k1 = 1.2, b = 0.75, Snowball stemming, as in the pilot); (ii) PostgreSQL FTS exactly as
  deployed in RAISE (`ts_rank_cd`, AND-first / OR-fallback), run on the server only.
- Candidate pool: top-50 per channel. Convex fusion uses **exact** scores of both channels for every candidate (no
  imputation); RRF uses ranks within each top-50 list (absent → 51), k = 60. Fusion code is a verbatim copy of RAISE's
  `score_fusion.py` (equivalence checked by `test_fusion_equivalence.py` whenever RAISE is importable).
- Actions: `dense_only`, `lex_only`, `rrf_w{0,.25,.5,.75,1}`, `cvx_a{0,.1,…,1}` (α = dense weight); A0 = production
  rule (FTS only: w_lex = 1 if the AND stage matched, else 0.5).

## 4. Relevance judgements (haemophilia, clinician-written questions)
Pools: top-20 of every compared system. Grades 0–3 by the haematologist co-authors, blind to the system; double
annotation on ≥ 20 % of pairs, weighted κ reported. LLM judgements only to extend pools, with their agreement reported.

## 5. Splits (per corpus, seed 20261109)
Queries are shuffled once: 50 % train (controller fitting), 25 % calibration (a*, λ̂, isotonic calibration), 25 % test,
evaluated once. The haemophilia clinician set is test-only; its controllers come from the other corpora and from the
development questions (leave-one-corpus-out also reported, with and without recalibration on 50 target queries).

## 6. Metrics and statistics
nDCG@10 (graded when qrels are graded), Recall@{5,10,20}, MRR@10; URisk/TRisk against a*; win/tie/loss; 5th percentile
of Δ; ECE/Brier of the confidence; LLM call rate, tokens, latency. Paired randomisation tests, Holm correction, bootstrap
95 % CIs over queries. Flat queries are reported, never dropped.

## 7. What is never done
No result is estimated, interpolated or hard-coded; a missing result is reported as missing. Test splits are opened
once, after prompts, action grid and signal set are frozen. LLM outputs never replace relevance judgements.

## Amendments
1. **2026-11-08 — a\* chosen on fit ∪ calibration, not on calibration alone.** With 25 % of a small corpus, the
   calibration set alone gives an unstable a\*. a\* is still never chosen on evaluation/test queries. Made before any
   test split was read.
2. **2026-11-08 — development protocol.** Before the single test run, controllers are developed on the 75 % non-test
   pool only, re-split 50 times into fit/calibration/evaluation (50/25/25 %); `controllers.py` never reads the locked
   test queries. The confidence threshold κ of Algorithm 1 is not used yet (deviation rule on λ only).
3. **2026-11-08 — haemophilia evaluation questions.** The 100 evaluation questions (CA-01..CA-50, CB-01..CB-50) were
   drafted with an AI assistant by the two haematologist co-authors, who selected and corrected them; they were not
   drafted from the corpus. This origin is declared in the paper. Hand-written exact-term questions may be added later
   as a separate, labelled stratum.
4. **2026-11-08 — pool depth 10 instead of 20.** Pools = union of the top-10 of all 18 actions (2,209 pairs; depth 20
   would need 4,152). nDCG@10, MRR@10 and Recall@5/10 of every action are fully judged; Recall@20 is not reported on
   this set. Each haematologist judges his/her own set plus 10 questions of the other set (20 questions doubly judged);
   passages are shuffled per question and shown without system, rank or score (make_judging_sheets.py).
5. **2026-11-09 — expert judgements on a blind subsample; LLM judgements for the rest.** The first returned files
   (2026-11-08) contained automatic AI pre-annotations for all 2,209 pairs, labelled "clinical evaluation not performed"
   (identical grades on the doubly judged questions); they are kept as LLM judgements only
   (judging/returned_20261108/PROVENANCE.md). Expert judgements are collected blind (no AI tool, no pre-grades, new pair
   ids and order) on 40 questions: 20 per set, 5 per theme, each expert judging his/her 20 plus 5 of the other set
   (10 questions doubly judged; judging_blind/). Primary clinical results use the expert subset; LLM–expert agreement is
   reported on it; results on all 100 questions with LLM grades are secondary and labelled as such. LLM model, version
   and prompt of the pre-annotation must be obtained and reported.
6. **2026-11-09 — expert role reduced to validating the LLM judge (supersedes the expert subsample of amendment 5).**
   The clinicians declined large judgement sets (maximum about 40 pairs each, in French). Relevance of all 2,209 pairs
   is taken from the LLM judgements; the experts validate them on 70 question–passage pairs sampled at most one per
   question and stratified by LLM grade (18/17/18/17 for grades 0/1/2/3), 40 pairs per expert including 10 shared
   (sample_validation_pairs.py, make_validation_sheets.py). Questions and passages are shown in French (translated by
   the first author's AI assistant, English originals shown alongside); LLM grades are hidden. Reported: weighted kappa
   expert–expert (10 pairs) and LLM–expert (70 pairs, also re-weighted to the grade prevalence of the full pool),
   confusion matrix. All clinical retrieval results are labelled "LLM-judged, expert-validated on 70 pairs".
   The files of amendment 5 (notation_aveugle_*) were not sent.
7. **2026-11-12 — pre-specified sensitivity analyses for the LLM judge (before any clinical retrieval result is
   computed).** First validation file (MC, 40 pairs, saved by a human user in Excel, no comments): LLM vs expert
   quadratic weighted kappa 0.48 [0.23, 0.70]; the LLM is more lenient (mean grade difference +0.65; at the threshold
   >= 2 it marks 20 pairs relevant against 8 for the expert, kappa 0.30). Consequently every clinical result will be
   reported under three relevance variants, fixed now: (a) raw LLM grades; (b) LLM grades mapped to the mean expert grade
   observed for each LLM grade in the validation sample (pooled over experts once both files are in); (c) strict binary
   relevance, relevant only if LLM grade = 3. Conclusions are stated only if they hold under all three.
8. **2026-11-12 — RQ4 (answer quality) withdrawn.** A blinded expert rating of generated answers is not feasible with
   the clinicians' available time. The paper evaluates retrieval only (RQ1–RQ3); answer quality is stated as a
   limitation and future work. LLM cost and latency of the controllers remain reported with RQ2/RQ3.
9. **2026-11-16 — LLM-judge validation closed at 40 pairs (one expert).** Final validation set = the 40 MC pairs graded
   by Maha Charfi (ten per LLM grade). The 30 ME-only pairs remain unjudged; no independent second expert: the FK file
   (2026-11-16) is a non-blind review of the MC grades (37/40 identical) and is not used for agreement
   (judging_validation/returned/PROVENANCE.md). The mapping of relevance variant (b) uses the MC grades only.
10. **2026-11-16 — LLM controllers frozen; server run.** Prompts and schemas of B1/B2 frozen in code/prompts.py
   (PROMPT_VERSION 2026-11-16-v1, model qwen2.5:7b via Ollama, local; B1 temperature 0; B2 grades the union of the
   top-3 dense and top-3 BM25 passages, shuffled with a fixed seed, 3 samples at temperature 0.7; alpha_L and w_lex_L
   from the DCG of each list). Server bundle: experiments/hybrid_server_bundle.tar.gz (sha256 33b7a3fe…). Rankings are
   saved without evaluation where no qrels exist. Known limitation: on the clinical set, FTS-only results may rank
   passages outside the judged depth-10 pools (built from the BM25/dense action space); their share will be reported,
   and unjudged passages are counted as non-relevant.
11. **2026-11-16 — production-platform code removed from the experiments.** The FTS lexical retriever is an
   independent re-implementation of the production design on plain PostgreSQL 16 (code/fts_pg.py: weights A/B/C,
   conjunctive stage then disjunctive fallback, ts_rank_cd normalisation 32, conjunctive hits offset by +1, A0 rule),
   and fusion.py is rewritten with identical conventions (equivalence test: 200/200 locally). The pilot keeps the
   production implementation. Checked on PostgreSQL 16.2 (embedded server) on hemophilia-dev and SciFact: same
   qualitative pattern as the pilot (FTS alone far below BM25; full FTS weight below dense-only; A0 above rrf_w1.0).
   Server code is in the private repository hybrid-retrieval-exp (supersedes the tarball of amendment 10).
12. **2026-11-17 — execution host of the LLM controllers.** On the shared server (load 17–19 on 12 cores), one call
   took 2–6 min (median 153 s), i.e. about a week for the plan. The LLM calls therefore run on the first author's Mac
   (Apple M1, 8 GB, Ollama 0.40.16 with Metal) for hemophilia-dev, hemophilia-clin, SciFact and NFCorpus, and on the
   server (Ollama 0.6.8, CPU) for TREC-COVID only. Same model (qwen2.5:7b, digest 845dbda0ea48), same frozen prompts
   (2026-11-16-v1), same top-3 lists (read from the saved rankings; checked identical to recomputation on 400 prompts).
   The 191 hemophilia-dev calls already made on the server are reused from cache. Outputs may differ slightly between
   hosts (sampling at temperature 0.7 is not bit-reproducible across hardware); the host of every call is recorded.
   FiQA is not run with the LLM controllers.

13. **2026-11-18 — all LLM controller calls on one host (supersedes the TREC-COVID part of amendment 12).** The
   server run of TREC-COVID was stopped before producing any call; TREC-COVID is run on the Mac like the other corpora
   (same model digest, same frozen prompts 2026-11-16-v1), reading the top-3 lists from the server rankings
   (results/trec-covid/rankings.jsonl.gz; BEIR trec-covid corpus, all 300 top-3 ids present). Every LLM controller
   output in the paper therefore comes from one host, except the 191 hemophilia-dev calls reused from the server cache
   (amendment 12). llm_features.py now retries an Ollama HTTP error three times before counting the call invalid (one
   HTTP 500 interrupted the SciFact run on 2026-11-17; no call was lost, the run resumed from cache).

14. **2026-11-19 — DAT baseline frozen before evaluation.** DAT (Hsu & Tzeng 2025) is re-implemented as published in
   code/dat_features.py (DAT_VERSION 2026-11-19-v1): one call per query, the LLM scores the top-1 dense and top-1 BM25
   passages 0–5 on the published rubric (prompt in our own wording, same structure and output format: two integers);
   alpha by the published case rules, rounded to one decimal. Choices the paper leaves open, fixed here: same model as
   B1/B2 (qwen2.5:7b, Mac host), temperature 0, passages truncated to 300 words, the first two integers of the reply
   are parsed (otherwise invalid → fixed action a*); with weighted RRF, w_lex = 1 − alpha snapped to the grid. The
   prompt and rules were fixed after reading the paper and before any DAT output was evaluated; the top-1 lists are read
   from the saved rankings (same as B2). Run on all five sets including hemophilia-clin (outputs not evaluated).
15. **2026-11-19 — FTS/A0 development analysis; hypotheses.** FTS results (server run of 2026-11-17, amendment 11) are
   analysed on the development pool with the same re-splits as controllers.py (fts_analysis.py); A0 is compared with the
   tuned fixed FTS weight a*_FTS and with full lexical weight. The hypotheses H1–H4 of the paper were written on this
   date, after the development results; they are stated in the paper as expectations from prior work, not as
   pre-registered hypotheses (the pre-registered items are RQ1–RQ3 of section 1).
16. **2026-11-21 — locked-test analysis frozen.** TEST_PLAN.md fixes the single test run (public corpora: one
   deterministic fit/calibration split of the development pool, test = locked 25 %; clinical set: everything chosen on
   the 60 development questions, λ̂ by CRC on 5-fold cross-fitted predictions, plus a secondary analysis recalibrated on
   50 clinical questions; three relevance variants; bootstrap CIs, sign-flip randomisation tests with Holm correction).
   Executed by locked_test.py --run, which refuses a second run and logs every attempt; the code was checked by
   --dry-run on development data only (pseudo-test inside the development pool; clinical branch on the development
   questions; utility recomputed from rankings identical to the stored matrix, max |diff| 1e-16). The clinical LLM
   grades were loaded to check the reader (2,209 pairs, 100 questions, grade counts equal to the known pool prevalence);
   no clinical retrieval metric was computed. SHA-256 (first 16 hex) at freezing: TEST_PLAN.md d52af94d0471dc5e,
   locked_test.py 33d09db882a8301e, controllers.py ab7e15ca96366f77, llm_controllers.py 21211fd04a4d1eed,
   utility_matrix_beir.py 708ce3f322a5184e, sample_validation_pairs.py 5e5f399889f4658b, agreement.json
   94d36faee7de722a; LLM features / DAT outputs: scifact a15a5ed3fb53c439 / 34a9a842d01883e8, nfcorpus
   11024dc46955eed9 / cc3ef3c2f3c2855a, trec-covid a5e2ef0b2a88b1db / 669303a798261392, hemophilia-dev
   68181fd63dfeccac / 3223d3cd1c8201a6, hemophilia-clin 146cb7dca882def0 / edace089cbbed228.

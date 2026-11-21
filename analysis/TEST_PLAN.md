# Locked-test analysis plan — frozen before any test query or clinical relevance grade is read

Written 2026-11-21, after the development results of PROTOCOL.md amendments 1–15 and before the locked test splits
are opened. Executed once by `locked_test.py --run`. Anything not listed here is not part of the confirmatory test;
any later analysis is labelled exploratory. Nothing in this file may be edited after the run; corrections go to
PROTOCOL.md as dated amendments.

## 1. Frozen components
- Retrievers, action grids, fusion code, signal set: as in the server run of 2026-11-17 (results/ and
  server_results/20261117_2021/results/, rankings and signals already computed for every query).
- LLM controllers: prompts 2026-11-16-v1 (B1, B2) and DAT 2026-11-19-v1, qwen2.5:7b digest 845dbda0ea48; outputs
  already computed for every query, including test queries and the clinical set (llm_mac/*-llm, *-dat).
- Controller code: controllers.py (A1, A2, A3, CRC rule) and llm_controllers.py (snapping, LLM features), unchanged.
- Risk levels ε ∈ {0.02, 0.05}. Metric nDCG@10 (graded where judgements are graded).

## 2. Public corpora (SciFact, NFCorpus, TREC-COVID, FiQA)
- Test set: the locked 25 % (sha256(seed 20261109 + qid)), test queries with ≥ 1 relevant document. Sizes are fixed by
  the hash: about 75 (SciFact), 80 (NFCorpus), 12 (TREC-COVID), 162 (FiQA).
- Development pool (the other 75 %) split ONCE, deterministically: queries ordered by sha256(seed + "fitcal" + qid);
  first two thirds = fit, last third = calibration (50 % / 25 % of the corpus, as in PROTOCOL section 5).
- a* = best fixed action on fit ∪ calibration. A1, A2, A3, A3+L fitted on fit. λ̂ of C and C+L by CRC on calibration.
  B1, B2, DAT zero-shot (no fitting).
- Configurations: BM25 × weighted RRF, BM25 × convex fusion, FTS × weighted RRF (A0 added; no LLM controller with FTS).
  FiQA: no LLM controllers (not run, amendment 12).
- Controllers: FIXED, A1, A2, A3, C(0.02), C(0.05), B1, B2, DAT, A3+L, C+L(0.02), C+L(0.05), A0 (FTS), ORACLE.

## 3. Clinical set (hemophilia-clin, 100 clinician-selected questions, test only)
- Relevance: the LLM grades of the 2,209 pooled pairs (judging/returned_20261108, amendment 5–6) under the three
  variants of amendment 7: (a) raw LLM grade 0–3; (b) LLM grade mapped to the mean expert grade observed for that LLM
  grade on the 40 validation pairs of Maha Charfi (judging_validation/agreement.json); (c) binary, relevant iff LLM
  grade = 3. Graded nDCG@10 with gain 2^g − 1 (b uses the real-valued mapped grade). Unjudged passages are
  non-relevant; the share of unjudged passages in the top-10 of each action is reported (FTS lists may leave the
  depth-10 pools, amendment 10). Questions without any relevant passage under a variant are excluded for that variant
  and counted.
- Primary analysis (no clinical data used for any choice): a* = best fixed action on the 60 haemophilia development
  questions; A1, A2, A3, A3+L fitted on the same 60 questions; λ̂ by CRC on the 60 development questions using 5-fold
  cross-fitted gain predictions (n = 60 ≥ 1/ε − 1 for both ε); B1, B2, DAT zero-shot. This measures the controllers and
  the guarantee under the shift from LLM-drafted development questions to clinician-written questions.
- Secondary analysis (recalibration on target questions, PROTOCOL section 5): clinical questions split once by
  sha256(seed + "clincal" + qid) into 50 calibration / 50 evaluation; a* and λ̂ re-chosen on the 50 calibration
  questions (controllers still fitted on the development questions); evaluation on the other 50.
- Reference only (not deployable): the in-sample best fixed action on the clinical set, to show the cost of the shift.
- Same configurations and controllers as section 2.

## 4. Statistics (every corpus × configuration × controller)
- Mean nDCG@10, mean Δ vs a*, win / loss / deviation shares, mean loss E[max(0, −Δ)].
- 95 % bootstrap CI of mean Δ over test queries (10,000 resamples, seed 20261121).
- Paired sign-flip randomisation test of mean Δ = 0 (two-sided, 10,000 permutations, seed 20261121); Holm correction
  over the adaptive controllers (all except FIXED and ORACLE) within each corpus × configuration (× variant).
- Risk check of C and C+L: mean loss on the test queries compared with ε, with the one-sided 95 % bootstrap upper
  bound.

## 5. Confirmatory statements decided by the run
1. H2/H3: an adaptive controller "beats the fixed blend" on a corpus only if its Holm-adjusted p < 0.05 and Δ > 0.
   On the clinical set, only if this holds under all three relevance variants.
2. A0 vs a*_FTS: same rule, per corpus.
3. H4: the guarantee "held" for a setting if the measured test mean loss ≤ ε. Reported for every setting; with ~12–162
   test queries this is a check of the order of magnitude, not a test of the expectation.
4. Development and test results are both reported; if they disagree, the test result is the one stated in the abstract.

# PAIR-Fuse: Qt/C++-Edu Dataset and Code

This repository accompanies the paper:

> **PAIR-Fuse: Prompt-guided Adaptive Integration of LLM and KG Embedding Rankings for Knowledge Graph Completion**
> Xiuxia Tian, Jingjing Wang, Ao-Ying Zhou

It contains the Qt/C++-Edu knowledge graph, the candidate sets, the prompts, and the code needed to reproduce every number in the paper.

## Repository layout

```
dataset/qt_cpp_kg/               Qt/C++-Edu knowledge graph (Chinese)
  train.txt / valid.txt / test.txt   triple splits (2,784 / 348 / 348)
  entity2id.txt, relation2id.txt     id mappings (2,803 entities, 16 relations)
  entity2text.txt                    entity names + short text descriptions
  relation2text.txt                  relation names + short text
  data.txt                           raw course text with extracted SPO lists (construction source)
  candidates/                        CLEAN candidate sets used in the paper
    candidates_{head,tail}.jsonl         retriever output, no ground-truth insertion (deployment)
    candidates_{head,tail}_eval.jsonl    same, with evaluation-only insertion appended at the
                                         end and per-query flags (GoldInjected)
  affected_protocol_candidates/      candidate files of the affected protocol (ground-truth
                                     inserted at the top), used only for the pitfalls analysis
  rotate/                            frozen RotatE checkpoint (dim 1280)
dataset/fb15k237/                FB15k-237 candidate cache for the 1,000-query sample and
  rotate/                          RotatE checkpoint (dim 256) + structural scores
prompts/prompts_clean.json       the two listwise prompts (P0 naive, P1 discipline-aware)
src/                             all scripts (see below)
outputs/qt_cpp_kg/               LLM outputs, structural scores, fused rankings, final metrics
outputs/fb15k237/                FB15k-237 outputs for the sanity check
first_pipeline_outputs/          outputs of our first (affected) pipeline, kept only so that
                                 the isolation analysis of the evaluation protocol pitfalls
                                 can be verified (Section V-A of the paper)
```

## What each script does

| script | purpose |
|---|---|
| `build_clean_candidates.py` | builds the clean candidate sets (character-bigram retrieval over entity descriptions, top-80, no ground-truth access) |
| `link_prediction_ds.py` | zero-training LLM ranker (OpenAI-compatible API; JSON-only output; deterministic fallback; resumable) |
| `export_struct_scores_from_rotate.py` | computes RotatE scores for the candidates of a prediction file |
| `fuse_llm_struct_rule.py` | fusion policies: Safe-RRF, weighted RRF, flip switch |
| `fuse_and_rerank.py` | linear fusion baseline |
| `kgc_eval.py` | evaluation (CJK-aware relaxed matching, multi-gold filtered) |
| `final_eval.py` | unified evaluation on unique queries, offline + deployment settings |
| `rotate_valid_eval.py` | full-graph filtered RotatE evaluation on the validation split |
| `analysis_battery.py` | bootstrap confidence intervals, McNemar tests, relation-group metrics |
| `eval_dirty.py`, `dirty_baselines.py` | the affected-protocol comparison of Section V-A |
| `clean_retriever_coverage.py` | retriever coverage measurement |

## Reproducing the main results

1. Candidate sets are provided in `dataset/qt_cpp_kg/candidates/`. To rebuild them: `python src/build_clean_candidates.py` (expects the dataset at `dataset/qt_cpp_kg/`).
2. LLM rankings are provided in `outputs/qt_cpp_kg/` (model: MiniMax-M2.5, temperature 0). To rerun them you need an OpenAI-compatible API key; see `python src/link_prediction_ds.py --help`.
3. Structural scores: `python src/export_struct_scores_from_rotate.py --task {head|tail} --pred <llm_output> --ckpt dataset/qt_cpp_kg/rotate/rotate_qt_dim1280_recip.pt --out <out> --dataset_dir dataset/qt_cpp_kg`.
4. Fusion: `python src/fuse_llm_struct_rule.py --input <llm_output> --struct <struct_scores> --output <out> --k_rrf 8 --weights 1.0,1.0 --blend_mode rrf` (and the other policies described in the paper).
5. Metrics: `python src/final_eval.py` writes `outputs/qt_cpp_kg/final_results.json` (the numbers of Table V); `python src/analysis_battery.py` writes the confidence intervals and group-wise metrics.

The affected-protocol numbers of Table VII are produced by `src/eval_dirty.py` and `src/dirty_baselines.py` on `dataset/qt_cpp_kg/affected_protocol_candidates/`.

## Notes

- The test split contains 348 triples, which give 203 unique tail queries and 319 unique head queries. All metrics in the paper count unique queries once.
- FB15k-237 is a public benchmark; we include only our candidate cache for the 1,000 sampled tail queries and the corresponding structural scores. The official splits can be obtained from the original source.
- No API keys are included in this repository.

## License

- Code: MIT License (see `LICENSE`).
- Qt/C++-Edu dataset and candidate sets: CC-BY 4.0 (see `DATA_LICENSE`).
- FB15k-237 remains subject to its original terms of use.

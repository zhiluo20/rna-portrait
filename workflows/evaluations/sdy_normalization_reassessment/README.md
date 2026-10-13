# SDY2823 frozen-model normalization reassessment

This is an exploratory negative result, not a new trained model or a submission-ready Cell study. The publication contains workflow code, checksum-bound provenance and aggregate metrics. It is **not a standalone full-runtime/data release**: the local rc4 runtime, retained count matrices, mapping, frozen queries and assay-derived judgments are required to repeat the participant-level analysis.

## What was tested

After historical cohort normalization was recovered, the unchanged model was reevaluated on the same 96 baseline donors. Size factors were estimated using either 23,323 mapped reference genes (primary) or all 58,302 measured Ensembl rows (sensitivity). The entire candidate cohort was used without labels; this is transductive normalization, not a training-only or prospective single-sample reference.

The run produced 384 complete portraits, with zero failures, and ranked all 30 questions with three rewrites each. Independent flow-based relevance grades are available for only Q01, Q02 and Q19 (T cells, B cells and their combination). Rewrites were averaged within question before averaging the three correlated questions. The other 27 questions remain unvalidated.

Seventeen prior methods/controls were retained. Ten fixed new conditions were added: direct and fused RNA embeddings, full portrait with and without released calibration in both normalization arms, and primary-arm gene-shuffle and sample-mismatch controls. The new shuffle permutes all 23,323 mapped gene values; the earlier CPM control permuted only measured selected features. They are different interventions.

## Result and scientific gate

| Method / primary condition | Mean nDCG@10 | Exact-tie expected nDCG@10 | High-grade candidates per 10 |
| --- | ---: | ---: | ---: |
| EPIC, fixed query-to-feature mapping | 0.800 | 0.800 | 6.667 |
| CellWhisperer | 0.337 | 0.337 | 2.556 |
| Full portrait, previous CPM condition | 0.395 | 0.338 | 3.222 |
| Full portrait, primary size-factor condition | 0.203 | 0.198 | 1.444 |
| Direct RNA, primary condition | 0.327 | 0.327 | 2.556 |
| Fused RNA, primary condition | 0.373 | 0.373 | 3.000 |
| Gene-value shuffle, primary condition | 0.411 | 0.411 | 3.333 |
| Sample mismatch, primary condition | 0.239 | 0.239 | 1.889 |

All 27 conditions and all three questions are retained in the aggregate CSV files. No favorable subset, normalization arm or tie-handling rule was selected after seeing outcomes. Both normalization arms produced identical full-portrait metrics; calibration on/off did not change the 96 retrieved texts in either arm. More distinct descriptions (56 versus the old 23) did not imply higher relevance or statement accuracy.

The planned feasibility gate is not passed. This run does not establish a practical retrieval advantage over the strong traditional comparator, and the single gene-shuffle control did not show the expected performance loss. It does not prove general model failure or identify its cause. EPIC and MCP-counter use manually fixed query-to-feature mappings and are not unrestricted natural-language parsers; that mapping cost was not measured. Do not expand confirmatory or wet-lab investment or tune these observed questions into apparent confirmation. Retain the negative result and reassess task/model fit first.

## Evidence to finding to next action

| Evidence | Finding | Next action |
| --- | --- | --- |
| 384 per-library records and input-route assertions | Both RNA branches used the declared transformed values; alternate normalization was disabled | Preserve the input contract and all failure records |
| 900 reconstructed score groups; maximum error 0 | Saved embeddings reproduce every new score matrix and ranking | Do not confuse arithmetic consistency with biological validity |
| 243 independently checked metric groups; maximum error 3.33e-16 | Metrics and candidate joins are internally verified | Report the full exploratory comparison |
| Prior 153 groups differ by at most 1.11e-16; original files unchanged | Historical outputs were preserved | Keep old CPM labels, not relabel them as recovered normalization |
| Poor primary results and non-degraded gene shuffle | Current three-query utility gate is not met | Diagnose without outcome-driven selection; revisit Cell positioning |

## Local reproduction requirements

The drivers accept explicit `--study-root`, `--input-root` and `--output-root` paths. The retained study layout includes:

- `rna-portrait/` with the separately verified **local rc4 full runtime** and `data_availability/full_runtime_contract_v4.json` (SHA-256 `7f0cef39f12e774a6a7b346a91490e7a3bdd84680a8ce402ba0814952b39e5eb`). The public compact API alone is insufficient.
- `restored_runtime_20261009_v2/model_artifacts/` with 14 hash-pinned assets, plus `released_data_package/processed_data/training_matrix/` with 139 complete shard pairs.
- `repair_v1/paired_candidates/SDY2823/` with the raw RNA matrix, unchanged map, baseline pool, original v3 rankings and frozen flow-derived qrels.
- `repair_v1/text_encoder_minilm/`, its existing encoder protocol, `repair_v1/pilot_rankings_v1/text_embeddings.npz`, and `queries.json`.
- The preparation driver expects the independently implemented normalizer under the retained study's `preprocessing_recovery_20261013/github_patch/workflows/preprocessing/` path. Its source is also available in PR #5.
- The input directory contains the unchanged `PROTOCOL.json`, `preparation.json` and `prepared_inputs.npz`. A new output directory must contain an appropriate `RERUN_AUTHORIZATION.json` provenance record before inference; that record is hashed into the execution contract.

Python 3.12 and exact runtime package versions are checked against the rc4 manifest before model loading. This run used the existing Python 3.12.14 environment, CPU, two PyTorch threads, and seed 20261013; no packages or checkpoints were replaced. Raw counts, participant-level predictions/judgments, weights and local absolute paths are not included in the public patch. These missing public components remain an explicit full-release reproducibility gate, not a claimed completed release.

Inspect the interfaces from a checkout with the declared dependencies:

```bash
python workflows/evaluations/sdy_normalization_reassessment/run_inference.py --help
python workflows/evaluations/sdy_normalization_reassessment/evaluate_reassessment.py --help
python workflows/evaluations/sdy_normalization_reassessment/verify_outputs.py --help
python -m unittest discover -s tests -p 'test_sdy_reassessment_metrics.py' -v
```

Execute in order: preparation (only for fresh inputs), inference, evaluation, independent verification, then the report notebook. Do not rerun preparation over the frozen inputs. Completed inference/evaluation outputs are hash-verified rather than overwritten; partial failures stop for inspection. Input, code or parameter changes require a new run version. `build_report.py` runs five notebook code cells and exports a bounded HTML report without installing nbconvert.

## Migration and remaining scope

The project was moved to the external SSD while the earlier run was being finalized. Its copied snapshot omitted those new outputs. The author expressly authorized an SSD rerun. The inaccessible earlier run was not read or restored, and bytewise agreement with it is not claimed. The new run and the submitted manuscript have separate version histories.

No new training, clinical validation, independent human statement rating, GitHub main-branch merge or journal submission occurred. Original anchor training-split provenance, exact historical environment, a prospective single-sample reference and genuinely held-out utility remain unresolved. Source and result files stay separate from the unchanged v1.12 manuscript.

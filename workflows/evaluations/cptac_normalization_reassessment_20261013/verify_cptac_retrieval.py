"""Independent evaluation audit; does not import the production metric helpers.

Reconstructs every candidate set from source rankings and protein qrels, checks
deterministic and sklearn tie-averaged nDCG, and independently rebuilds macros.
Does not rerun/train models, change queries, or substitute missing judgments.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import argparse
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score



def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def independently_measure(ids, scores, grades, cohorts, k=10):
    """Straight array formulas plus sklearn, not the production implementation."""
    ids, scores, grades, cohorts = map(np.asarray, (ids, scores, grades, cohorts))
    assert len(ids) == len(set(ids)) == len(scores) == len(grades) == len(cohorts)
    assert np.isfinite(scores).all()
    known = np.isfinite(grades)
    assert np.isin(grades[known], [0, 1, 2]).all()
    order = sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))
    top = np.array(order[:k], dtype=int)
    conditional_order = np.array([i for i in order if known[i]], dtype=int)
    d = 1 / np.log2(np.arange(k) + 2)
    result = dict(candidate_cases=len(ids), judged_cases=int(known.sum()),
                  judged_fraction=float(known.mean()),
                  full_pool_top10_judged=int(known[top].sum()),
                  full_pool_top10_judged_fraction=float(known[top].mean()),
                  full_pool_model_abstentions=int((scores <= -2).sum()))
    names = ["ndcg", "precision", "recall", "strong_discovery", "tie_expected_ndcg",
             "tie_expected_precision", "tie_expected_recall"]
    result.update({f"conditional_{name}_at_10": np.nan for name in names})
    result.update(conditional_cohorts_in_top10=np.nan, full_pool_precision_at_10=np.nan,
                  full_pool_ndcg_at_10=np.nan)
    ideal = np.sort(np.expm1(grades[known] * np.log(2)))[::-1][:k]
    ideal_dcg = float(ideal @ d[:len(ideal)])
    if len(conditional_order) >= k and ideal_dcg > 0:
        picked = conditional_order[:k]
        observed = grades[picked]
        result.update(
            conditional_ndcg_at_10=float((np.exp2(observed) - 1) @ d / ideal_dcg),
            conditional_precision_at_10=float(np.count_nonzero(observed) / k),
            conditional_recall_at_10=float(np.count_nonzero(observed) / np.count_nonzero(grades[known])),
            conditional_strong_discovery_at_10=float(np.count_nonzero(observed == 2) / k),
            conditional_cohorts_in_top10=len(set(cohorts[picked])),
            conditional_tie_expected_ndcg_at_10=float(ndcg_score(
                (np.exp2(grades[known]) - 1)[None, :], scores[known][None, :], k=k, ignore_ties=False)))
        # A top-k boundary tie gives each member an equal inclusion probability.
        cutoff = scores[conditional_order[k - 1]]
        above, boundary = known & (scores > cutoff), known & (scores == cutoff)
        remaining = k - int(above.sum())
        expected_positive = np.count_nonzero(grades[above]) + remaining * np.mean(grades[boundary] > 0)
        result["conditional_tie_expected_precision_at_10"] = float(expected_positive / k)
        result["conditional_tie_expected_recall_at_10"] = float(expected_positive / np.count_nonzero(grades[known]))
    if len(ids) >= k and known[top].all():
        result["full_pool_precision_at_10"] = float(np.mean(grades[top] > 0))
        if (known.all() or np.count_nonzero(grades[known] == 2) >= k) and ideal_dcg > 0:
            result["full_pool_ndcg_at_10"] = float((np.exp2(grades[top]) - 1) @ d / ideal_dcg)
    return result


def verify(study_root, reassessment_root, write=True):
    STUDY=study_root.resolve(); ROOT=STUDY/'repair_v1'
    COHORT=ROOT/'paired_candidates/CPTAC_PanCancer'
    NEW=reassessment_root.resolve(); EVALUATION=NEW/'evaluation_v1'
    contract = json.loads((EVALUATION / "contract.json").read_text())
    for relative, expected in contract["source_sha256"].items():
        assert sha(STUDY / relative) == expected, relative
    pool = pd.read_csv(COHORT / "rna_inputs_v1/mechanically_valid_candidates.csv").set_index("public_case_id")
    ids = pool.index.to_numpy(str)
    assert len(pool) == 942 and pool.index.is_unique
    parts = []
    for folder in ["rna_predictions_v1", "cellwhisperer_v1", "conventional_rankings_v1", "full_portrait_v1"]:
        part = pd.read_csv(COHORT / folder / "rankings.csv.gz", float_precision="round_trip")
        parts.append(part)
    parts.append(pd.read_csv(NEW/'inference_v1/rankings_all_30_queries.csv.gz',float_precision='round_trip'))
    rankings = pd.concat(parts, ignore_index=True)
    keys = ["method", "query_id", "variant", "sample_id"]
    assert not rankings.duplicated(keys).any()
    assert len(rankings) == 32 * 90 * 942
    matrices = {}
    for method, part in rankings.groupby("method"):
        matrices[method] = part.pivot(index=["query_id", "variant"], columns="sample_id", values="score").reindex(columns=ids)
        assert matrices[method].shape == (90, 942)
        assert np.isfinite(matrices[method].to_numpy()).all()
    qrels = pd.read_csv(COHORT / "protein_concordance_v2/qrels.csv")
    assert not qrels.duplicated(["scope", "cohort", "query_id", "sample_id"]).any()
    labels = {(scope, cohort): part.pivot(index="query_id", columns="sample_id", values="grade")
              for (scope, cohort), part in qrels.groupby(["scope", "cohort"])}
    contexts = [("pooled", "all", "all", ids)]
    for stratum, status in [("prior_alignment_test", "known_alignment_test_case"),
                            ("no_recorded_case_match", "unresolved_no_case_match_in_this_crosswalk")]:
        contexts.append(("pooled", "all", stratum, pool.index[pool.exposure_status.eq(status)].to_numpy(str)))
    contexts.extend(("within_cohort", cancer, "all", block.index.to_numpy(str)) for cancer, block in pool.groupby("cohort"))
    rebuilt = []
    for scope, cancer, stratum, subset in contexts:
        truth = labels[scope, cancer].reindex(columns=subset)
        cohorts = pool.loc[subset, "cohort"].to_numpy(str)
        for method, matrix in matrices.items():
            for (query, variant), row in matrix[subset].iterrows():
                result = independently_measure(subset, row.to_numpy(), truth.loc[query].to_numpy(), cohorts)
                rebuilt.append(dict(scope=scope, cohort=cancer, stratum=stratum, method=method,
                                    query_id=query, variant=variant, **result))
        print("Independently checked", scope, cancer, stratum, flush=True)
    group_keys = ["scope", "cohort", "stratum", "method", "query_id", "variant"]
    actual = pd.DataFrame(rebuilt).set_index(group_keys).sort_index()
    saved = pd.read_csv(EVALUATION / "per_query_paraphrase_results.csv").set_index(group_keys).sort_index()
    assert actual.index.equals(saved.index) and len(actual) == 37440
    differences = {}
    for metric in actual:
        a, b = actual[metric].to_numpy(float), saved[metric].to_numpy(float)
        assert np.array_equal(np.isnan(a), np.isnan(b)), f"Missingness differs: {metric}"
        assert np.allclose(a, b, atol=1e-12, rtol=0, equal_nan=True), metric
        differences[metric] = float(np.nanmax(abs(a - b)))
    macro_names = ["conditional_ndcg_at_10", "conditional_precision_at_10", "conditional_recall_at_10",
                   "conditional_strong_discovery_at_10", "conditional_tie_expected_ndcg_at_10",
                   "conditional_tie_expected_precision_at_10", "conditional_cohorts_in_top10", "full_pool_top10_judged_fraction"]
    # Independent explicit nested averaging: paraphrases -> cohorts -> questions.
    macro_rows = []
    for (scope, stratum, method), block in actual.reset_index().groupby(["scope", "stratum", "method"]):
        query_means = []
        for query, query_block in block.groupby("query_id"):
            cohort_means = [sub[macro_names].mean() for _, sub in query_block.groupby("cohort")]
            query_means.append(pd.DataFrame(cohort_means).mean())
        q = pd.DataFrame(query_means)
        macro_rows.append(dict(scope=scope, stratum=stratum, method=method, **q.mean().to_dict(),
                               evaluable_queries=int(q.conditional_ndcg_at_10.notna().sum())))
    macro_keys = ["scope", "stratum", "method"]
    macro = pd.DataFrame(macro_rows).set_index(macro_keys).sort_index()
    old_macro = pd.read_csv(EVALUATION / "descriptive_macro_results.csv").set_index(macro_keys).sort_index()
    assert macro.index.equals(old_macro.index)
    assert np.allclose(macro[old_macro.columns], old_macro, atol=1e-12, rtol=0, equal_nan=True)
    # Independent raw embedding -> similarity -> saved ranking parity, all 9 conditions.
    embeddings = np.load(COHORT / "rna_predictions_v1/embeddings.npz")
    queries = np.load(ROOT / "pilot_rankings_v1/text_embeddings.npz")
    query_index = pd.MultiIndex.from_arrays([queries["query_id"], queries["variant"]], names=["query_id", "variant"])
    score_differences = {}
    for method in embeddings.files:
        if method == "public_case_id":
            continue
        calculated = queries["query_rna"] @ embeddings[method].T
        stored = matrices[method].reindex(index=query_index, columns=embeddings["public_case_id"]).to_numpy()
        assert np.allclose(calculated, stored, atol=1e-7, rtol=0), method
        score_differences[method] = float(np.max(abs(calculated - stored)))
    for file,axis in [('rna_embeddings.npz','query_rna'),('portrait_embeddings.npz','query_raw')]:
        added=np.load(NEW/'inference_v1'/file)
        for method in added.files:
            if not method.startswith(('rna_','full_portrait_')):continue
            calculated=queries[axis] @ added[method].T
            stored=matrices[method].reindex(index=query_index,columns=added['sample_id']).to_numpy()
            assert np.allclose(calculated,stored,atol=1e-7,rtol=0),method
            score_differences[method]=float(np.max(abs(calculated-stored)))
    added_rna=np.load(NEW/'inference_v1/rna_embeddings.npz')
    old_assignments=pd.read_csv(COHORT/'rna_predictions_v1/mismatch_assignments.csv').set_index('assigned_case').loc[added_rna['sample_id']]
    assert np.array_equal(added_rna['mismatched_source_sample_id'],old_assignments.global_expression_source_case.to_numpy(str))
    assert np.array_equal(added_rna['within_cohort_source_sample_id'],old_assignments.within_cohort_expression_source_case.to_numpy(str))
    direct=added_rna['rna_direct_sf_reference']
    case_index=pd.Index(added_rna['sample_id'])
    for key,method in [('mismatched_source_sample_id','rna_mismatched_global_sf_reference'),
                       ('within_cohort_source_sample_id','rna_mismatched_within_cohort_sf_reference')]:
        idx=case_index.get_indexer(added_rna[key])
        assert (idx>=0).all() and (idx!=np.arange(942)).all() and len(set(idx))==942
        assert np.array_equal(added_rna[method],direct[idx])
    stability=pd.read_csv(EVALUATION/'paraphrase_stability.csv')
    stability=stability.set_index(['scope','cohort','stratum','method','query_id'])
    max_stability_error=0.; checked_stability=0
    for scope,cancer,stratum,subset in contexts:
        truth=labels[scope,cancer].reindex(columns=subset)
        for method,matrix in matrices.items():
            for query in truth.index:
                good=np.isfinite(truth.loc[query].to_numpy())
                eligible=np.array(subset)[good]
                observed=stability.loc[(scope,cancer,stratum,method,query)]
                assert observed.judged_cases==len(eligible)
                if len(eligible)<10:
                    assert np.isnan(observed.conditional_top10_jaccard)
                else:
                    tops=[]
                    for variant in [1,2,3]:
                        row=matrix.loc[(query,variant)]
                        tops.append(set(sorted(eligible,key=lambda i:(-row[i],i))[:10]))
                    val=sum(len(a.intersection(b))/len(a.union(b)) for a,b in itertools.combinations(tops,2))/3
                    error=abs(val-observed.conditional_top10_jaccard)
                    assert error<=1e-12
                    max_stability_error=max(max_stability_error,error)
                checked_stability+=1
    assert checked_stability==len(stability)==12480
    report = dict(checked_utc=datetime.now(timezone.utc).isoformat(), status="passed",
                  scope="Arithmetic, indexing, score direction, shared candidate masks and equal-query macros; not complete biological validation",
                  independent_of_production_metric_helpers=True, metric_groups=len(actual), methods=len(matrices),
                  contexts=len(contexts), source_contract_sha256=sha(EVALUATION / "contract.json"),
                  verifier_sha256=sha(__file__), max_metric_abs_difference=differences,
                  rna_similarity_abs_differences=score_differences,
                  paraphrase_stability_groups=checked_stability,max_stability_abs_error=max_stability_error,
                  old_and_new_mismatch_assignments_identical=True,
                  undefined_conditional_groups=int(actual.conditional_ndcg_at_10.isna().sum()),
                  unreported_full_pool_ndcg_groups=int(actual.full_pool_ndcg_at_10.isna().sum()),
                  caveats=["Measurability is a nonrandom filter; conditional scores are not full-corpus retrieval efficacy.",
                           "Protein abundance panels do not establish cell composition, function or individual statement correctness.",
                           "Weak performance is not explained by an arithmetic or saved-matrix indexing discrepancy in these checks.",
                           "No exposure-free external or Cell-suitability conclusion follows from this audit."])
    if write:
        (EVALUATION / "independent_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report, macro


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root',type=Path,required=True)
    parser.add_argument('--reassessment-root',type=Path,required=True)
    args=parser.parse_args()
    report, macro = verify(args.study_root,args.reassessment_root)
    print(json.dumps(report, indent=2))
    print(macro[["conditional_ndcg_at_10", "conditional_precision_at_10"]].to_string())

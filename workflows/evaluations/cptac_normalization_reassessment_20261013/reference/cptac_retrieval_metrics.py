"""Metrics with explicit incomplete-evidence boundaries and exact score ties."""
import math
import numpy as np
from tie_aware_retrieval import expected_metrics


def measure(ids,scores,grades,cohorts,k=10):
    ids=np.asarray(ids,dtype=str);scores=np.asarray(scores,dtype=float);grades=np.asarray(grades,dtype=float);cohorts=np.asarray(cohorts,dtype=str)
    if not (len(ids)==len(scores)==len(grades)==len(cohorts)) or len(set(ids))!=len(ids):raise ValueError('Mismatched or duplicated candidate IDs')
    if not np.isfinite(scores).all():raise ValueError('Nonfinite predictions')
    judged=np.isfinite(grades)
    if not np.isin(grades[judged],[0,1,2]).all():raise ValueError('Invalid grades')
    n=len(ids);top=np.lexsort((ids,-scores))[:k]
    result={'candidate_cases':n,'judged_cases':int(judged.sum()),'judged_fraction':float(judged.mean()) if n else None,
            'full_pool_top10_judged':int(judged[top].sum()),'full_pool_top10_judged_fraction':float(judged[top].mean()) if len(top) else None,
            'full_pool_precision_at_10':None,'full_pool_ndcg_at_10':None,'full_pool_metric_status':'incomplete_top10_judgments',
            'conditional_ndcg_at_10':None,'conditional_precision_at_10':None,'conditional_recall_at_10':None,
            'conditional_strong_discovery_at_10':None,'conditional_tie_expected_ndcg_at_10':None,
            'conditional_tie_expected_precision_at_10':None,'conditional_tie_expected_recall_at_10':None,
            'conditional_cohorts_in_top10':None,'conditional_status':'insufficient_judged_pool',
            'full_pool_model_abstentions':int((scores<=-2).sum())}
    discount=1/np.log2(np.arange(k)+2)
    ideal=np.sort(grades[judged])[::-1][:k]
    ideal_dcg=float(np.sum((2**ideal-1)*discount[:len(ideal)]))
    if n>=k and judged[top].all():
        result['full_pool_precision_at_10']=float(np.mean(grades[top]>0))
        # Unknown cases outside top-k cannot change the ideal if k known grade-2
        # cases already exist. Otherwise do not invent a full-corpus denominator.
        if judged.all() or int((grades[judged]==2).sum())>=k:
            result['full_pool_ndcg_at_10']=float(np.sum((2**grades[top]-1)*discount)/ideal_dcg) if ideal_dcg else None
            result['full_pool_metric_status']='evaluated' if ideal_dcg else 'no_relevant_cases'
        else:result['full_pool_metric_status']='precision_known_but_ideal_ranking_unresolved'
    if judged.sum()>=k and ideal_dcg>0:
        idx=np.flatnonzero(judged)
        order=idx[np.lexsort((ids[idx],-scores[idx]))][:k]
        actual=grades[order]
        records=[(ids[i],float(scores[i]),int(grades[i])) for i in idx]
        tied=expected_metrics(records,k=k)
        result.update(conditional_ndcg_at_10=float(np.sum((2**actual-1)*discount)/ideal_dcg),
                      conditional_precision_at_10=float(np.mean(actual>0)),conditional_recall_at_10=float(np.sum(actual>0)/np.sum(grades[judged]>0)),
                      conditional_strong_discovery_at_10=float(np.mean(actual==2)),conditional_tie_expected_ndcg_at_10=tied['tie_expected_ndcg_at_10'],
                      conditional_tie_expected_precision_at_10=tied['tie_expected_precision_at_10'],conditional_tie_expected_recall_at_10=tied['tie_expected_recall_at_10'],
                      conditional_cohorts_in_top10=len(set(cohorts[order])),conditional_status='evaluated_on_common_measurable_pool',
                      conditional_cutoff_tie_size=tied['cutoff_tie_size'])
    elif judged.sum()>=k:result['conditional_status']='no_relevant_judged_cases'
    return result

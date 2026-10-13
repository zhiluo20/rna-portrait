"""Independent arithmetic/lineage checks of the bounded SDY diagnosis."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import ndcg_score


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify(study_root,output_root):
    r,out=Path(study_root).resolve(),Path(output_root).resolve()
    protocol=json.loads((out/'PROTOCOL.json').read_text())
    inference=json.loads((out/'inference_summary.json').read_text())
    summary=json.loads((out/'analysis_summary.json').read_text())
    contract=json.loads((out/'analysis_contract.json').read_text())
    assert sha(out/'analysis_contract.json')==summary['analysis_contract_sha256']
    assert sha(out/'PROTOCOL.json')==contract['protocol_sha256']
    assert sha(out/'analyze_controls.py')==contract['driver_sha256']
    for payload in [inference,summary]:
        for name,expected in payload['output_sha256'].items():assert sha(out/name)==expected
    for name,expected in contract['source_sha256'].items():assert sha(r/name)==expected
    scores=np.load(out/'scores.npz');embeddings=np.load(out/'embeddings.npz')
    query=np.load(r/'repair_v1/pilot_rankings_v1/text_embeddings.npz')
    ids=scores['library_id'].astype(str);assert len(set(ids))==len(ids)==96
    max_score_error=0.
    for family in ['unmodified',*protocol['control_families']]:
        banks=embeddings[family][None,:,:] if family=='unmodified' else embeddings[family]
        saved=scores[family][None,:,:] if family=='unmodified' else scores[family]
        for z,wanted in zip(banks,saved):
            actual=query['query_rna']@z.T
            max_score_error=max(max_score_error,float(np.max(np.abs(actual-wanted))))
    assert max_score_error==0
    qrels=pd.read_csv(r/'repair_v1/paired_candidates/SDY2823/composition_pilot_v1/qrels.csv')
    truth=qrels.set_index(['query_id','sample_id']).grade.to_dict()
    metrics=pd.read_csv(out/'control_metrics_all_variants.csv',dtype={'variant':str},float_precision='round_trip')
    qi_lookup={(str(q),str(v)):i for i,(q,v) in enumerate(zip(scores['query_id'],scores['variant']))}
    seeds=scores['seeds'].tolist();errors=[]
    for row in metrics.itertuples():
        qi=qi_lookup[(row.query_id,row.variant)]
        values=scores['unmodified'][qi] if row.family=='unmodified' else scores[row.family][seeds.index(row.seed),qi]
        g=np.array([truth[(row.query_id,s)] for s in ids])
        gains=2.0**g-1;idx=np.lexsort((ids,-values))[:10]
        denominator=sum(sorted(gains,reverse=True)[i]/math.log2(i+2) for i in range(10))
        primary=sum(gains[j]/math.log2(i+2) for i,j in enumerate(idx))/denominator
        ties=float(ndcg_score(gains[None,:],values.astype(float)[None,:],k=10,ignore_ties=False))
        errors.extend([abs(primary-row.ndcg_at_k),abs(ties-row.tie_expected_ndcg_at_10),
                       abs(float(np.mean(g[idx]>0))-row.precision_at_k),
                       abs(float(np.mean(g[idx]==2))-row.upper_quartile_discovery_at_10)])
    assert len(metrics)==567 and max(errors)<1e-12
    recomputed=metrics.groupby(['family','seed','query_id']).ndcg_at_k.mean().groupby(['family','seed']).mean()
    stored=pd.read_csv(out/'control_replicate_means.csv',float_precision='round_trip').set_index(['family','seed'])
    assert np.allclose(recomputed,stored.loc[recomputed.index,'ndcg_at_k'],rtol=0,atol=1e-12)
    for row in summary['control_distributions']:
        values=recomputed.loc[row['family']].to_numpy()
        assert np.median(values)==row['median'] and int(np.sum(values>=summary['normal_Linux_ndcg']))==row['replicates_at_least_unmodified']
    null=np.load(out/'sample_label_randomization.npz');permutations=null['permutations']
    assert permutations.shape==(10000,96) and np.array_equal(np.sort(permutations,axis=1),np.tile(np.arange(96),(10000,1)))
    parent=r/'normalization_reassessment_20261013/ssd_rerun_v1'
    parent_ranks=pd.read_csv(parent/'rankings_measured_three_queries.csv.gz',dtype={'variant':str},float_precision='round_trip')
    by_group={key:group for key,group in parent_ranks.groupby(['method','query_id','variant'])}
    index={s:i for i,s in enumerate(ids)}
    qids=['Q01','Q02','Q19'];gains={q:np.array([2.0**truth[(q,s)]-1 for s in ids]) for q in qids}
    discounts=1/np.log2(np.arange(2,12));idcgs={q:np.dot(np.sort(gains[q])[::-1][:10],discounts) for q in qids}
    max_null_error=0.
    # Directly rerank all 27 methods on 20 deterministic draws, independent of
    # the coefficient-matrix implementation. No draw selected by its value.
    for draw in range(0,10000,500):
        permutation=permutations[draw]
        for mi,method in enumerate(null['methods'].astype(str)):
            values=[]
            for qid in qids:
                for variant in ['1','2','3']:
                    top=by_group[(method,qid,variant)].sort_values(['score','sample_id'],ascending=[False,True]).head(10).sample_id
                    permuted_gains=[gains[qid][permutation[index[s]]] for s in top]
                    values.append(sum(v/math.log2(i+2) for i,v in enumerate(permuted_gains))/idcgs[qid])
            max_null_error=max(max_null_error,abs(float(np.mean(values))-float(null['means'][draw,mi])))
    assert max_null_error<1e-12
    analytic=float(np.mean([gains[q].mean()*discounts.sum()/idcgs[q] for q in qids]))
    assert abs(analytic-summary['analytic_random_ranking_mean'])<1e-12
    # Geometry cross-check uses Gram-matrix eigenvalues instead of SVD.
    saved=np.load(parent/'rna_embeddings.npz')
    old=np.load(r/'repair_v1/paired_candidates/SDY2823/rna_predictions_v1/embeddings.npz')
    lookup={s:i for i,s in enumerate(old['library_id'].astype(str))}
    banks={'old_CPM_direct':old['rna_direct'][[lookup[s] for s in ids]],'old_CPM_fused':old['rna_fused'][[lookup[s] for s in ids]],
           **{name:saved[name] for name in saved.files if name.startswith('rna_')}}
    geometry=pd.read_csv(out/'embedding_geometry.csv',float_precision='round_trip').set_index('condition')
    geometry_error=0.
    for name,z in banks.items():
        z=z.astype(float);z/=np.linalg.norm(z,axis=1,keepdims=True);z-=z.mean(axis=0)
        eigen=np.maximum(np.linalg.eigvalsh(z@z.T),0)
        dimension=float(eigen.sum()**2/np.sum(eigen**2));first=float(eigen[-1]/eigen.sum())
        geometry_error=max(geometry_error,abs(dimension-geometry.loc[name,'centered_participation_dimension']),abs(first-geometry.loc[name,'first_PC_variance_fraction']))
    assert geometry_error<1e-10
    correlations=pd.read_csv(out/'cross_query_score_correlations.csv',float_precision='round_trip')
    corr_error=0.
    for row in correlations.itertuples():
        vectors=[]
        for qid in [row.query_a,row.query_b]:
            values=[by_group[(row.method,qid,v)].set_index('sample_id').loc[ids].score.to_numpy() for v in ['1','2','3']]
            vectors.append(rankdata(np.mean(values,axis=0)))
        if any(np.ptp(v)==0 for v in vectors):assert pd.isna(row.mean_score_spearman)
        else:corr_error=max(corr_error,abs(float(np.corrcoef(*vectors)[0,1])-row.mean_score_spearman))
    assert corr_error<1e-12
    clipping=pd.read_csv(out/'preprocessing_sample_diagnostics.csv')
    control_inputs=pd.read_csv(out/'control_input_diagnostics.csv')
    return {'score_groups_reconstructed':63*90,'max_score_error':max_score_error,'control_metric_groups_independently_checked':567,
        'max_metric_error':max(errors),'all_10000_sample_permutations_are_bijections':True,
        'independent_null_draw_method_checks':20*27,'max_direct_null_error':max_null_error,
        'geometry_eigenvalue_max_error':geometry_error,'query_score_rank_correlation_max_error':corr_error,
        'analytic_random_mean':analytic,'old_source_files_unchanged':True,
        'preprocessing_upper_clip_max':float(clipping.upper_clip_fraction.max()),
        'preprocessing_lower_clip_max':float(clipping.lower_clip_fraction.max()),
        'control_upper_clip_max':float(control_inputs.upper_clip_fraction_measured.max()),
        'control_lower_clip_max':float(control_inputs.lower_clip_fraction_measured.max()),
        'scope':'Arithmetic/data-lineage verification, not general model causality or independent confirmation'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args();report=verify(a.study_root,a.output_root)
    target=a.output_root/'independent_validation.json'
    if target.exists():assert json.loads(target.read_text())==report
    else:
        with target.open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

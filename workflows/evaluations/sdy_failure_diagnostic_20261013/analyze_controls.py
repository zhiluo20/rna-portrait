"""Evaluate all fixed controls and descriptive diagnostics after inference."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import itertools
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def safe_spearman(left,right):
    if len(np.unique(left))<2 or len(np.unique(right))<2:return None
    value=float(spearmanr(left,right).statistic)
    assert np.isfinite(value)
    return value


def unit_rows(z):
    z=np.asarray(z,dtype=float)
    return z/np.linalg.norm(z,axis=1,keepdims=True)


def geometry(z):
    z=unit_rows(z);cosines=(z@z.T)[np.triu_indices(len(z),1)]
    squared=np.linalg.svd(z-z.mean(axis=0),compute_uv=False)**2
    total=squared.sum()
    assert total>0
    return {'pair_cosine_min':float(cosines.min()),'pair_cosine_median':float(np.median(cosines)),
        'pair_cosine_mean':float(cosines.mean()),'pair_cosine_p95':float(np.quantile(cosines,.95)),
        'centered_participation_dimension':float(total**2/np.sum(squared**2)),
        'first_PC_variance_fraction':float(squared[0]/total),'centered_sum_squares':float(total)}


def run(study_root,output_root):
    r,out=Path(study_root).resolve(),Path(output_root).resolve()
    parent=r/'normalization_reassessment_20261013/ssd_rerun_v1'
    cohort=r/'repair_v1/paired_candidates/SDY2823'
    protocol=json.loads((out/'PROTOCOL.json').read_text())
    inference=json.loads((out/'inference_summary.json').read_text())
    assert inference['labels_read'] is False and inference['failed_inferences']==0
    assert inference['contract_sha256']==sha(out/'inference_contract.json')
    for name,expected in inference['output_sha256'].items():assert sha(out/name)==expected
    qrel_path=cohort/'composition_pilot_v1/qrels.csv'
    assert sha(qrel_path)==protocol['qrels_sha256']
    rank_path=parent/'rankings_measured_three_queries.csv.gz'
    metric_path=parent/'three_question_descriptive_means.csv'
    helper_path=parent/'retrieval_metrics.py'
    old_embedding_path=cohort/'rna_predictions_v1/embeddings.npz'
    sources=[qrel_path,rank_path,metric_path,helper_path,old_embedding_path,
             parent/'rna_embeddings.npz',parent.parent/'prepared_inputs.npz',
             r/'repair_v1/pilot_rankings_v1/text_embeddings.npz']
    contract={'created_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(out/'PROTOCOL.json'),
        'driver_sha256':sha(__file__),'inference_summary_sha256':sha(out/'inference_summary.json'),
        'source_sha256':{str(p.relative_to(r)):sha(p) for p in sources},
        'label_access_after_inference':True,'scope':'Post hoc diagnostics; no inferential superiority P values or new independent samples'}
    with (out/'analysis_contract.json').open('x') as stream:stream.write(json.dumps(contract,indent=2)+'\n')
    sys.path[:0]=[str(parent),str(r/'rna-portrait')]
    from retrieval_metrics import metrics_for_ranking
    data=np.load(parent.parent/'prepared_inputs.npz')
    ids=data['library_id'].astype(str);index={s:i for i,s in enumerate(ids)}
    new=np.load(out/'scores.npz');z=np.load(out/'embeddings.npz')
    assert np.array_equal(new['library_id'],ids) and np.array_equal(z['library_id'],ids)
    qrels=pd.read_csv(qrel_path)
    assert len(qrels)==288 and not qrels.duplicated(['query_id','sample_id']).any()
    assert qrels.grade.isin([0,1,2]).all()
    truth=qrels.set_index(['query_id','sample_id']).grade.to_dict()
    qids=['Q01','Q02','Q19'];qid_index={q:i for i,q in enumerate(qids)}
    gains=np.array([[2.0**truth[(q,s)]-1 for s in ids] for q in qids])
    grade=np.array([[truth[(q,s)] for s in ids] for q in qids],dtype=int)
    positions=[i for i,q in enumerate(new['query_id']) if q in qids]
    assert len(positions)==9
    rows=[];independent_errors=[]
    conditions=[('unmodified',-1,new['unmodified'])]
    for family in protocol['control_families']:
        conditions += [(family,int(seed),new[family][rep]) for rep,seed in enumerate(new['seeds'])]
    for family,seed,scores in conditions:
        for qi in positions:
            qid,variant=str(new['query_id'][qi]),str(new['variant'][qi])
            records=[(sample,float(scores[qi,i]),truth[(qid,sample)]) for i,sample in enumerate(ids)]
            result=metrics_for_ranking(records)
            independent=float(ndcg_score(gains[qid_index[qid]][None,:],scores[qi].astype(float)[None,:],k=10,ignore_ties=False))
            independent_errors.append(abs(independent-result['tie_expected_ndcg_at_10']))
            rows.append({'family':family,'seed':seed,'query_id':qid,'variant':variant,**result})
    metrics=pd.DataFrame(rows)
    assert len(metrics)==567 and max(independent_errors)<1e-12
    fields=['ndcg_at_k','tie_expected_ndcg_at_10','precision_at_k','upper_quartile_discovery_at_10']
    per_question=metrics.groupby(['family','seed','query_id'])[fields].mean()
    replicate_means=per_question.groupby(['family','seed'])[fields].mean()
    normal=float(replicate_means.loc[('unmodified',-1),'ndcg_at_k'])
    shuffled_summary=[]
    for family in protocol['control_families']:
        values=replicate_means.loc[family,'ndcg_at_k'].to_numpy()
        assert len(values)==31
        shuffled_summary.append({'family':family,'replicates':31,'mean':float(values.mean()),'median':float(np.median(values)),
            'sd':float(values.std(ddof=1)),'min':float(values.min()),'max':float(values.max()),
            'p025':float(np.quantile(values,.025)),'p975':float(np.quantile(values,.975)),
            'original_seed_value':float(replicate_means.loc[(family,protocol['seed_start']),'ndcg_at_k']),
            'replicates_at_least_unmodified':int(np.sum(values>=normal)),'unmodified_Linux_ndcg':normal,
            'scope':'Technical seed distribution, not independent-cohort uncertainty or a hypothesis test'})
    ranks=pd.read_csv(rank_path,dtype={'variant':str},float_precision='round_trip')
    old_means=pd.read_csv(metric_path,float_precision='round_trip').set_index('method')
    methods=sorted(ranks.method.unique());method_index={m:i for i,m in enumerate(methods)}
    assert len(methods)==27 and len(ranks)==27*9*96
    discounts=1/np.log2(np.arange(2,12))
    idcg=np.sum(np.sort(gains,axis=1)[:,-10:][:,::-1]*discounts,axis=1)
    coefficients=np.zeros((27,3,96),dtype=float)
    parent_scores={};score_rows=[]
    for key,group in ranks.groupby(['method','query_id','variant']):
        method,qid,variant=key
        assert len(group)==96 and group.sample_id.is_unique and set(group.sample_id)==set(ids)
        ordered=group.sort_values(['score','sample_id'],ascending=[False,True])
        top=[index[s] for s in ordered.head(10).sample_id]
        coefficients[method_index[method],qid_index[qid],top] += discounts/(9*idcg[qid_index[qid]])
        score=group.set_index('sample_id').loc[ids].score.to_numpy()
        parent_scores[key]=score
        score_rows.append({'method':method,'query_id':qid,'variant':variant,'score_sd':float(score.std()),
            'score_range':float(np.ptp(score)),'unique_scores':len(np.unique(score)),
            'score_grade_spearman':safe_spearman(score,grade[qid_index[qid]])})
    observed=np.einsum('mqs,qs->m',coefficients,gains)
    assert np.allclose(observed,old_means.loc[methods,'ndcg_at_k'],rtol=0,atol=1e-12)
    exact_mean=float(np.mean(gains.mean(axis=1)*discounts.sum()/idcg))
    assert np.allclose(np.einsum('mqs,q->m',coefficients,gains.mean(axis=1)),exact_mean,rtol=0,atol=1e-12)
    rng=np.random.default_rng(protocol['sample_label_randomization']['seed'])
    permutations=np.stack([rng.permutation(96) for _ in range(protocol['sample_label_randomization']['draws'])]).astype(np.int16)
    null=[]
    for start in range(0,len(permutations),250):
        permuted=gains[:,permutations[start:start+250]].transpose(1,0,2)
        null.append(permuted.reshape(len(permuted),-1)@coefficients.reshape(27,-1).T)
    null=np.concatenate(null)
    assert null.shape==(10000,27)
    null_rows=[]
    for mi,method in enumerate(methods):
        values=null[:,mi];standard_error=values.std(ddof=1)/np.sqrt(len(values))
        assert abs(values.mean()-exact_mean)<6*standard_error+1e-12
        null_rows.append({'method':method,'observed_ndcg':float(observed[mi]),'analytic_random_mean':exact_mean,
            'simulation_mean':float(values.mean()),'simulation_sd':float(values.std(ddof=1)),
            'reference_p025':float(np.quantile(values,.025)),'reference_p975':float(np.quantile(values,.975)),
            'fraction_random_assignments_at_least_observed':float(np.mean(values>=observed[mi])),
            'simulation_mean_error':float(values.mean()-exact_mean),
            'interpretation':'Conditional descriptive sample-label randomization; not general-superiority P values'})
    np.savez_compressed(out/'sample_label_randomization.npz',methods=np.asarray(methods),permutations=permutations,means=null,coefficients=coefficients)
    # Explicitly assess Linux/Mac parity without silently replacing old rankings.
    parity=[]
    for family,method in [('unmodified','rna_direct_sf_reference'),('all_mapped','rna_value_shuffled_sf_reference')]:
        matrix=new['unmodified'] if family=='unmodified' else new['all_mapped'][0]
        for qi in positions:
            qid,variant=str(new['query_id'][qi]),str(new['variant'][qi]);old=parent_scores[(method,qid,variant)]
            left=ids[np.lexsort((ids,-matrix[qi]))][:10];right=ids[np.lexsort((ids,-old))][:10]
            parity.append({'family':family,'query_id':qid,'variant':variant,
                'max_score_error':float(np.max(np.abs(matrix[qi]-old))),
                'top10_same_order':bool(np.array_equal(left,right)),'top10_jaccard':len(set(left)&set(right))/len(set(left)|set(right))})
    frozen=np.load(parent/'rna_embeddings.npz');old=np.load(old_embedding_path)
    old_idx={s:i for i,s in enumerate(old['library_id'].astype(str))}
    assert all(s in old_idx for s in ids)
    banks={'old_CPM_direct':old['rna_direct'][[old_idx[s] for s in ids]],
           'old_CPM_fused':old['rna_fused'][[old_idx[s] for s in ids]],
           **{name:frozen[name] for name in frozen.files if name.startswith('rna_')}}
    geometry_rows=[{'condition':name,**geometry(bank)} for name,bank in banks.items()]
    query_bank=np.load(r/'repair_v1/pilot_rankings_v1/text_embeddings.npz')
    query_rows=[];between_rows=[];query_summary=[]
    for space,key in [('raw_text','query_raw'),('projected_RNA_query','query_rna')]:
        query=unit_rows(query_bank[key]);centroids=[];question_ids=sorted(set(query_bank['query_id']))
        all_within=[]
        for qid in question_ids:
            indices=np.flatnonzero(query_bank['query_id']==qid);assert len(indices)==3
            pairs=[float(query[a]@query[b]) for a,b in itertools.combinations(indices,2)]
            all_within+=pairs
            query_rows.append({'space':space,'query_id':qid,'within_paraphrase_cosine_mean':float(np.mean(pairs)),
                               'within_paraphrase_cosine_min':float(np.min(pairs))})
            centroids.append(query[indices].mean(axis=0))
        centroids=unit_rows(centroids);between=[]
        for i,j in itertools.combinations(range(len(question_ids)),2):
            value=float(centroids[i]@centroids[j]);between.append(value)
            between_rows.append({'space':space,'query_a':str(question_ids[i]),'query_b':str(question_ids[j]),'centroid_cosine':value})
        query_summary.append({'space':space,'questions':30,'within_paraphrase_mean':float(np.mean(all_within)),
            'between_question_centroid_mean':float(np.mean(between)),
            'descriptive_within_minus_between':float(np.mean(all_within)-np.mean(between))})
    # Input summaries and score associations are diagnostic only, with no fit.
    values=data['sf_reference'];counts=data['counts'].astype(float);library=counts.sum(axis=1)
    features={'mapped_count_sum':library,'size_factor':data['sf_reference_factors'],
              'median_log_sf_reference':np.median(values,axis=1),
              'IQR_log_sf_reference':np.quantile(values,.75,axis=1)-np.quantile(values,.25,axis=1),
              'zero_fraction_mapped':(counts==0).mean(axis=1)}
    correlation_rows=[];grade_correlations=[];query_score_pairs=[]
    for name,feature in features.items():
        for qid in qids:grade_correlations.append({'input_summary':name,'query_id':qid,'spearman_to_grade':safe_spearman(feature,grade[qid_index[qid]])})
        for (method,qid,variant),score in parent_scores.items():
            correlation_rows.append({'method':method,'query_id':qid,'variant':variant,'input_summary':name,'spearman':safe_spearman(feature,score)})
    for method in methods:
        vectors={q:np.mean([parent_scores[(method,q,str(v))] for v in [1,2,3]],axis=0) for q in qids}
        for left,right in itertools.combinations(qids,2):
            query_score_pairs.append({'method':method,'query_a':left,'query_b':right,'mean_score_spearman':safe_spearman(vectors[left],vectors[right])})
    import torch
    import rna_portrait as rp
    torch.set_num_threads(2)
    model=rp.load_model(r/'rna-portrait/models',device='cpu')
    mapping={g:i for i,g in enumerate(data['model_gene'].astype(str))}
    present=np.array([g.upper() in mapping for g in model.selected_genes])
    columns=np.array([mapping[g.upper()] for g in model.selected_genes if g.upper() in mapping])
    assert present.sum()==3524
    preprocessing_rows=[]
    for condition,matrix in [('old_reference_CPM',np.log1p(counts/library[:,None]*1e6)),('sf_reference',values)]:
        raw=np.zeros((96,4096),dtype=np.float32);raw[:,present]=matrix[:,columns]
        standard=(raw-model.expr_mean)/model.expr_std
        for i,sample in enumerate(ids):
            preprocessing_rows.append({'condition':condition,'library_id':sample,
                'median_selected_log':float(np.median(raw[i,present])),
                'upper_clip_fraction':float((standard[i,present]>8).mean()),
                'lower_clip_fraction':float((standard[i,present]<-8).mean()),
                'zero_fraction_measured_selected':float((raw[i,present]==0).mean())})
    tables={
        'control_metrics_all_variants.csv':metrics,'control_per_question.csv':per_question.reset_index(),
        'control_replicate_means.csv':replicate_means.reset_index(),'control_distributions.csv':pd.DataFrame(shuffled_summary),
        'sample_label_randomization_summary.csv':pd.DataFrame(null_rows),'Linux_Mac_ranking_parity.csv':pd.DataFrame(parity),
        'embedding_geometry.csv':pd.DataFrame(geometry_rows),'query_paraphrase_geometry.csv':pd.DataFrame(query_rows),
        'query_pair_geometry.csv':pd.DataFrame(between_rows),'query_geometry_summary.csv':pd.DataFrame(query_summary),
        'parent_score_diagnostics.csv':pd.DataFrame(score_rows),'score_input_correlations.csv':pd.DataFrame(correlation_rows),
        'input_grade_correlations.csv':pd.DataFrame(grade_correlations),'cross_query_score_correlations.csv':pd.DataFrame(query_score_pairs),
        'preprocessing_sample_diagnostics.csv':pd.DataFrame(preprocessing_rows)}
    for name,frame in tables.items():
        assert not (out/name).exists()
        frame.to_csv(out/name,index=False)
    for name,expected in contract['source_sha256'].items():assert sha(r/name)==expected
    summary={'completed_utc':datetime.now(timezone.utc).isoformat(),'analysis_contract_sha256':sha(out/'analysis_contract.json'),
        'candidates':96,'judged_questions':qids,'control_families':2,'technical_replicates_per_family':31,
        'evaluated_control_metric_groups':len(metrics),'independent_ndcg_max_error':max(independent_errors),
        'normal_Linux_ndcg':normal,'control_distributions':shuffled_summary,
        'random_assignment_draws':10000,'analytic_random_ranking_mean':exact_mean,
        'Linux_Mac_top10_orders_all_equal':bool(pd.DataFrame(parity).top10_same_order.all()),
        'Linux_Mac_max_score_error':max(row['max_score_error'] for row in parity),
        'query_geometry':query_summary,'old_results_unchanged':True,'training_started':False,
        'biological_replicates_added':0,'general_superiority_P_values_reported':False,
        'remaining_gate':'Does not establish practical advantage, human statement correctness or cross-cohort confirmation.',
        'output_sha256':{name:sha(out/name) for name in list(tables)+['sample_label_randomization.npz']}}
    with (out/'analysis_summary.json').open('x') as stream:stream.write(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='output_sha256'},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args();run(a.study_root,a.output_root)

"""Outcome-free reconstruction of frozen CPTAC vectors and synthetic metric QA.

Does not interrupt the full inference job, change inputs, or open protein qrels.
Saved numerical reconstruction is distinct from biological validation.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import sys
import numpy as np
import pandas as pd
import torch


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write_new(path,payload):
    with path.open('x') as stream:stream.write(json.dumps(payload,indent=2,allow_nan=False)+'\n')


def metric_checks(study,drivers):
    sys.path.insert(0,str(study/'repair_v1'))
    from cptac_retrieval_metrics import measure
    spec=importlib.util.spec_from_file_location('independent_cptac_verifier',drivers/'verify_cptac_retrieval.py')
    verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
    rng=np.random.default_rng(2026101309)
    count=0;maximum=0.;null_checks=0
    for n in [9,10,37,100]:
        ids=np.array([f'SYNTHETIC_{i:03d}' for i in range(n)])
        cohorts=np.array([f'GROUP_{i%3}' for i in range(n)])
        for replicate in range(20):
            for missing in [0.,.25,.8,1.]:
                for tied in [False,True]:
                    scores=rng.normal(size=n)
                    if tied:scores=np.round(scores)
                    if replicate%5==0:scores[:3]=-2.
                    grades=rng.integers(0,3,size=n).astype(float)
                    if replicate%4==0:grades[:]=0.
                    grades[rng.random(n)<missing]=np.nan
                    actual=measure(ids,scores,grades,cohorts)
                    expected=verifier.independently_measure(ids,scores,grades,cohorts)
                    for key,value in expected.items():
                        observed=np.nan if actual[key] is None else float(actual[key])
                        if np.isnan(value):
                            assert np.isnan(observed),(n,replicate,key)
                            null_checks+=1
                        else:
                            difference=abs(observed-float(value))
                            assert difference<=1e-12,(n,replicate,key,difference)
                            maximum=max(maximum,difference)
                    # Neither helper may convert a wholly unjudged pool into zero accuracy.
                    if missing==1.:
                        assert actual['conditional_ndcg_at_10'] is None
                        assert actual['full_pool_precision_at_10'] is None
                    count+=1
    rejects=0
    cases=[(['a','a'],[1,2],[1,2],['x','x']),
           (['a','b'],[1,np.nan],[1,2],['x','x']),
           (['a','b'],[1,2],[1,3],['x','x'])]
    for case in cases:
        for function in [measure,verifier.independently_measure]:
            try:function(*case)
            except (ValueError,AssertionError):rejects+=1
            else:raise AssertionError('Invalid synthetic input accepted')
    return {'synthetic_fixtures':count,'undefined_metric_checks':null_checks,'invalid_input_rejections':rejects,
            'max_metric_abs_difference':maximum,'uses_real_protein_outcomes':False,
            'scope':'Agreement of independent formulas on synthetic boundary conditions, not biological validation or end-to-end evaluation.'}


def main(study,out,drivers):
    study,out,drivers=[p.resolve() for p in [study,out,drivers]]
    report_dir=out/'preflight_v1';report_dir.mkdir(exist_ok=False)
    repo=study/'rna-portrait'
    paths=[out/'PROTOCOL.json',out/'preparation.json',out/'prepared_inputs.npz',
           out/'inference_v1/execution_contract.json',out/'inference_v1/rna_embeddings.npz',
           study/'repair_v1/paired_candidates/CPTAC_PanCancer/rna_predictions_v1/mismatch_assignments.csv',
           study/'repair_v1/paired_candidates/CPTAC_PanCancer/rna_inputs_v1/mechanically_valid_candidates.csv',
           study/'repair_v1/pilot_rankings_v1/text_embeddings.npz',
           study/'repair_v1/cptac_retrieval_metrics.py',study/'repair_v1/tie_aware_retrieval.py',
           drivers/'verify_cptac_retrieval.py']
    hashes={str(p.relative_to(study)):sha(p) for p in paths}
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':hashes,
              'driver_sha256':sha(Path(__file__)),'vector_tolerance_abs':1e-6,'metric_tolerance_abs':1e-12,
              'scope':'Outcome-free reconstruction of saved compact RNA vectors; synthetic metric checks. No full-portrait rerun or actual retrieval performance evaluation.',
              'new_training':False,'protein_qrel_values_read':False}
    write_new(report_dir/'PROTOCOL.json',protocol)
    contract=json.loads((out/'inference_v1/execution_contract.json').read_text())
    for relative,expected in contract['source_sha256'].items():assert sha(repo/relative)==expected
    for relative,expected in contract['compact_model_sha256'].items():assert sha(repo/relative)==expected
    sys.path.insert(0,str(repo))
    import rna_portrait as rp
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    model=rp.load_model(repo/'models',device='cpu')
    inputs=np.load(out/'prepared_inputs.npz')
    saved=np.load(out/'inference_v1/rna_embeddings.npz')
    ids=inputs['sample_id'].astype(str)
    assert np.array_equal(ids,saved['sample_id'])
    genes=pd.Index(inputs['model_gene'].astype(str))
    selected=np.asarray(model.selected_genes)
    positions=genes.get_indexer(selected)
    present=positions>=0
    assert present.sum()==3502 and np.array_equal(present,inputs['measured_selected_mask'])
    assert np.array_equal(selected,inputs['selected_genes'])
    raw={}
    for arm in ['sf_reference','sf_full_measured']:
        raw[arm]=np.zeros((942,4096),dtype=np.float32)
        raw[arm][:,present]=inputs[arm][:,positions[present]].astype(np.float32)
    def encode(values):
        assert values.shape==(942,4096) and np.isfinite(values).all()
        assert np.all(values[:,~present]==0.)
        standardized=np.clip((values-model.expr_mean)/model.expr_std,-8,8).astype(np.float32)
        batches=[]
        with torch.inference_mode():
            for start in range(0,942,128):
                batches.append(model.alignment_model.encode_expr(torch.from_numpy(standardized[start:start+128])).numpy())
        return np.concatenate(batches)
    reconstructed={}
    for arm in raw:
        direct=encode(raw[arm]);reconstructed['rna_direct_'+arm]=direct
        with torch.inference_mode():
            portrait,_=model.portrait_model(torch.from_numpy(direct),torch.from_numpy(model.prototype_bank))
        mixed=(1.-model.fusion_alpha)*direct+model.fusion_alpha*portrait.numpy()
        reconstructed['rna_fused_'+arm]=mixed/np.maximum(np.linalg.norm(mixed,axis=1,keepdims=True),1e-8)
    rng=np.random.default_rng(20261010)
    selected_positions=np.flatnonzero(present)
    shuffled=raw['sf_reference'].copy()
    for i in range(942):shuffled[i,present]=rng.permutation(shuffled[i,present])
    assert np.array_equal(np.sort(shuffled[:,present],axis=1),np.sort(raw['sf_reference'][:,present],axis=1))
    reconstructed['rna_value_shuffled_sf_reference']=encode(shuffled)
    mask_hashes={}
    for fraction in [.1,.3,.5]:
        name=f'rna_mask_{int(fraction*100)}pct_sf_reference'
        changed=raw['sf_reference'].copy();masked=np.zeros_like(changed,dtype=bool)
        for i in range(942):
            dropped=rng.choice(selected_positions,size=int(round(3502*fraction)),replace=False)
            changed[i,dropped]=0.;masked[i,dropped]=True
        assert not masked[:,~present].any() and np.all(masked.sum(axis=1)==int(round(3502*fraction)))
        reconstructed[name]=encode(changed)
        mask_hashes[name]=hashlib.sha256(masked.tobytes()).hexdigest()
    def derange(size):
        order=rng.permutation(size);permutation=np.empty(size,dtype=int)
        permutation[order]=np.roll(order,1)
        assert np.all(permutation!=np.arange(size))
        return permutation
    global_perm=derange(942);within_perm=np.arange(942)
    pool=pd.read_csv(paths[6]).set_index('public_case_id').loc[ids]
    for cohort in sorted(pool.cohort.unique()):
        idx=np.flatnonzero(pool.cohort.to_numpy()==cohort)
        within_perm[idx]=idx[derange(len(idx))]
    old=pd.read_csv(paths[5]).set_index('assigned_case').loc[ids]
    assert np.array_equal(ids[global_perm],saved['mismatched_source_sample_id'])
    assert np.array_equal(ids[global_perm],old.global_expression_source_case.to_numpy(str))
    assert np.array_equal(ids[within_perm],saved['within_cohort_source_sample_id'])
    assert np.array_equal(ids[within_perm],old.within_cohort_expression_source_case.to_numpy(str))
    reconstructed['rna_mismatched_global_sf_reference']=reconstructed['rna_direct_sf_reference'][global_perm]
    reconstructed['rna_mismatched_within_cohort_sf_reference']=reconstructed['rna_direct_sf_reference'][within_perm]
    errors={}
    for method,value in reconstructed.items():
        difference=float(np.max(np.abs(value-saved[method])))
        assert difference<=1e-6,(method,difference)
        assert np.allclose(np.linalg.norm(saved[method],axis=1),1.,atol=1e-5)
        errors[method]=difference
    assert len(errors)==10
    query=np.load(paths[7])
    with torch.inference_mode():query_rna=model.alignment_model.encode_text(torch.from_numpy(query['query_raw'])).numpy()
    query_error=float(np.max(np.abs(query_rna-query['query_rna'])))
    assert query_error<=1e-6
    metrics=metric_checks(study,drivers)
    for relative,expected in hashes.items():assert sha(study/relative)==expected,relative
    result={'completed_utc':datetime.now(timezone.utc).isoformat(),'status':'passed',
            'protocol_sha256':sha(report_dir/'PROTOCOL.json'),'source_sha256':hashes,
            'reconstructed_vector_conditions':10,'reconstructed_vectors':9420,
            'max_vector_abs_errors':errors,'query_projection_max_abs_error':query_error,
            'shuffled_multiset_checks':942,'masked_missing_positions_unchanged':True,
            'mask_matrix_sha256':mask_hashes,'global_and_within_mismatch_match_historical_assignments':True,
            'synthetic_metrics':metrics,'existing_inputs_vectors_and_model_source_unchanged':True,
            'new_training':False,'protein_qrel_values_read':False,'model_performance_evaluated':False,
            'not_claimed':['full portrait inference complete','independent biological validation','retrieval advantage','Cell readiness']}
    write_new(report_dir/'RESULT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root',type=Path,required=True)
    parser.add_argument('--reassessment-root',type=Path,required=True)
    parser.add_argument('--drivers-root',type=Path,required=True)
    args=parser.parse_args();main(args.study_root,args.reassessment_root,args.drivers_root)

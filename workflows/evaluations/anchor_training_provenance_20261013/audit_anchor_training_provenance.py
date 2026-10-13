"""Read-only archival anchor-sidecar audit; never edits weights or live inference.

Preserves candidate provenance in the study workspace, then checks membership,
training standardization and archived model outputs. Does not read protein qrels.
"""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import argparse
import ast
import hashlib
import json
import re
import shutil
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch


RUN='train_merged_tcga_geo_deseq2_omicverse_hvg4096_sbert_allminilm'
CHECKPOINT_SHA='d88de4518075dc2d9ceb3b8eb5bc5831879cecbd69b97789c05f88fb8d40de48'


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,payload):
    with path.open('x') as stream:stream.write(json.dumps(payload,indent=2,allow_nan=False)+'\n')


def class_ast(path):
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BulkRNALanguageAligner')
    return ast.dump(node,include_attributes=False)


def main(study,source,out):
    study,source,out=[p.resolve() for p in [study,source,out]]
    run=source/'output/bulk_multimodal_embedding'/RUN
    matrix=source/'Data/01_merged_training_dataset'
    old_reference=study/'repair_v1/dataflow_audit_v1/runtime_reference_split_manifest.json'
    old_audit=study/'restored_runtime_20261009_v2/anchor_backbone_audit.json'
    training_code=source/'_code/bulk_multimodal_embedding/train_bulk_multimodal_embedding_sbert.py'
    portable=study/'rna-portrait/workflows/model_training/train_rna_language_alignment.py'
    checkpoint=run/'bulk_multimodal_embedding.pt'
    assert sha(checkpoint)==CHECKPOINT_SHA==json.loads(old_audit.read_text())['sha256']
    assert sha(training_code)=='dc66abd53d4d5dec0937534941e9c39c51d796ccf24ce299bb3d2e71ad8790d1'
    assert class_ast(training_code)==class_ast(portable)
    source_paths=[checkpoint,run/'sample_embeddings.csv',run/'summary.json',run/'training_history.csv',
                  training_code,matrix/'meta.parquet',matrix/'sample_ids.npy',matrix/'genes.npy',matrix/'expr_log.parquet']
    hashes={str(p.relative_to(source)):sha(p) for p in source_paths}
    cptac=study/'repair_v1/paired_candidates/CPTAC_PanCancer'
    local_paths=[old_reference,old_audit,portable,cptac/'original_development_case_crosswalk.csv',
                 cptac/'rna_inputs_v1/mechanically_valid_candidates.csv']
    local_hashes={str(p.relative_to(study)):sha(p) for p in local_paths}
    out.mkdir(exist_ok=False)
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for p in source_paths:
        if p.name=='expr_log.parquet':continue
        target=snapshot/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target);assert sha(target)==hashes[str(p.relative_to(source))]
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'driver_sha256':sha(Path(__file__)),
              'source_sha256':hashes,'existing_local_sources_sha256':local_hashes,
              'scope':'Recover and cross-check the actual anchor-run sidecar. Not a new split, model fit, full-system exposure guarantee or biological validation.',
              'checks':['Unique sidecar IDs and source-cohort grouping','All rows versus previously verified semantic/runtime split',
                        'Recompute float32 mean/std on archived train IDs in recorded order',
                        'Replay all archived age/tumor predictions using matching checkpoint and AST-identical frozen model class'],
              'statistical_tolerance_abs':1e-5,'prediction_tolerances_abs':{'pred_age_years':1e-3,'tumor_probability':1e-5},
              'tolerance_scope':'Numerical replay, not clinical prediction correctness. Record failures rather than widening tolerances.',
              'source_files_mutated':False,'protein_qrel_values_read':False,'new_training':False,
              'current_CPTAC_candidate_pool_or_live_job_changed':False}
    save(out/'PROTOCOL.json',protocol)
    columns=['sample_id','gse','split','feat_age','pred_age','tumor_probability']
    table=pd.read_csv(run/'sample_embeddings.csv',usecols=columns,keep_default_na=False)
    table['sample_id']=table.sample_id.astype(str)
    assert len(table)==25199 and table.sample_id.is_unique and not table.sample_id.eq('').any()
    assert set(table.split)=={'train','val','test'} and table.groupby('gse').split.nunique().max()==1
    summary=json.loads((run/'summary.json').read_text())
    split_counts=table.split.value_counts().to_dict()
    assert split_counts==summary['split_counts'] and len(table)==summary['n_samples']
    previous=json.loads(old_reference.read_text())['sample_to_split']
    split_series=table.sample_id.map(previous)
    comparison=table[['sample_id','gse','split']].copy()
    comparison['previous_runtime_split']=split_series
    comparison.to_csv(out/'split_comparison.private.csv',index=False)
    meta=pd.read_parquet(matrix/'meta.parquet')
    input_ids=np.load(matrix/'sample_ids.npy',allow_pickle=True).astype(str)
    assert len(meta)==len(input_ids)==len(table) and len(set(input_ids))==len(input_ids)
    assert set(input_ids)==set(table.sample_id)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    genes=np.asarray(payload['selected_genes'],dtype=str)
    assert len(genes)==4096 and len(set(genes))==4096
    expression_path=matrix/'expr_log.parquet'
    schema=pq.ParquetFile(expression_path).schema_arrow
    pandas_metadata=json.loads(schema.metadata[b'pandas'])
    index_field=pandas_metadata['index_columns'][0]
    assert isinstance(index_field,str)
    gene_axis=pd.read_parquet(expression_path,columns=[]).index.astype(str)
    canonical=gene_axis.str.replace(r'\.\d+$','',regex=True).str.upper()
    raw_genes=gene_axis[canonical.isin(genes)].tolist()
    print('Loading only the checkpoint gene rows for archival numerical replay',flush=True)
    expression=pd.read_parquet(expression_path,columns=table.sample_id.tolist(),filters=[(index_field,'in',raw_genes)])
    expression.index=expression.index.astype(str).str.replace(r'\.\d+$','',regex=True).str.upper()
    expression=expression.groupby(level=0).mean().loc[genes,table.sample_id.tolist()]
    assert expression.shape==(4096,25199) and np.isfinite(expression.to_numpy()).all()
    train_ids=table.loc[table.split.eq('train'),'sample_id'].tolist()
    training=expression.loc[:,train_ids].T.values.astype(np.float32)
    mean=training.mean(axis=0);std=training.std(axis=0);std=np.where(std<1e-3,1.,std)
    stored_mean=np.asarray(payload['expr_mean'],dtype=np.float32)
    stored_std=np.asarray(payload['expr_std'],dtype=np.float32)
    parameter_checks={}
    for label,new,stored in [('expr_mean',mean,stored_mean),('expr_std',std,stored_std)]:
        error=np.abs(new-stored)
        parameter_checks[label]={'values':len(new),'exact_values':int((new==stored).sum()),
                                 'max_abs_error':float(error.max()),'within_1e_minus_5':bool((error<=1e-5).all())}
    train_age=pd.to_numeric(table.loc[table.split.eq('train'),'feat_age'],errors='coerce').to_numpy(float)
    parameter_checks['age_mean_abs_error']=float(abs(np.nanmean(train_age)-payload['age_mean']))
    parameter_checks['age_std_abs_error']=float(abs(np.nanstd(train_age)-payload['age_std']))
    del training
    sys.path.insert(0,str(portable.parent))
    import train_rna_language_alignment as frozen
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert frozen.EMBED_DIM==payload['config']['embed_dim']
    model=frozen.BulkRNALanguageAligner(n_genes=len(genes),n_text_features=payload['config']['n_text_features'],n_sources=len(payload['source_map']))
    model.load_state_dict(payload['state_dict'],strict=True);model.eval()
    arrays=expression.T.to_numpy(dtype=np.float32)
    predicted_age=[];predicted_tumor=[]
    with torch.inference_mode():
        for start in range(0,len(table),2048):
            batch=np.clip((arrays[start:start+2048]-stored_mean)/stored_std,-8,8)
            z=model.encode_expr(torch.from_numpy(batch))
            predicted_age.append((model.age_head(z).reshape(-1)*payload['age_std']+payload['age_mean']).numpy())
            predicted_tumor.append(torch.sigmoid(model.tumor_head(z).reshape(-1)).numpy())
    predicted_age=np.concatenate(predicted_age);predicted_tumor=np.concatenate(predicted_tumor)
    prediction_checks={}
    replay=table[['sample_id','split']].copy()
    for label,new,limit in [('pred_age',predicted_age,1e-3),('tumor_probability',predicted_tumor,1e-5)]:
        old=pd.to_numeric(table[label],errors='raise').to_numpy(float)
        error=np.abs(new-old)
        prediction_checks[label]={'records':len(error),'max_abs_error':float(error.max()),'tolerance_abs':limit,
                                  'within_tolerance_records':int((error<=limit).sum()),'all_within_tolerance':bool((error<=limit).all())}
        replay[label+'_archived']=old;replay[label+'_recomputed']=new;replay[label+'_abs_error']=error
    replay.to_csv(out/'prediction_replay.private.csv',index=False)
    crosswalk=pd.read_csv(cptac/'original_development_case_crosswalk.csv',keep_default_na=False)
    assignments=table.set_index('sample_id').split
    crosswalk['archived_anchor_split']=crosswalk.sample_id.astype(str).map(assignments)
    known=crosswalk.dropna(subset=['archived_anchor_split']).groupby('case_submitter_id').archived_anchor_split.agg(lambda x:'|'.join(sorted(set(x))))
    pool=pd.read_csv(cptac/'rna_inputs_v1/mechanically_valid_candidates.csv',keep_default_na=False)
    pool['archived_anchor_case_splits']=pool.public_case_id.map(known).fillna('not_linked_by_existing_crosswalk')
    pool[['public_case_id','exposure_status','archived_anchor_case_splits']].to_csv(out/'CPTAC_anchor_linkage.private.csv',index=False)
    for relative,expected in hashes.items():assert sha(source/relative)==expected,relative
    for relative,expected in local_hashes.items():assert sha(study/relative)==expected,relative
    result={'completed_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(out/'PROTOCOL.json'),
            'source_checkpoint_sha256':CHECKPOINT_SHA,'sidecar_samples':len(table),'source_cohorts':table.gse.nunique(),
            'archived_split_counts':split_counts,'missing_from_previous_runtime_split':int(split_series.isna().sum()),
            'split_disagreements_vs_previous_runtime':int((split_series!=table.split).sum()),
            'checkpoint_has_split_assignments':'split_assignments' in payload,
            'checkpoint_config_has_seed':'seed' in payload.get('config',{}),'standardization_checks':parameter_checks,
            'prediction_replay_checks':prediction_checks,'AST_identical_model_class':True,
            'CPTAC_candidate_anchor_linkage_counts':pool.archived_anchor_case_splits.value_counts().to_dict(),
            'original_source_files_unchanged':True,'live_runtime_and_CPTAC_pool_unchanged':True,
            'new_training':False,'protein_qrel_values_read':False,
            'remaining_limits':['Sidecar recovery does not make the checkpoint self-contained.',
                                'Does not establish original training membership for all later heads, priors or rule development.',
                                'Case crosswalk nonmatches do not establish absence of patient/project or other development exposure.',
                                'Numerical model-output replay is not biological accuracy or a new independent cohort.']}
    save(out/'RESULT.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args();main(args.study_root,args.source_root,args.output_root)

"""Frozen direct-RNA controls for bounded post hoc diagnosis; no labels read."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import traceback


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_new(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def run(study_root, output_root):
    r, out = Path(study_root).resolve(), Path(output_root).resolve()
    lock = (out/'.inference.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (out/'inference_contract.json').exists(), 'Existing run must be inspected, not overwritten'
    parent = r/'normalization_reassessment_20261013/ssd_rerun_v1'
    input_root = parent.parent
    protocol = json.loads((out/'PROTOCOL.json').read_text())
    old_contract = json.loads((parent/'execution_contract.json').read_text())
    old_summary = json.loads((parent/'execution_summary.json').read_text())
    assert sha(input_root/'prepared_inputs.npz') == protocol['parent_input_sha256']
    assert sha(input_root/'PROTOCOL.json') == protocol['parent_protocol_sha256']
    assert sha(r/'queries.json') == protocol['query_sha256']
    assert sha(r/'repair_v1/paired_candidates/SDY2823/composition_pilot_v1/qrels.csv') == protocol['qrels_sha256']
    assert sha(parent/'execution_contract.json') == old_summary['execution_contract_sha256']
    for name, expected in old_summary['output_sha256'].items(): assert sha(parent/name) == expected
    repo = r/'rna-portrait'
    for group in ['source_sha256','compact_model_sha256']:
        for name, expected in old_contract[group].items(): assert sha(repo/name) == expected
    query_path = r/'repair_v1/pilot_rankings_v1/text_embeddings.npz'
    assert sha(query_path) == old_contract['query_bank_sha256']
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='')
    sys.path.insert(0, str(repo))
    import numpy as np
    import pandas as pd
    import torch
    import rna_portrait as rp
    torch.set_num_threads(2)
    assert torch.__version__ == '2.3.1+cpu' and torch.version.cuda is None
    model = rp.load_model(repo/'models', device='cpu')
    data = np.load(input_root/'prepared_inputs.npz')
    ids, genes, values = data['library_id'].astype(str), data['model_gene'].astype(str), data['sf_reference']
    assert values.shape == (96,23323) and len(set(ids)) == 96 and len(set(genes)) == 23323
    assert np.isfinite(values).all() and all(g == g.strip().upper() for g in genes)
    query_bank = np.load(query_path)
    mac_bank = np.load(parent/'rna_embeddings.npz')
    assert np.array_equal(ids, mac_bank['library_id'])
    lookup = {g:i for i,g in enumerate(genes)}
    measured = np.array([g.upper() in lookup for g in model.selected_genes])
    selected_rows = np.flatnonzero(measured)
    mapped_cols = np.array([lookup[g.upper()] for g in model.selected_genes if g.upper() in lookup])
    assert measured.shape == (4096,) and measured.sum() == 3524
    baseline_raw = np.zeros((96,4096), dtype=np.float32)
    baseline_raw[:,measured] = values[:,mapped_cols]
    families = list(protocol['control_families'])
    seeds = list(range(protocol['seed_start'], protocol['seed_start']+protocol['replicates_per_control_family']))
    contract = {'created_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(out/'PROTOCOL.json'),
        'driver_sha256':sha(__file__),'parent_contract_sha256':sha(parent/'execution_contract.json'),
        'parent_inference_summary_sha256':sha(parent/'execution_summary.json'),
        'input_sha256':sha(input_root/'prepared_inputs.npz'),'query_bank_sha256':sha(query_path),
        'parent_embedding_sha256':sha(parent/'rna_embeddings.npz'),
        'runtime_source_sha256':old_contract['source_sha256'],'model_sha256':old_contract['compact_model_sha256'],
        'library_ids':ids.tolist(),'families':families,'seeds':seeds,'measured_selected_genes':3524,'missing_selected_genes':572,
        'python':sys.version.split()[0], 'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','torch','scipy','scikit-learn']},
        'device':'cpu','threads':2,'labels_read':False,'training_started':False,
        'scope':'Direct encoder controls only; no full-portrait recomputation or new normalization'}
    save_new(out/'inference_contract.json',contract)
    started = time.monotonic()
    state = {'pid':os.getpid(),'status':'running','contract_sha256':sha(out/'inference_contract.json'),'controls_completed':0}
    def checkpoint(): (out/'inference_status.json').write_text(json.dumps(state,indent=2)+'\n')
    checkpoint()
    api_errors, diagnostic_rows = [], []

    def encode(raw, family, seed):
        standardized = (raw-model.expr_mean)/model.expr_std
        x = np.clip(standardized,-8,8).astype(np.float32)
        zs=[]
        with torch.inference_mode():
            # Single-row forwards preserve the public API's batch shape.
            for row in x:
                zs.append(model.alignment_model.encode_expr(torch.from_numpy(row[None,:])).numpy()[0])
        z=np.asarray(zs)
        assert np.isfinite(z).all() and np.allclose(np.linalg.norm(z,axis=1),1,atol=1e-5)
        assert np.all(raw[:,~measured]==0)
        for i,sample in enumerate(ids):
            diagnostic_rows.append({'family':family,'seed':seed,'library_id':sample,
                'upper_clip_fraction_measured':float((standardized[i,measured]>8).mean()),
                'lower_clip_fraction_measured':float((standardized[i,measured]<-8).mean()),
                'zero_fraction_measured':float((raw[i,measured]==0).mean())})
        return z

    try:
        normal = encode(baseline_raw,'unmodified',protocol['seed_start'])
        for i in protocol['platform_parity']['Linux_API_reference_rows']:
            check = model.encode_expression(dict(zip(genes,values[i])),input_scale='training_log')
            assert check['matched_genes']==3524
            api_errors.append(float(np.max(np.abs(normal[i]-check['direct_embedding']))))
        mac_normal_error=float(np.max(np.abs(normal-mac_bank['rna_direct_sf_reference'])))
        assert mac_normal_error <= protocol['platform_parity']['Linux_vs_frozen_Mac_embedding_absolute_tolerance']
        full_sorted = np.sort(values,axis=1)
        selected_sorted = np.sort(baseline_raw[:,measured],axis=1)
        banks={family:[] for family in families}
        multiset_checks=0
        for family in families:
            for seed in seeds:
                rng=np.random.default_rng(seed)
                raw=np.zeros_like(baseline_raw)
                api_inputs={}
                for i in range(len(ids)):
                    if family=='all_mapped':
                        permuted=rng.permutation(values[i])
                        assert np.array_equal(np.sort(permuted),full_sorted[i])
                        raw[i,measured]=permuted[mapped_cols]
                        if seed==seeds[0] and i in protocol['platform_parity']['Linux_API_reference_rows']:
                            api_inputs[i]=dict(zip(genes,permuted))
                    else:
                        assert family=='selected_measured'
                        permuted=rng.permutation(baseline_raw[i,measured])
                        assert np.array_equal(np.sort(permuted),selected_sorted[i])
                        raw[i,measured]=permuted
                        if seed==seeds[0] and i in protocol['platform_parity']['Linux_API_reference_rows']:
                            mapped=values[i].copy();mapped[mapped_cols]=permuted
                            api_inputs[i]=dict(zip(genes,mapped))
                    multiset_checks+=1
                z=encode(raw,family,seed)
                for i, supplied in api_inputs.items():
                    check=model.encode_expression(supplied,input_scale='training_log')
                    api_errors.append(float(np.max(np.abs(z[i]-check['direct_embedding']))))
                banks[family].append(z)
                state['controls_completed']+=1;state['latest_family']=family;state['latest_seed']=seed;checkpoint()
                if seed==seeds[0] or (seed-seeds[0]+1)%5==0 or seed==seeds[-1]:
                    print(f'{family}: {seed-seeds[0]+1}/{len(seeds)}; elapsed {time.monotonic()-started:.1f}s',flush=True)
        assert len(api_errors)==6 and max(api_errors)<=protocol['platform_parity']['optimized_vs_Linux_API_absolute_tolerance']
        banks={k:np.asarray(v) for k,v in banks.items()}
        old_control_error=float(np.max(np.abs(banks['all_mapped'][0]-mac_bank['rna_value_shuffled_sf_reference'])))
        assert old_control_error<=protocol['platform_parity']['Linux_vs_frozen_Mac_embedding_absolute_tolerance']
        scores={family:np.stack([query_bank['query_rna']@z.T for z in bank]) for family,bank in banks.items()}
        normal_scores=query_bank['query_rna']@normal.T
        assert normal_scores.shape==(90,96) and all(v.shape==(31,90,96) for v in scores.values())
        np.savez_compressed(out/'embeddings.npz',library_id=ids,seeds=np.asarray(seeds),unmodified=normal,**banks)
        np.savez_compressed(out/'scores.npz',library_id=ids,seeds=np.asarray(seeds),query_id=query_bank['query_id'],
                            variant=query_bank['variant'],unmodified=normal_scores,**scores)
        pd.DataFrame(diagnostic_rows).to_csv(out/'control_input_diagnostics.csv',index=False)
        for group in ['source_sha256','compact_model_sha256']:
            for name,expected in old_contract[group].items():assert sha(repo/name)==expected
        assert sha(input_root/'prepared_inputs.npz')==contract['input_sha256']
        summary={'completed_utc':datetime.now(timezone.utc).isoformat(),'contract_sha256':sha(out/'inference_contract.json'),
            'control_families':2,'replicates_each':31,'candidate_samples':96,'new_control_vectors':5952,
            'normal_baseline_vectors':96,'query_variants_scored':90,'multiset_checks':multiset_checks,
            'API_parity_checks':len(api_errors),'max_API_parity_error':max(api_errors),
            'Mac_normal_max_abs_embedding_error':mac_normal_error,'Mac_original_shuffle_max_abs_embedding_error':old_control_error,
            'labels_read':False,'training_started':False,'failed_inferences':0,'elapsed_seconds':time.monotonic()-started,
            'output_sha256':{n:sha(out/n) for n in ['embeddings.npz','scores.npz','control_input_diagnostics.csv']}}
        save_new(out/'inference_summary.json',summary)
        state.update(status='complete',failed_inferences=0,labels_read=False);checkpoint()
        print(json.dumps(summary,indent=2),flush=True)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),traceback=traceback.format_exc());checkpoint()
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args();run(a.study_root,a.output_root)

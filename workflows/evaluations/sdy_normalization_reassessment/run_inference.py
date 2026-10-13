"""SSD rerun of the frozen SDY protocol; no assay-label reading or training.

Requires the local rc4 full runtime and retained study inputs. This is not a
standalone full-model release. New records are contract-bound and resumable.
"""
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


def save_new(path, payload):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(payload, indent=2, allow_nan=False) + '\n')


def check_hashes(root, hashes):
    for relative, expected in hashes.items():
        if sha(root / relative) != expected:
            raise RuntimeError('Changed file: ' + relative)


def run(study_root, input_root, output_root):
    r, inp, out = [Path(x).resolve() for x in [study_root, input_root, output_root]]
    repo = r / 'rna-portrait'
    assets = r / 'restored_runtime_20261009_v2/model_artifacts'
    matrix = r / 'released_data_package/processed_data/training_matrix'
    cohort = r / 'repair_v1/paired_candidates/SDY2823'
    protocol = json.loads((inp / 'PROTOCOL.json').read_text())
    preparation = json.loads((inp / 'preparation.json').read_text())
    assert sha(inp / 'PROTOCOL.json') == preparation['protocol_sha256']
    assert sha(inp / 'prepared_inputs.npz') == preparation['prepared_inputs_sha256']
    check_hashes(r, preparation['source_sha256'])
    check_hashes(r, preparation['immutable_old_files_sha256'])
    print('Prepared inputs and historical outputs verified', flush=True)
    sys.path[:0] = [str(repo), str(repo / 'workflows/model_training')]
    from runtime_integrity import check_contract, check_environment, reject_ambient_overrides
    reject_ambient_overrides(os.environ)
    manifest_path = repo / 'data_availability/full_runtime_contract_v4.json'
    assert sha(manifest_path) == '7f0cef39f12e774a6a7b346a91490e7a3bdd84680a8ce402ba0814952b39e5eb'
    manifest = json.loads(manifest_path.read_text())
    integrity = check_contract(manifest, repo_root=repo, artifact_root=assets,
                               training_matrix=matrix, matrix_layout='sharded')
    packages = {name: importlib.metadata.version(name) for name in manifest['tested_packages']}
    check_environment(manifest, packages, sys.version_info)
    os.environ.update({
        'RNA_PORTRAIT_ARTIFACT_ROOT': str(assets), 'RNA_PORTRAIT_TRAINING_MATRIX': str(matrix),
        'RNA_PORTRAIT_MODEL_TRAINING_CODE': str(repo / 'workflows/model_training'),
        'RNA_PORTRAIT_ALLOW_PARTIAL_RUNTIME': '0', 'BMM_FORCE_CPU': '1',
        'BMM_CPU_THREADS': '2', 'BMM_CPU_INTEROP_THREADS': '2', 'BMM_NUM_WORKERS': '2',
        'BMM_DATASET_LAYOUT': 'sharded', 'TOKENIZERS_PARALLELISM': 'false',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
    })
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(protocol['seed'])
    np.random.seed(protocol['seed'])
    encoder = r / 'repair_v1/text_encoder_minilm'
    encoder_protocol_path = cohort / 'composition_pilot_v2/protocol.json'
    encoder_protocol = json.loads(encoder_protocol_path.read_text())
    check_hashes(encoder, encoder_protocol['encoder_sha256'])
    query_path = r / 'repair_v1/pilot_rankings_v1/text_embeddings.npz'
    query_bank = np.load(query_path)
    definitions = json.loads((r / 'queries.json').read_text())['queries']
    variants = [{'query_id': q['query_id'], **v} for q in definitions for v in q['queries']]
    assert len(variants) == 90
    assert [v['query_id'] for v in variants] == query_bank['query_id'].tolist()
    assert [v['variant'] for v in variants] == query_bank['variant'].tolist()
    data = np.load(inp / 'prepared_inputs.npz')
    ids, genes = data['library_id'].astype(str), data['model_gene'].astype(str)
    assert len(ids) == len(set(ids)) == 96 and len(genes) == len(set(genes)) == 23323
    assert all(g == g.upper().strip() for g in genes)
    arms, modes = list(protocol['arms']), ['released_calibration', 'no_calibration']
    code = sorted(p for folder in [repo / 'rna_portrait', repo / 'workflows/model_training']
                  for p in folder.glob('*.py') if not p.name.startswith('._'))
    compact_files = sorted(p for p in (repo / 'models').rglob('*') if p.is_file() and not p.name.startswith('._'))
    contract = {
        'run_id': 'SDY2823-cohort-size-factors-ssd-rerun-v1',
        'protocol_sha256': sha(inp / 'PROTOCOL.json'), 'preparation_sha256': sha(inp / 'preparation.json'),
        'inputs_sha256': sha(inp / 'prepared_inputs.npz'), 'driver_sha256': sha(__file__),
        'authorization_sha256': sha(out / 'RERUN_AUTHORIZATION.json'),
        'runtime_manifest_sha256': sha(manifest_path), 'runtime_dependency': 'Local full runtime rc4, not public compact API alone',
        'source_sha256': {str(p.relative_to(repo)): sha(p) for p in code},
        'compact_model_sha256': {str(p.relative_to(repo)): sha(p) for p in compact_files},
        'model_asset_sha256': {item['path']: item['sha256'] for item in manifest['model_assets']},
        'encoder_protocol_sha256': sha(encoder_protocol_path), 'encoder_sha256': encoder_protocol['encoder_sha256'],
        'query_bank_sha256': sha(query_path), 'queries_sha256': sha(r / 'queries.json'),
        'python': sys.version.split()[0], 'packages': packages, 'device': 'cpu', 'threads': 2,
        'seed': protocol['seed'], 'library_ids': ids.tolist(), 'arms': arms, 'calibration_modes': modes,
        'input_scale': 'training_log', 'top_k': 3, 'rerank_beta': .3, 'family_consistency_gamma': .1,
        'assay_labels_read': False, 'new_training': False,
        'scope': 'Exploratory whole-candidate-cohort normalization; not prospective single-sample preprocessing or statement-correctness validation',
    }
    contract_path = out / 'execution_contract.json'
    if contract_path.exists():
        assert json.loads(contract_path.read_text()) == contract, 'Contract changed; choose a new version'
    else:
        save_new(contract_path, contract)
    contract_hash = sha(contract_path)
    if (out / 'execution_summary.json').exists():
        complete = json.loads((out / 'execution_summary.json').read_text())
        assert complete['execution_contract_sha256'] == contract_hash
        check_hashes(out, complete['output_sha256'])
        print('Already complete and hash-verified; no computation repeated', flush=True)
        return
    started = time.monotonic()
    import rna_portrait as rp
    import infer_unknown_rna_semantic_explainer as explainer
    import infer_unknown_rna_sample as anchor
    assert all(Path(m.__file__).resolve().is_relative_to(repo) for m in [rp, explainer, anchor])
    compact = rp.load_model(repo / 'models', device='cpu')
    with torch.inference_mode():
        projected = compact.alignment_model.encode_text(torch.from_numpy(query_bank['query_raw'])).numpy()
    query_error = float(np.max(np.abs(projected - query_bank['query_rna'])))
    assert query_error <= 1e-6
    vectors, diagnostics = {}, []
    rng = np.random.default_rng(protocol['seed'])
    for arm in arms:
        values = data[arm]
        assert values.shape == (96, 23323) and np.isfinite(values).all() and (values >= 0).all()
        direct, fused = [], []
        for i, sample in enumerate(ids):
            encoded = compact.encode_expression(dict(zip(genes, values[i])), input_scale='training_log')
            assert encoded['matched_genes'] == 3524
            assert encoded['preprocessing']['normalization_library_sum'] is None
            direct.append(encoded['direct_embedding']); fused.append(encoded['fused_embedding'])
            diagnostics.append({'library_id': sample, 'arm': arm, **encoded['preprocessing']})
        vectors['rna_direct_' + arm], vectors['rna_fused_' + arm] = np.asarray(direct), np.asarray(fused)
        print('Compact inference complete:', arm, flush=True)
    shuffled = []
    for row in data['sf_reference']:
        shuffled.append(compact.encode_expression(dict(zip(genes, rng.permutation(row))), input_scale='training_log')['direct_embedding'])
    vectors['rna_value_shuffled_sf_reference'] = np.asarray(shuffled)
    permutation = rng.permutation(len(ids))
    while np.any(permutation == np.arange(len(ids))):
        permutation = rng.permutation(len(ids))
    vectors['rna_mismatched_sf_reference'] = vectors['rna_direct_sf_reference'][permutation]
    for z in vectors.values():
        assert z.shape == (96, 256) and np.isfinite(z).all()
        assert np.allclose(np.linalg.norm(z, axis=1), 1, atol=1e-5)
    embedding_path = out / 'rna_embeddings.npz'
    if embedding_path.exists():
        previous = np.load(embedding_path)
        assert np.array_equal(previous['library_id'], ids)
        assert np.array_equal(previous['mismatched_source_library_id'], ids[permutation])
        for name, z in vectors.items():
            assert np.array_equal(previous[name], z), 'Resumed vectors differ'
    else:
        np.savez_compressed(embedding_path, library_id=ids, mismatched_source_library_id=ids[permutation], **vectors)
    print('Loading full semantic and structured-anchor runtime', flush=True)
    runtime, anchor_artifacts = explainer.load_runtime(), anchor.load_artifacts()
    for meta in [runtime['train_meta'], anchor_artifacts.train_meta]:
        assert len(meta) == manifest['training_reference_count']
        assert hashlib.sha256('\n'.join(meta.sample_id.astype(str)).encode()).hexdigest() == manifest['training_reference_ids_sha256']
    original_compute, original_runner = explainer._compute_structured_anchor, explainer._load_structured_anchor_runner()
    current_values, observed = {}, {}

    def audited_runner(text, **kwargs):
        assert kwargs['value_mode'] == 'log1p' and kwargs['allow_alternate'] is False
        result = original_runner(text, **kwargs)
        assert result['openworld']['chosen_route'] == 'declared_preprocessing'
        observed['anchor_runner_declared_transform'] = True
        return result

    def audited_compute(values, *, pretransformed=False):
        assert pretransformed is True and values == current_values
        observed['same_values_both_branches'] = True
        return original_compute(values, pretransformed=pretransformed)

    explainer._load_structured_anchor_runner = lambda: audited_runner
    explainer._compute_structured_anchor = audited_compute
    text_rows, outputs, completed = [], [embedding_path], {}
    for arm in arms:
        for mode in modes:
            enabled = mode == 'released_calibration'
            os.environ['SEMANTIC_DISABLE_EVIDENCE_SCORER'] = '0' if enabled else '1'
            os.environ['SEMANTIC_DISABLE_EVIDENCE_PRIORS'] = '0' if enabled else '1'
            explainer._load_state_evidence_scorer.cache_clear(); explainer._load_state_evidence_priors.cache_clear()
            assert (explainer._load_state_evidence_scorer() is not None) == enabled
            assert bool(explainer._load_state_evidence_priors()['family_priors']) == enabled
            method = 'full_portrait_' + mode + '_' + arm
            folder = out / 'portraits' / method
            folder.mkdir(parents=True, exist_ok=True)
            for i, sample in enumerate(ids):
                target = folder / (sample + '.json')
                row_hash = hashlib.sha256(data[arm][i].tobytes()).hexdigest()
                if target.exists():
                    record = json.loads(target.read_text())
                    assert record['status'] == 'ok', 'Previous failure must be diagnosed, not silently retried'
                    assert record['execution_contract_sha256'] == contract_hash and record['input_row_sha256'] == row_hash
                else:
                    record = {'library_id': sample, 'condition': method, 'input_row_sha256': row_hash,
                              'execution_contract_sha256': contract_hash}
                    before = time.monotonic()
                    try:
                        current_values, observed = dict(zip(genes, data[arm][i])), {}
                        payload = explainer.explain_counts(current_values, top_k=3, rerank_beta=.3,
                                                           family_consistency_gamma=.1, input_scale='training_log')
                        assert payload['artifact_completeness']['structured_assets_complete']
                        assert payload['matched_selected_genes'] == 3524
                        assert payload['input_preprocessing']['output_scale'] == 'training_log'
                        assert payload['input_preprocessing']['normalization_library_sum'] is None
                        assert observed == {'same_values_both_branches': True, 'anchor_runner_declared_transform': True}
                        record.update(status='ok', payload=payload, input_route_audit=observed)
                    except Exception as exc:
                        record.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
                    record['elapsed_seconds'] = time.monotonic() - before
                    save_new(target, record)
                    if record['status'] != 'ok':
                        raise RuntimeError('Failed inference recorded at ' + str(target))
                assert record['library_id'] == sample and record['condition'] == method
                portrait = record['payload']['semantic_explanation']['semantic_portrait']
                assert all(isinstance(portrait.get(f), str) and portrait[f] for f in protocol['text_fields'])
                text = '\n'.join(f.replace('_', ' ') + ': ' + portrait[f] for f in protocol['text_fields'])
                text_rows.append({'method': method, 'library_id': sample, 'portrait_text': text})
                outputs.append(target)
                if i == 0 or (i+1) % 12 == 0:
                    print(f'{method}: {i+1}/96; elapsed {time.monotonic()-started:.1f}s', flush=True)
            completed[method] = 96
    from sentence_transformers import SentenceTransformer
    encoder_model = SentenceTransformer(str(encoder), device='cpu')
    unique = sorted(set(row['portrait_text'] for row in text_rows))
    z = encoder_model.encode(unique, batch_size=32, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    q = encoder_model.encode([v['text'] for v in variants], batch_size=32, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    encoder_error = float(np.max(np.abs(q - query_bank['query_raw'])))
    assert encoder_error <= 1e-6
    scores = {method: query_bank['query_rna'] @ v.T for method, v in vectors.items()}
    index, text_vectors = {text: i for i, text in enumerate(unique)}, {}
    texts = pd.DataFrame(text_rows)
    for method, group in texts.groupby('method'):
        group = group.set_index('library_id').loc[ids]
        text_vectors[method] = z[[index[t] for t in group.portrait_text]]
        scores[method] = query_bank['query_raw'] @ text_vectors[method].T
    assert len(scores) == 10
    ranking_rows = []
    for method, score in scores.items():
        assert score.shape == (90, 96) and np.isfinite(score).all()
        for qi, v in enumerate(variants):
            for rank, i in enumerate(np.lexsort((ids, -score[qi])), 1):
                ranking_rows.append({'method': method, 'query_id': v['query_id'], 'variant': v['variant'],
                                     'sample_id': ids[i], 'rank': rank, 'score': float(score[qi, i])})
    text_embedding_path = out / 'portrait_embeddings.npz'
    assert not text_embedding_path.exists(), 'Inspect partial finalization before retrying'
    np.savez_compressed(text_embedding_path, library_id=ids, **text_vectors)
    outputs.append(text_embedding_path)
    for name, frame in [('rankings_all_30_queries.csv.gz', pd.DataFrame(ranking_rows)),
                        ('portrait_texts.csv', texts), ('input_diagnostics.csv', pd.DataFrame(diagnostics))]:
        target = out / name
        assert not target.exists(), 'Inspect partial finalization before retrying'
        frame.to_csv(target, index=False, compression='gzip' if name.endswith('.gz') else None)
        outputs.append(target)
    check_hashes(r, preparation['immutable_old_files_sha256'])
    check_hashes(repo, contract['source_sha256']); check_hashes(repo, contract['compact_model_sha256'])
    check_hashes(assets, contract['model_asset_sha256'])
    summary = {
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'execution_contract_sha256': contract_hash,
        'integrity': integrity,
        'integrity_scope_note': 'rc4 contains a historical unresolved-normalization label; see the separate 20261013 recovery audit for updated provenance.',
        'candidate_donors': 96, 'successful_full_inferences': len(text_rows), 'failed_inferences': 0,
        'full_conditions': completed, 'added_methods_and_controls': list(scores),
        'ranked_rows': len(ranking_rows), 'query_variants': 90,
        'unique_portrait_texts': texts.groupby('method').portrait_text.nunique().to_dict(),
        'query_projection_max_abs_error': query_error, 'text_query_encoding_max_abs_error': encoder_error,
        'execution_seconds_after_integrity_checks': time.monotonic()-started,
        'assay_label_values_read': False, 'old_outputs_unchanged': True, 'source_and_weights_unchanged': True,
        'training_started': False, 'retrieval_accuracy_evaluated': False,
        'output_sha256': {str(p.relative_to(out)): sha(p) for p in outputs},
    }
    save_new(out / 'execution_summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ['output_sha256', 'integrity']}, indent=2), flush=True)


class Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, text):
        self.original.write(text); self.log.write(text); self.log.flush()

    def flush(self):
        self.original.flush(); self.log.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    with (args.output_root / '.inference.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (args.output_root / 'inference.log').open('a', buffering=1) as log:
            original_out, original_error = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = Tee(original_out, log), Tee(original_error, log)
            try:
                print(json.dumps({'attempt_started_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
                                  'executable': sys.executable, 'argv': sys.argv}), flush=True)
                run(args.study_root, args.input_root, args.output_root)
            except BaseException:
                traceback.print_exc()
                raise
            finally:
                sys.stdout, sys.stderr = original_out, original_error

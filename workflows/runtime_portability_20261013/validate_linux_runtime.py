"""Check the isolated Linux runtime without training or replacing old results."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
import subprocess
import time

ACTIVE_STATUS_PATH = None


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def compare(actual, expected, path='', differences=None, numbers=None):
    differences = [] if differences is None else differences
    numbers = [] if numbers is None else numbers
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected or type(actual) is not type(expected): differences.append({'path': path, 'kind': 'exact_value', 'expected': expected, 'actual': actual})
    elif isinstance(expected, int):
        if type(actual) is not int or actual != expected: differences.append({'path': path, 'kind': 'integer', 'expected': expected, 'actual': actual})
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isfinite(actual):
            differences.append({'path': path, 'kind': 'nonfinite_or_type'})
        else:
            error = abs(actual-expected)
            numbers.append({'path': path, 'expected': expected, 'actual': actual, 'absolute_error': error})
            if not math.isclose(actual, expected, abs_tol=1e-5, rel_tol=1e-5):
                differences.append({'path': path, 'kind': 'float_tolerance', 'expected': expected, 'actual': actual, 'absolute_error': error})
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            differences.append({'path': path, 'kind': 'dictionary_keys'})
        else:
            for key in expected: compare(actual[key], expected[key], path+'.'+key, differences, numbers)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            differences.append({'path': path, 'kind': 'list_length'})
        else:
            for i, (x, y) in enumerate(zip(actual, expected)): compare(x, y, path+f'[{i}]', differences, numbers)
    else:
        raise TypeError(type(expected))
    return differences, numbers


def main():
    global ACTIVE_STATUS_PATH
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--installation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    workspace, installation, out = [x.resolve() for x in [a.workspace, a.installation, a.output]]
    assert out.is_relative_to(workspace) and not out.exists()
    state = json.loads((installation / 'setup_status.json').read_text())
    assert state['status'] == 'complete' and state['CPU_only_verified']
    py = Path(state['environment']) / 'bin/python'
    study = workspace / 'research/cell_revision_20261008'
    repo = study / 'rna-portrait'
    candidate = study / 'runtime_release_candidate_20261010_rc4'
    source_manifest = repo / 'data_availability/full_runtime_contract_v4.json'
    assert sha(source_manifest) == '7f0cef39f12e774a6a7b346a91490e7a3bdd84680a8ce402ba0814952b39e5eb'
    manifest = json.loads(source_manifest.read_text())
    linux_contract = json.loads(source_manifest.read_text())
    linux_contract['tested_packages']['torch'] = '2.3.1+cpu'
    linux_contract['candidate_platform_note'] = 'Linux x86_64 CPU validation candidate; only expected torch build label changed. Model/code/data hashes and preprocessing are unchanged. Validation status is recorded separately.'
    expected_files = sorted((candidate / 'expected').glob('external_180__*.json'))
    assert len(expected_files) == 3
    examples, targets = [], {}
    for path in expected_files:
        item = json.loads(path.read_text())
        source = candidate / 'examples' / item['input_file']
        assert sha(source) == item['input_sha256']
        examples.append(source); targets[item['input_file']] = item
    tests = sorted(p for p in (repo / 'tests').glob('*.py') if not p.name.startswith('._'))
    out.mkdir()
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'driver_sha256': sha(__file__),
        'source_manifest_sha256': sha(source_manifest), 'installation_state_sha256': sha(installation / 'setup_status.json'),
        'linux_wheel_lock_sha256': sha(installation / 'requirements-linux-cpu-py312.lock'),
        'source_sha256': {str(path.relative_to(study)): sha(path) for path in tests+expected_files+examples},
        'model_examples': 3, 'source_Mac_tests_expected': 69,
        'numeric_comparison': {'absolute_tolerance': 1e-5, 'relative_tolerance': 1e-5},
        'exact_comparison': 'All dictionary/list structure, strings, booleans, nulls and integers including sample/rank identities must agree.',
        'selection': 'The same three rc4 historical engineering examples, not independently selected biological validation samples.',
        'no_retraining_or_SDY_recalculation': True,
        'boundary': 'Cross-platform smoke compatibility only; not proof of all future outputs, statement correctness, normalization transfer or Cell utility.'}
    save(out / 'PROTOCOL.json', protocol)
    save(out / 'linux_runtime_contract.candidate.json', linux_contract)
    env = os.environ.copy()
    for key in list(env):
        if key in {'PYTHONPATH','PYTHONHOME'} or key.startswith(('RNA_PORTRAIT_','SRP_','BMM_','SPTA_','SEMANTIC_')):
            env.pop(key)
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
        HF_HOME=str(out/'empty_hf_cache'), TORCH_HOME=str(out/'empty_torch_cache'), CUDA_VISIBLE_DEVICES='')
    env['PYTHONPATH'] = str(repo)
    jobs = [
        ('tests', [str(py), '-m', 'pytest', '-q', '-p', 'no:cacheprovider', str(repo/'tests'), '-c', str(repo/'pyproject.toml')]),
        ('full-smoke', [str(py), str(repo/'workflows/run_full_portrait.py'), *map(str, examples),
            '--artifact-root', str(study/'restored_runtime_20261009_v2/model_artifacts'),
            '--training-matrix', str(study/'released_data_package/processed_data/training_matrix'),
            '--manifest', str(out/'linux_runtime_contract.candidate.json'), '--matrix-layout', 'sharded',
            '--input-scale', 'legacy_untransformed', '--taxonomy', 'corrected', '--threads', '2',
            '--output', str(out/'full_inference.json')]),
    ]
    record = {'status': 'running', 'pid': os.getpid(), 'protocol_sha256': sha(out/'PROTOCOL.json'), 'steps': []}
    ACTIVE_STATUS_PATH = out / 'status.json'
    save(out/'status.json', record)
    for name, command in jobs:
        started = time.monotonic()
        print('Validation:', name, flush=True)
        with (out/(name+'.stdout.log')).open('x') as stdout, (out/(name+'.stderr.log')).open('x') as stderr:
            child = subprocess.Popen(command, cwd=repo, env=env, stdout=stdout, stderr=stderr)
            record['active_child'] = {'pid': child.pid, 'name': name}; save(out/'status.json', record)
            exit_code = child.wait()
        record['steps'].append({'name': name, 'command': command, 'returncode': exit_code, 'seconds': time.monotonic()-started})
        record.pop('active_child', None); save(out/'status.json', record)
        if exit_code:
            record.update(status='failed', reason='Inspect '+name+' logs; no tolerance or code changed')
            save(out/'status.json', record)
            raise RuntimeError(record['reason'])
        if name == 'tests':
            assert '69 passed' in (out/'tests.stdout.log').read_text(), 'Unexpected test coverage; inspect, do not relabel'
    result = json.loads((out/'full_inference.json').read_text())
    assert len(result['records']) == 3
    differences, numbers, per_example = [], [], []
    for item in result['records']:
        expected = targets[item['input_file']]
        assert item['input_sha256'] == expected['input_sha256']
        mismatches, values = compare(item['payload'], expected['payload'], item['input_file'])
        differences += mismatches; numbers += values
        per_example.append({'input_file': item['input_file'], 'difference_count': len(mismatches),
                            'numeric_fields': len(values), 'max_absolute_error': max(v['absolute_error'] for v in values)})
    for relative, expected in protocol['source_sha256'].items(): assert sha(study/relative) == expected
    assert sha(source_manifest) == protocol['source_manifest_sha256']
    compare_result = {'examples': per_example, 'numeric_fields': len(numbers),
        'max_numeric_absolute_error': max(v['absolute_error'] for v in numbers),
        'differences': differences, 'numeric_details': numbers, 'tolerance_changed_after_observation': False,
        'accepted_with_predeclared_tolerance': not differences, 'bitwise_equal': all(v['absolute_error'] == 0 for v in numbers) and not differences,
        'scope': protocol['boundary']}
    save(out/'comparison.json', compare_result)
    record.update(status='complete' if not differences else 'failed', tests_passed=69, examples=3,
                  accepted_with_predeclared_tolerance=not differences, difference_count=len(differences), training_started=False,
                  completed_utc=datetime.now(timezone.utc).isoformat())
    save(out/'status.json', record)
    print(json.dumps({k: record[k] for k in ['status','tests_passed','examples','accepted_with_predeclared_tolerance','difference_count','training_started']}, indent=2), flush=True)
    assert not differences, 'Compatibility check failed; preserve all observed differences'


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if ACTIVE_STATUS_PATH is not None and ACTIVE_STATUS_PATH.exists():
            failed = json.loads(ACTIVE_STATUS_PATH.read_text())
            failed.update(status='failed' if 'active_child' not in failed else 'parent_error_check_live_child', error=repr(exc))
            save(ACTIVE_STATUS_PATH, failed)
        raise

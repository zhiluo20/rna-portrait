"""Validate transported wheels on Linux and install a new platform hash lock."""
from pathlib import Path
from datetime import datetime, timezone
from email.parser import BytesParser
import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import zipfile


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def norm(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--records', type=Path, required=True)
    a = p.parse_args()
    workspace, out = a.workspace.resolve(), a.records.resolve()
    study = workspace / 'research/cell_revision_20261008'
    prior = study / 'lab_runtime_20261013_v1'
    assert out.is_relative_to(workspace) and platform.system() == 'Linux' and platform.machine() == 'x86_64'
    lock_handle = (out / '.setup.lock').open('a')
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old_plan = json.loads((prior / 'install_plan.json').read_text())
    old_state = json.loads((prior / 'setup_status.json').read_text())
    assert old_state['status'] == 'failed'
    assert any(row['name'] == 'create-venv' and row['returncode'] == 0 for row in old_state['stages'])
    assert not any(row['name'] == 'install-offline' for row in old_state['stages'])
    environment = Path(old_plan['environment'])
    assert environment.parent == Path.home() / '.venvs'
    py = environment / 'bin/python'
    incoming = out / 'incoming_wheels'
    transport = json.loads((out / 'transport_status.json').read_text())
    assert transport['status'] == 'downloaded_for_Linux_validation' and transport['installed_on_Mac'] is False
    expected_files = {row['filename'] for row in transport['files']}
    assert {x.name for x in incoming.glob('*.whl')} == expected_files
    for row in transport['files']:
        assert sha(incoming / row['filename']) == row['sha256']
        assert (incoming / row['filename']).stat().st_size == row['bytes']
    requested_path = prior / 'requested-linux-versions.txt'
    assert sha(requested_path) == transport['requested_versions_sha256']
    plan = {'authorization': old_plan['authorization'], 'environment': str(environment),
        'workspace': str(workspace), 'prior_failed_plan_sha256': sha(prior / 'install_plan.json'),
        'prior_failed_state_sha256': sha(prior / 'setup_status.json'),
        'transport_manifest_sha256': sha(out / 'transport_status.json'),
        'driver_sha256': sha(__file__), 'requested_versions': old_plan['requested_versions'],
        'method': 'No lab DNS/network changes. Official-source Linux wheel candidates transferred over configured SSH; resolve actual dependency closure again on Linux offline, then install require-hashes lock.',
        'CPU_only': True, 'reuse_only_task_created_partial_venv': True,
        'scientific_inference_or_training': False, 'global_environments_modified': False}
    plan_path = out / 'install_plan.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan
    else:
        save(plan_path, plan)
    state_path = out / 'setup_status.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'plan_sha256': sha(plan_path), 'stages': [], 'attempts': []}
    assert state['plan_sha256'] == sha(plan_path)
    state['attempts'].append({'started_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid()})
    state.update(pid=os.getpid(), status='running')
    save(state_path, state)
    env = os.environ.copy()
    env.pop('PYTHONPATH', None); env.pop('PYTHONHOME', None)
    env.update(PYTHONNOUSERSITE='1', PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK='1',
               PIP_NO_INDEX='1', PIP_CACHE_DIR=str(out / 'pip-cache'), TOKENIZERS_PARALLELISM='false')

    def step(name, command):
        previous = [row for row in state['stages'] if row['name'] == name and row['returncode'] == 0]
        if previous:
            assert previous[-1]['command'] == [str(x) for x in command]
            return
        log_path = out / f'{len(state["stages"])+1:02d}-{name}.log'
        started = time.monotonic()
        print('Starting:', name, flush=True)
        with log_path.open('x') as log:
            child = subprocess.Popen(command, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT)
            state['active_child'] = {'pid': child.pid, 'stage': name}; save(state_path, state)
            code = child.wait()
        state['stages'].append({'name': name, 'command': [str(x) for x in command], 'returncode': code,
            'seconds': time.monotonic()-started, 'log': log_path.name, 'log_sha256': sha(log_path)})
        state.pop('active_child', None); save(state_path, state)
        print(f'{name}: exit {code}', flush=True)
        if code: raise RuntimeError('Failed stage; preserve logs and partial environment: ' + name)

    try:
        inventory_before = json.loads(subprocess.run([str(py), '-m', 'pip', 'list', '--format=json'], env=env, check=True, capture_output=True, text=True).stdout)
        if not state['stages']:
            assert {norm(row['name']) for row in inventory_before} <= {'pip', 'setuptools', 'wheel'}
            save(out / 'before_install_packages.json', inventory_before)
        resolved = out / 'resolved_linux_wheels'
        resolved.mkdir(exist_ok=True)
        step('resolve-on-linux-offline', [str(py), '-m', 'pip', 'download', '--no-index', '--only-binary=:all:',
            '--find-links', str(incoming), '--dest', str(resolved), '-r', str(requested_path)])
        records, versions = [], {}
        for path in sorted(resolved.glob('*.whl')):
            with zipfile.ZipFile(path) as archive:
                names = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
                assert len(names) == 1
                metadata = BytesParser().parsebytes(archive.read(names[0]))
            name, version = norm(metadata['Name']), metadata['Version']
            assert name not in versions and name != 'appnope'
            assert not name.startswith(('nvidia-', 'rocm-', 'cuda-')) and name not in {'triton', 'pytorch-triton'}
            versions[name] = version
            records.append({'name': name, 'version': version, 'filename': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path)})
        for name, wanted in plan['requested_versions'].items(): assert versions[name] == wanted
        content = ''.join(f'{r["name"]}=={r["version"]} --hash=sha256:{r["sha256"]}\n' for r in sorted(records, key=lambda x: x['name']))
        lock = out / 'requirements-linux-cpu-py312.lock'
        if lock.exists(): assert lock.read_text() == content
        else:
            lock.write_text(content)
            save(out / 'wheel_manifest.json', {'wheels': records, 'total_bytes': sum(r['bytes'] for r in records),
                'unused_transport_candidates': sorted(expected_files - {r['filename'] for r in records}),
                'dependency_resolution_host': 'Linux x86_64 Python 3.12.3', 'network_used_by_Linux_resolver': False})
        step('install-locked-offline', [str(py), '-m', 'pip', 'install', '--no-index', '--find-links', str(resolved),
             '--require-hashes', '-r', str(lock), '--report', str(out / 'pip-install-report.json')])
        step('pip-check', [str(py), '-m', 'pip', 'check'])
        packages = json.loads(subprocess.run([str(py), '-m', 'pip', 'list', '--format=json'], env=env, check=True, capture_output=True, text=True).stdout)
        installed = {norm(row['name']): row['version'] for row in packages}
        for name, wanted in versions.items(): assert installed[name] == wanted
        save(out / 'installed_packages.json', installed)
        probe = '''import sys,json,platform
import numpy,pandas,pyarrow,torch,scipy,sklearn,sentence_transformers,matplotlib,PIL
import nbformat,nbclient,ipykernel
torch.set_num_threads(2)
assert torch.__version__ == "2.3.1+cpu"
assert torch.version.cuda is None and torch.version.hip is None and not torch.cuda.is_available()
assert float(torch.tensor([1.,2.,3.]).sum()) == 6.
print(json.dumps({"python":sys.version.split()[0],"executable":sys.executable,"platform":platform.platform(),"torch":torch.__version__,"cuda_build":torch.version.cuda,"hip_build":torch.version.hip,"imports_passed":True,"model_forward_tested":False,"training_started":False}))
'''
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
        result = subprocess.run([str(py), '-c', probe], env=env, check=True, capture_output=True, text=True)
        (out / 'import_probe.stderr.log').write_text(result.stderr)
        save(out / 'import_probe.json', json.loads(result.stdout))
        state.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
            environment=str(environment), distributions=len(installed), linux_lock_sha256=sha(lock),
            CPU_only_verified=True, scientific_inference_started=False, training_started=False)
        save(state_path, state)
        print(json.dumps({k: state[k] for k in ['status', 'environment', 'distributions', 'CPU_only_verified', 'scientific_inference_started', 'training_started']}, indent=2), flush=True)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        state.pop('active_child', None); save(state_path, state)
        raise


if __name__ == '__main__':
    main()

"""Create an authorized, isolated Linux CPU runtime without touching old envs.

Plan first, then download binary wheels, record a platform-specific hash lock,
install offline from that lock, and validate imports/CPU status. No model
training or scientific inference is performed by this program.
"""
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
import shutil
import subprocess
import sys
import time
import traceback
import zipfile


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def normalized(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    workspace, environment, out = [x.resolve() for x in [args.workspace, args.environment, args.records]]
    assert platform.system() == 'Linux' and platform.machine() == 'x86_64'
    assert sys.version_info[:2] == (3, 12), 'Use the confirmed system Python 3.12 to create the new env'
    assert environment.parent == Path.home() / '.venvs' and environment.name.startswith('rna-portrait-cell-cpu-')
    assert out.is_relative_to(workspace) and workspace.is_dir()
    assert shutil.disk_usage(environment.parent if environment.parent.exists() else Path.home()).free > 8 * 1024**3
    out.mkdir(exist_ok=True)
    lock_handle = (out / '.setup.lock').open('a')
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    study = workspace / 'research/cell_revision_20261008'
    mac_lock = study / 'runtime_release_candidate_20261010_rc4/code/requirements-runtime-macos-arm64-py312.lock'
    manifest_path = study / 'rna-portrait/data_availability/full_runtime_contract_v4.json'
    assert sha(manifest_path) == '7f0cef39f12e774a6a7b346a91490e7a3bdd84680a8ce402ba0814952b39e5eb'
    manifest = json.loads(manifest_path.read_text())
    wanted = {}
    for line in mac_lock.read_text().splitlines():
        if not line or line.startswith('#'): continue
        requirement = line.split()[0]
        name, version = requirement.split('==')
        wanted[normalized(name)] = version
    assert len(wanted) == 47 and wanted['torch'] == '2.3.1'
    wanted['torch'] = '2.3.1+cpu'
    notebook = {'nbformat': '5.10.4', 'nbclient': '0.10.0', 'ipykernel': '6.29.5'}
    wanted.update(notebook)
    plan = {
        'authorization': 'Author explicitly approved an independent Linux Python 3.12 CPU environment; no training; preserve existing environments and SSD data.',
        'platform': {'system': platform.system(), 'machine': platform.machine(), 'python': sys.version.split()[0]},
        'environment': str(environment), 'workspace': str(workspace),
        'interpreter': str(Path(sys.executable).resolve()), 'interpreter_sha256': sha(Path(sys.executable).resolve()),
        'driver_sha256': sha(__file__), 'historical_macos_lock_sha256': sha(mac_lock),
        'runtime_manifest_sha256': sha(manifest_path), 'requested_versions': wanted,
        'notebook_direct_versions_from_existing_Mac_audit_env': notebook,
        'sources': {'torch': 'https://download.pytorch.org/whl/cpu', 'other_wheels': 'https://pypi.org/simple'},
        'reference_documentation': ['https://pytorch.org/get-started/previous-versions/', 'https://docs.python.org/3.12/library/venv.html'],
        'platform_differences': ['Linux x86_64, not macOS ARM64', 'Python 3.12.3, not the prior 3.12.14', 'torch 2.3.1+cpu CPU build, not macOS torch 2.3.1'],
        'install_method': 'System Python venv --copies on the lab internal filesystem; binary-wheel download only; new Linux SHA-256 lock; offline require-hashes installation.',
        'no_system_or_preexisting_env_modification': True, 'no_training': True,
        'equivalence_not_assumed': True,
    }
    plan_path = out / 'install_plan.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan, 'Plan or source changed; create a new installation record version'
    else:
        assert not environment.exists(), 'Do not adopt or overwrite an existing environment'
        save(plan_path, plan)
        (out / 'requested-linux-versions.txt').write_text(''.join(f'{n}=={v}\n' for n, v in sorted(wanted.items())))
    if not args.install:
        print(json.dumps({'plan': str(plan_path), 'environment': str(environment), 'requested_packages': len(wanted), 'installed': False}, indent=2))
        return
    env = os.environ.copy()
    env.pop('PYTHONPATH', None); env.pop('PYTHONHOME', None)
    env.update(PYTHONNOUSERSITE='1', PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK='1',
               PIP_CACHE_DIR=str(out / 'pip-cache'), TOKENIZERS_PARALLELISM='false')
    state_path = out / 'setup_status.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        assert state['plan_sha256'] == sha(plan_path)
    else:
        assert not environment.exists()
        state = {'plan_sha256': sha(plan_path), 'environment': str(environment), 'stages': [], 'attempts': []}
    state['attempts'].append({'started_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid()})
    state['pid'], state['status'] = os.getpid(), 'running'
    save(state_path, state)
    py = environment / 'bin/python'
    wheelhouse = out / 'wheelhouse'
    wheelhouse.mkdir(exist_ok=True)

    def step(name, command):
        existing = [s for s in state['stages'] if s['name'] == name and s['returncode'] == 0]
        if existing:
            assert existing[-1]['command'] == [str(x) for x in command]
            print('Already completed:', name, flush=True)
            return
        log = out / f'{len(state["stages"])+1:02d}-{name}.log'
        before = time.monotonic()
        record = {'name': name, 'command': [str(x) for x in command], 'started_utc': datetime.now(timezone.utc).isoformat(), 'log': log.name}
        print('Starting:', name, flush=True)
        with log.open('x') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=out, env=env)
            state['active_child'] = {'pid': child.pid, 'stage': name}
            save(state_path, state)
            returncode = child.wait()
        record.update(returncode=returncode, elapsed_seconds=time.monotonic()-before, log_sha256=sha(log))
        state['stages'].append(record); state.pop('active_child', None)
        save(state_path, state)
        print(f'{name}: exit {returncode}, {record["elapsed_seconds"]:.1f}s', flush=True)
        if returncode:
            raise RuntimeError('Stage failed; preserve log and partial environment: ' + str(log))

    try:
        step('create-venv', [sys.executable, '-m', 'venv', '--copies', str(environment)])
        assert py.is_file() and (environment / 'pyvenv.cfg').is_file()
        step('download-cpu-torch', [str(py), '-m', 'pip', 'download', '--only-binary=:all:', '--no-deps',
             '--index-url', plan['sources']['torch'], '--dest', str(wheelhouse), 'torch==2.3.1+cpu'])
        step('resolve-linux-wheels', [str(py), '-m', 'pip', 'download', '--only-binary=:all:',
             '--index-url', plan['sources']['other_wheels'], '--find-links', str(wheelhouse),
             '--dest', str(wheelhouse), '-r', str(out / 'requested-linux-versions.txt')])
        wheels, locked = [], {}
        for path in sorted(wheelhouse.glob('*.whl')):
            with zipfile.ZipFile(path) as archive:
                members = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
                assert len(members) == 1
                metadata = BytesParser().parsebytes(archive.read(members[0]))
            name, version = normalized(metadata['Name']), metadata['Version']
            assert name not in locked, 'Multiple candidate wheels for ' + name
            assert not name.startswith(('nvidia-', 'cuda-', 'rocm-')) and name not in {'triton', 'pytorch-triton'}
            locked[name] = version
            wheels.append({'name': name, 'version': version, 'filename': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path)})
        for name, version in wanted.items():
            assert locked[name] == version, (name, locked.get(name), version)
        lock = ''.join(f'{w["name"]}=={w["version"]} --hash=sha256:{w["sha256"]}\n' for w in sorted(wheels, key=lambda x: x['name']))
        lock_path = out / 'requirements-linux-cpu-py312.lock'
        if lock_path.exists():
            assert lock_path.read_text() == lock
        else:
            lock_path.write_text(lock)
            save(out / 'wheel_manifest.json', {'wheels': wheels, 'total_bytes': sum(w['bytes'] for w in wheels), 'mac_wheel_hashes_reused': False})
        step('install-offline', [str(py), '-m', 'pip', 'install', '--no-index', '--find-links', str(wheelhouse),
             '--require-hashes', '-r', str(lock_path), '--report', str(out / 'pip-install-report.json')])
        step('pip-check', [str(py), '-m', 'pip', 'check'])
        installed = subprocess.run([str(py), '-m', 'pip', 'list', '--format=json'], check=True, capture_output=True, text=True, env=env)
        inventory = {normalized(row['name']): row['version'] for row in json.loads(installed.stdout)}
        for name, version in locked.items(): assert inventory[name] == version
        save(out / 'installed_packages.json', inventory)
        probe = '''import sys,json,platform
import numpy,pandas,pyarrow,torch,scipy,sklearn,sentence_transformers,matplotlib,PIL
import nbformat,nbclient,ipykernel
torch.set_num_threads(2)
assert torch.__version__ == "2.3.1+cpu"
assert torch.version.cuda is None and torch.version.hip is None
assert not torch.cuda.is_available()
assert float(torch.tensor([1.,2.,3.]).sum()) == 6.
print(json.dumps({"python":sys.version.split()[0],"executable":sys.executable,"platform":platform.platform(),"torch":torch.__version__,"cuda_build":torch.version.cuda,"hip_build":torch.version.hip,"cuda_available":torch.cuda.is_available(),"imports_passed":True,"model_forward_tested":False,"training_started":False}))
'''
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
        result = subprocess.run([str(py), '-c', probe], check=True, capture_output=True, text=True, env=env)
        (out / 'import_probe.stderr.log').write_text(result.stderr)
        save(out / 'import_probe.json', json.loads(result.stdout))
        assert sha(manifest_path) == plan['runtime_manifest_sha256']
        assert sha(mac_lock) == plan['historical_macos_lock_sha256']
        assert sha(Path(sys.executable).resolve()) == plan['interpreter_sha256']
        state.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
            installed_distribution_count=len(inventory), linux_lock_sha256=sha(lock_path),
            cpu_only_verified=True, model_forward_tested=False, training_started=False)
        save(state_path, state)
        print(json.dumps({k: state[k] for k in ['status', 'environment', 'installed_distribution_count', 'cpu_only_verified', 'model_forward_tested', 'training_started']}, indent=2), flush=True)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        state.pop('active_child', None)
        save(state_path, state)
        raise


if __name__ == '__main__':
    main()

"""Download public Linux CPU wheels for transfer; do not install on this Mac."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'linux_wheel_transport'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    assert not OUT.exists(), 'Inspect the existing transport rather than overwrite it'
    OUT.mkdir()
    wheels = OUT / 'incoming_wheels'
    wheels.mkdir()
    env = os.environ.copy()
    env.update(PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_CACHE_DIR=str(OUT / 'pip-cache'))
    platforms = ['linux_x86_64'] + [f'manylinux_2_{minor}_x86_64' for minor in range(39, 4, -1)] + ['manylinux2014_x86_64', 'manylinux2010_x86_64', 'manylinux1_x86_64']
    platform_flags = [item for platform in platforms for item in ['--platform', platform]]
    common = [sys.executable, '-m', 'pip', 'download', '--only-binary=:all:', '--python-version', '3.12.3',
              '--implementation', 'cp', '--abi', 'cp312', '--abi', 'abi3', '--abi', 'none', '--dest', str(wheels)]
    commands = [
        ('cpu-torch', common + ['--platform', 'linux_x86_64', '--no-deps', '--index-url', 'https://download.pytorch.org/whl/cpu', 'torch==2.3.1+cpu']),
        ('linux-wheel-candidates', common + platform_flags + ['--index-url', 'https://pypi.org/simple', '--find-links', str(wheels), '-r', str(ROOT / 'requested-linux-versions.txt')]),
    ]
    state = {'started_utc': datetime.now(timezone.utc).isoformat(), 'driver_sha256': sha(__file__),
             'requested_versions_sha256': sha(ROOT / 'requested-linux-versions.txt'),
             'scope': 'Official-source wheel transport only. Pip is running on Mac; actual dependency selection must be repeated offline on Linux before installation.',
             'installed_on_Mac': False, 'stages': [], 'pid': os.getpid(), 'status': 'running'}
    for name, command in commands:
        print('Downloading:', name, flush=True)
        before = time.monotonic()
        with (OUT / (name + '.log')).open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        state['stages'].append({'stage': name, 'command': command, 'returncode': result.returncode,
                                'seconds': time.monotonic()-before, 'log_sha256': sha(OUT / (name+'.log'))})
        state['status'] = 'running' if result.returncode == 0 else 'failed'
        (OUT / 'transport_status.json').write_text(json.dumps(state, indent=2)+'\n')
        assert result.returncode == 0, 'Download failed: ' + str(OUT / (name+'.log'))
    records = []
    for path in sorted(wheels.glob('*.whl')):
        assert 'macosx' not in path.name and 'win_amd64' not in path.name
        assert not path.name.startswith(('nvidia_', 'triton-', 'pytorch_triton-', 'cuda_', 'rocm_'))
        records.append({'filename': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path)})
    state.update(status='downloaded_for_Linux_validation', completed_utc=datetime.now(timezone.utc).isoformat(),
                 files=records, total_bytes=sum(row['bytes'] for row in records))
    (OUT / 'transport_status.json').write_text(json.dumps(state, indent=2)+'\n')
    print(json.dumps({'wheel_candidates': len(records), 'bytes': state['total_bytes'], 'installed_on_Mac': False}, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Durable two-phase lab job; records failure and never silently restarts."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback


def main(r, out):
    drivers = Path(__file__).resolve().parent
    state = drivers / 'job_status.json'
    assert not state.exists(), 'Prior job state exists; inspect before any resume'
    commands = [
        ('preparation', [sys.executable, '-u', str(drivers / 'prepare_inputs.py'), '--study-root', str(r), '--output-root', str(out)]),
        ('inference', [sys.executable, '-u', str(drivers / 'run_inference.py'), '--study-root', str(r), '--input-root', str(out), '--output-root', str(out / 'inference_v1')]),
    ]
    record = {'started_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
              'host': os.uname().nodename, 'executable': sys.executable, 'output_root': str(out),
              'script_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in drivers.glob('*.py')},
              'status': 'running', 'steps': [], 'training_started': False}
    def write():
        record['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temp = drivers / 'job_status.next.json'
        temp.write_text(json.dumps(record, indent=2) + '\n')
        temp.replace(state)
    write()
    try:
        for name, command in commands:
            if name == 'inference':
                (out / 'inference_v1').mkdir(exist_ok=False)
            step = {'name': name, 'command': command, 'status': 'running'}
            record['steps'].append(step); write()
            before = time.monotonic()
            with (drivers / (name + '.stdout.log')).open('x') as stdout, (drivers / (name + '.stderr.log')).open('x') as stderr:
                process = subprocess.Popen(command, stdout=stdout, stderr=stderr, cwd=str(r))
                step['pid'] = process.pid; write()
                step['returncode'] = process.wait()
            step['seconds'] = time.monotonic() - before
            step['status'] = 'complete' if step['returncode'] == 0 else 'failed'
            write()
            if step['returncode'] != 0:
                raise RuntimeError(name + ' failed; logs preserved, no automatic retry')
        record['status'] = 'complete'
    except BaseException:
        record['status'] = 'failed'
        record['error'] = traceback.format_exc()
        raise
    finally:
        write()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    main(args.study_root.resolve(), args.output_root.resolve())

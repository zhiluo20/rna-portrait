"""Durable evaluation followed by independent verification; no partial-model scoring."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback


def main(study,out,drivers):
    study,out,drivers=[p.resolve() for p in [study,out,drivers]]
    state=out/'evaluation_job_status.json'
    assert not state.exists(),'Inspect prior evaluation status before retrying'
    complete=json.loads((out/'inference_v1/execution_summary.json').read_text())
    job=json.loads((drivers/'job_status.json').read_text())
    assert job['status']=='complete' and complete['successful_full_inferences']==3768 and complete['failed_inferences']==0
    scripts=['evaluate_cptac_retrieval.py','verify_cptac_retrieval.py']
    record={'started_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'status':'running','steps':[],
            'script_sha256':{name:hashlib.sha256((drivers/name).read_bytes()).hexdigest() for name in scripts},
            'inference_summary_sha256':hashlib.sha256((out/'inference_v1/execution_summary.json').read_bytes()).hexdigest(),
            'new_training':False}
    def write():
        record['updated_utc']=datetime.now(timezone.utc).isoformat()
        pending=out/'evaluation_job_status.next.json'
        pending.write_text(json.dumps(record,indent=2)+'\n');pending.replace(state)
    write()
    try:
        for name in scripts:
            command=[sys.executable,'-u',str(drivers/name),'--study-root',str(study),'--reassessment-root',str(out)]
            entry={'name':name,'command':command,'status':'running'};record['steps'].append(entry);write()
            before=time.monotonic()
            with (out/(name+'.stdout.log')).open('x') as stdout,(out/(name+'.stderr.log')).open('x') as stderr:
                process=subprocess.Popen(command,cwd=study,stdout=stdout,stderr=stderr)
                entry['pid']=process.pid;write();entry['returncode']=process.wait()
            entry['seconds']=time.monotonic()-before
            entry['status']='complete' if entry['returncode']==0 else 'failed';write()
            if entry['returncode']!=0:raise RuntimeError(name+' failed; no automatic retry')
        record['status']='complete'
    except BaseException:
        record['status']='failed';record['error']=traceback.format_exc();raise
    finally:write()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root',type=Path,required=True)
    parser.add_argument('--reassessment-root',type=Path,required=True)
    parser.add_argument('--drivers-root',type=Path,required=True)
    args=parser.parse_args();main(args.study_root,args.reassessment_root,args.drivers_root)

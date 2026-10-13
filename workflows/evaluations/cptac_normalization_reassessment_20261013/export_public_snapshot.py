"""Export only explicitly allowed source and aggregate preparation records."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main(study,out):
    drivers=Path(__file__).resolve().parent
    public=out/'publication_preparation_v2'
    public.mkdir(exist_ok=False)
    prep=json.loads((out/'preparation.json').read_text())
    protocol=json.loads((out/'PROTOCOL.json').read_text())
    assert prep['protocol_sha256']==sha(out/'PROTOCOL.json')
    assert prep['prepared_inputs_sha256']==sha(out/'prepared_inputs.npz')
    sources=['prepare_inputs.py','run_inference.py','run_job.py','evaluate_cptac_retrieval.py','verify_cptac_retrieval.py',
             'README.md','build_preparation_notebook.py','export_public_snapshot.py']
    for name in sources:
        shutil.copy2(drivers/name,public/name)
    private_manifest=protocol.pop('immutable_old_files_sha256')
    protocol['public_copy_note']='Redacted copy, not the internal frozen original: historical individual-case filename ledger omitted; scientific protocol unchanged.'
    protocol['full_internal_protocol_sha256']=sha(out/'PROTOCOL.json')
    protocol['immutable_historical_file_count']=len(private_manifest)
    protocol['immutable_historical_ledger_sha256']=hashlib.sha256(json.dumps(private_manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (public/'PROTOCOL.public.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for key in ['source_sha256','immutable_old_files_sha256']:
        prep.pop(key)
    prep['full_internal_preparation_sha256']=sha(out/'preparation.json')
    prep['stage']='Input preparation validated. This publication is not model-performance acceptance.'
    (public/'preparation_summary.public.json').write_text(json.dumps(prep,indent=2)+'\n')
    subprocess.run([sys.executable,str(public/'build_preparation_notebook.py'),'--folder',str(public)],check=True)
    sources += ['PROTOCOL.public.json','preparation_summary.public.json','Reassessment_Preparation.ipynb']
    # Identifier boundaries avoid mistaking short numeric IDs for hash substrings.
    # The explicit aggregate/source allowlist is the primary privacy boundary.
    import pandas as pd
    ids=pd.read_csv(study/'repair_v1/paired_candidates/CPTAC_PanCancer/rna_inputs_v1/mechanically_valid_candidates.csv').public_case_id.astype(str).tolist()
    for name in sources:
        text=(public/name).read_text()
        assert not any(re.search(r'(?<![A-Za-z0-9_-])'+re.escape(case)+r'(?![A-Za-z0-9_-])',text) for case in ids),name
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'files':{name:sha(public/name) for name in sources},
              'file_count':len(sources),'participant_level_data_included':False,'weights_included':False,
              'notebook_code_cells_executed':3,'model_performance_evaluated':False,'new_training':False,
              'stage':'Preparation and fixed protocol; inference acceptance and evaluation are separate unfinished milestones.'}
    (public/'public_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=out/'public_preparation_snapshot_v2.tar.gz'
    assert not archive.exists()
    with tarfile.open(archive,'w:gz') as tar:
        for name in [*sources,'public_manifest.json']:tar.add(public/name,arcname=name)
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'archive_sha256':sha(archive),
             'manifest_sha256':sha(public/'public_manifest.json'),'allowlisted_files':len(sources),
             'whole_case_identifier_scan_passed':True,'notebook_code_cells_executed':3,
             'export_revision':'v1 withheld before any upload: a three-digit case ID matched a substring of an unrelated SHA256. v2 uses identifier boundaries; original v1 export retained.',
             'scientific_inputs_or_protocol_changed':False}
    (out/'public_preparation_export.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root',type=Path,required=True)
    parser.add_argument('--reassessment-root',type=Path,required=True)
    args=parser.parse_args();main(args.study_root.resolve(),args.reassessment_root.resolve())

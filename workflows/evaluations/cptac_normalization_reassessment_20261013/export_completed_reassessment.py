"""Seal report QA and export a whitelist of code and aggregate CPTAC evidence."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import base64
import hashlib
import json
import shutil
import tarfile
import pandas as pd


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):
    with path.open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main(study,out,drivers):
    study,out,drivers=[p.resolve() for p in [study,out,drivers]]
    report=out/'report_v2';old=out/'report_v1'
    expected_png='be64c2a6542eb48b29b45ece42cf2d9e3e2e7f92de24b575f70b9bd18215f5e5'
    assert sha(report/'cptac_reassessment_overview.png')==expected_png
    first=json.loads((old/'analysis_summary.json').read_text());second=json.loads((report/'analysis_summary.json').read_text())
    first.pop('completed_utc');second.pop('completed_utc');assert first==second
    for name in ['primary_all_conditions.csv','prespecified_comparators_by_context.csv','paired_question_comparisons.csv','figure_source_data.csv']:
        assert sha(report/name)==sha(old/name),name
    notebook=json.loads((report/'CPTAC_Normalization_Reassessment.ipynb').read_text())
    cells=[c for c in notebook['cells'] if c['cell_type']=='code']
    assert len(cells)==6 and all(c['execution_count'] is not None and all(o['output_type']!='error' for o in c['outputs']) for c in cells)
    images=[o['data']['image/png'] for c in cells for o in c['outputs'] if 'image/png' in o.get('data',{})]
    assert len(images)==1 and hashlib.sha256(base64.b64decode(images[0])).hexdigest()==expected_png
    qa={'recorded_utc':datetime.now(timezone.utc).isoformat(),'visual_inspection':'passed',
        'inspected_png_sha256':expected_png,'notebook_embeds_identical_png':True,'notebook_code_cells_executed':6,
        'revision':'v1 legend overlapped the footnotes. v2 adds bottom space; all analysis summaries and chart-source values are unchanged. Both versions retained.',
        'axes_zero_to_one':True,'labels_and_legend_unclipped':True,'all_32_conditions_retained_in_tables':True,
        'clinical_or_statement_accuracy_claimed':False,'Cell_utility_gate_passed':False}
    save(report/'VISUAL_QA.json',qa)
    baseline=study/'repair_v1/build_cptac_conventional_rankings.py'
    assert "q['positive_features']" in baseline.read_text() and "for _ in q['queries']" in baseline.read_text()
    save(report/'BASELINE_INPUT_SCOPE.json',{
        'source_relative_to_study':str(baseline.relative_to(study)),'source_sha256':sha(baseline),
        'structured_methods_use_predeclared_query_to_panel_map':True,
        'structured_scores_repeated_across_paraphrases_by_design':True,
        'interpretation':'Fixed-panel retrieval is a strong task reference, not evidence of open-ended free-text comprehension. Report query-interface differences; do not treat Jaccard 1 as a learned language advantage.'})
    public=out/'publication_completed_v1';public.mkdir(exist_ok=False)
    files={}
    for p in report.iterdir():
        if p.is_file() and not p.name.startswith('.'):
            files['report_v2/'+p.name]=p
    for name in ['contract.json','descriptive_macro_results.csv','per_query_paraphrase_results.csv','per_question_results.csv',
                 'query_candidate_coverage.csv','paraphrase_stability.csv','evaluation_summary.json','independent_validation.json']:
        files['evaluation_v1/'+name]=out/'evaluation_v1'/name
    for name in ['run_evaluation_job.py','build_reassessment_report.py','export_completed_reassessment.py']:
        files[name]=drivers/name
    for name in ['cptac_retrieval_metrics.py','tie_aware_retrieval.py','build_cptac_conventional_rankings.py']:
        files['reference/'+name]=study/'repair_v1'/name
    files['README.md']=drivers/'COMPLETED_README.md'
    forbidden={'sample_id','public_case_id','library_id','case_submitter_id','sample_ids','library_ids','sample_to_split','candidate_ids','output_sha256'}
    def check_json(obj):
        if isinstance(obj,dict):
            assert not set(obj).intersection(forbidden)
            for value in obj.values():check_json(value)
        elif isinstance(obj,list):
            for value in obj:check_json(value)
    for name,path in files.items():
        if path.suffix=='.csv':assert not set(pd.read_csv(path,nrows=0).columns).intersection(forbidden),name
        if path.suffix=='.json':check_json(json.loads(path.read_text()))
        target=public/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        assert sha(target)==sha(path)
    summary_path=out/'inference_v1/execution_summary.json'
    inference=json.loads(summary_path.read_text());private_outputs=inference.pop('output_sha256')
    inference.update(full_internal_summary_sha256=sha(summary_path),private_output_file_count=len(private_outputs),
                     public_copy_note='Per-case output filename ledger omitted. Complete internal summary retained on SSD.')
    check_json(inference)
    save(public/'inference_summary.public.json',inference)
    names=[*files,'inference_summary.public.json']
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'files':{name:sha(public/name) for name in names},
              'file_count':len(names),'participant_level_data_included':False,'weights_included':False,
              'real_retrieval_evaluation_complete':True,'Cell_utility_gate_passed':False,'new_training_started':False,
              'privacy_check':'Explicit aggregate/code whitelist; no sample-identifier columns or per-sample JSON keys. The private per-case output ledger is explicitly removed.'}
    save(public/'public_manifest.json',manifest)
    archive=out/'public_completed_snapshot.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for name in [*names,'public_manifest.json']:tar.add(public/name,arcname=name)
    receipt={'archive_sha256':sha(archive),'public_manifest_sha256':sha(public/'public_manifest.json'),
             'files':len(names),'archive_bytes':archive.stat().st_size,'visual_QA_passed':True,
             'notebook_code_cells_executed':6,'Cell_utility_gate_passed':False}
    save(out/'public_completed_export.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root',type=Path,required=True);parser.add_argument('--reassessment-root',type=Path,required=True);parser.add_argument('--drivers-root',type=Path,required=True)
    args=parser.parse_args();main(args.study_root,args.reassessment_root,args.drivers_root)

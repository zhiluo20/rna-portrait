"""Produce aggregate-only CPTAC findings, a reproducible figure and notebook.

Requires complete frozen inference and independent evaluation. Does not rerun
models, change thresholds, fit anything, or modify historical result files.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat as nbf
from nbclient import NotebookClient


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):
    with path.open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main(root):
    root=root.resolve();evaluation=root/'evaluation_v1';output=root/'report_v2'
    assert json.loads((root/'evaluation_job_status.json').read_text())['status']=='complete'
    inference=json.loads((root/'inference_v1/execution_summary.json').read_text())
    validation=json.loads((evaluation/'independent_validation.json').read_text())
    old_comparison=json.loads((evaluation/'evaluation_summary.json').read_text())
    assert inference['successful_full_inferences']==3768 and inference['failed_inferences']==0
    assert validation['status']=='passed' and validation['metric_groups']==37440 and validation['methods']==32
    source_files=[evaluation/x for x in ['descriptive_macro_results.csv','per_question_results.csv',
                  'paraphrase_stability.csv','query_candidate_coverage.csv','independent_validation.json','evaluation_summary.json']]
    source_files += [root/'PROTOCOL.json',root/'inference_v1/execution_summary.json',root/'inference_v1/portrait_texts.csv']
    inputs={str(p.relative_to(root)):sha(p) for p in source_files}
    output.mkdir(exist_ok=False)
    macro=pd.read_csv(evaluation/'descriptive_macro_results.csv')
    primary=macro[(macro.scope=='pooled')&(macro.stratum=='all')].set_index('method')
    questions=pd.read_csv(evaluation/'per_question_results.csv')
    question_matrix=questions[(questions.scope=='pooled')&(questions.stratum=='all')].pivot(index='query_id',columns='method',values='conditional_ndcg_at_10')
    assert question_matrix.shape==(30,32) and question_matrix.notna().all().all()
    portrait='full_portrait_released_calibration_sf_reference'
    comparators=['full_portrait_text_released_calibration','compact_marker_rank','combined_conventional','cellwhisperer_v1']
    deltas=[]
    for baseline in comparators:
        difference=question_matrix[portrait]-question_matrix[baseline]
        deltas.append({'model':portrait,'comparator':baseline,'queries':30,'wins':int((difference>1e-12).sum()),
                       'ties':int((difference.abs()<=1e-12).sum()),'losses':int((difference<-1e-12).sum()),
                       'mean_ndcg_difference':float(difference.mean()),'median_ndcg_difference':float(difference.median())})
    pd.DataFrame(deltas).to_csv(output/'paired_question_comparisons.csv',index=False)
    coverage=pd.read_csv(evaluation/'query_candidate_coverage.csv')
    pooled_coverage=coverage[(coverage.scope=='pooled')&(coverage.stratum=='all')]
    texts=pd.read_csv(root/'inference_v1/portrait_texts.csv').pivot(index='sample_id',columns='method',values='portrait_text')
    calibration={arm:int((texts['full_portrait_released_calibration_'+arm]==texts['full_portrait_no_calibration_'+arm]).sum()) for arm in ['sf_reference','sf_full_measured']}
    stability=pd.read_csv(evaluation/'paraphrase_stability.csv')
    stable=stability[(stability.scope=='pooled')&(stability.stratum=='all')].groupby('method').conditional_top10_jaccard.mean()
    primary=primary.join(stable)
    primary.to_csv(output/'primary_all_conditions.csv')
    selected=['compact_marker_rank','combined_conventional','compact_marker_zscore','cellwhisperer_v1',
              'fixed_conventional_text_minilm',portrait,'rna_direct_sf_reference','rna_fused_sf_reference']
    label={'compact_marker_rank':'Marker rank','combined_conventional':'Conventional combination',
           'compact_marker_zscore':'Marker z-score','cellwhisperer_v1':'CellWhisperer',
           'fixed_conventional_text_minilm':'Conventional fixed text',portrait:'Full portrait · primary SF',
           'rna_direct_sf_reference':'Direct RNA · primary SF','rna_fused_sf_reference':'Fused RNA · primary SF'}
    strata=macro[macro.method.isin(selected)].copy()
    strata.to_csv(output/'prespecified_comparators_by_context.csv',index=False)
    summary={'completed_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':inputs,
             'candidate_cases':942,'questions':30,'paraphrases_per_question':3,'methods':32,'contexts':13,
             'complete_portrait_inferences':3768,'failed_inferences':0,
             'judged_cases_per_pooled_query_range':[int(pooled_coverage.judged_cases.min()),int(pooled_coverage.judged_cases.max())],
             'primary_all_conditions':json.loads(primary.reset_index().to_json(orient='records',double_precision=15)),
             'paired_question_comparisons':deltas,'identical_six_field_texts_with_calibration_on_off':calibration,
             'unique_portrait_texts':inference['unique_portrait_texts'],
             'old_metric_groups_reproduced':old_comparison['old_metric_groups_reproduced'],
             'old_metric_max_abs_error':old_comparison['old_metric_max_abs_error'],
             'independently_verified_metric_groups':37440,'independently_verified_paraphrase_stability_groups':12480,
             'max_independent_metric_abs_error':max(validation['max_metric_abs_difference'].values()),
             'Cell_utility_gate_passed':False,'new_training_started':False,'manuscript_rewritten':False,
             'interpretation':'Normalization repair improves pooled RNA scores but does not establish practical gain over the frozen strong baselines. It is not a uniform improvement across all queries or contexts.',
             'decision':'Apply the original pilot gate: do not expand training, confirmatory studies, wet-lab investment or Cell-promotional rewriting without an author decision about the scientific route.',
             'limitations':['Conditional protein-evidence pools are nonrandom; missing evidence is not a negative label.',
                            'Protein panels are molecular proxies, not functional truth or human statement ratings.',
                            'Queries share panels and paraphrases are repeated measures; no independent-query or per-paraphrase superiority inference.',
                            '443 known test cases / 499 unlinked cases does not establish strict new patient or development independence.',
                            'Single shuffled/mismatched controls are descriptive; do not infer stable shuffle superiority or model input independence.',
                            'Full-job wall time overlapped separate audits and is not an isolated end-to-end speed benchmark.']}
    save(output/'analysis_summary.json',summary)
    chart_contract={'created_utc':datetime.now(timezone.utc).isoformat(),
                    'question':'How did normalization change pooled CPTAC retrieval, and how does primary recovered-scale retrieval compare with fixed baselines?',
                    'takeaway':'Pooled scores improve, but primary recovered-scale representations remain below fixed strong baselines.',
                    'surface':'Exported author-research diagnostic figure embedded in an executed notebook; not a manuscript main figure or HTML dashboard.',
                    'family':'Comparison and ranking','variant':'Grouped horizontal bars and ranked horizontal bars',
                    'expected_rows':{'normalization_panel':9,'baseline_panel':8},'grain':'Equal-query macro over 30 questions, each averaged over three rewrites',
                    'denominator':'Per-query common measurable protein pools among 942 locked cases; 146 to 942 judged cases.',
                    'renderer':'Matplotlib static PNG and SVG','footprint_inches':[13.4,7.8],
                    'palette_policy':'single-root preferred','colors':{'primary_blue':'#3979A8','blue_light':'#D4E3EF','ink':'#26323D','neutral':'#A4ABB2','grid':'#E6E9EC'},
                    'non_color_distinction':'Legend, semantic row groups, direct numeric labels, open/hatched sensitivity bars',
                    'axes':'Both panels start at zero and end at one; no uncertainty interval implied.',
                    'omitted_conditions':'Both calibration modes are retained in CSV; their six-field texts are exactly identical within each arm. All 32 conditions and 13 contexts remain in source tables.',
                    'branding':'Third-party author manuscript research; no OpenAI blossom or branding.',
                    'source_sha256':inputs,'visual_QA':'Must inspect exported PNG and verify notebook embeds the identical bytes.'}
    save(output/'chart_contract.json',chart_contract)
    colors=chart_contract['colors'];plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelcolor':colors['ink'],'text.color':colors['ink'],'xtick.color':colors['ink'],'ytick.color':colors['ink']})
    figure,(left,right)=plt.subplots(1,2,figsize=(13.4,7.8),gridspec_kw={'width_ratios':[1,1.2]})
    figure.subplots_adjust(left=.11,right=.975,bottom=.28,top=.76,wspace=.75)
    figure.suptitle('CPTAC retrieval: normalization and fixed baselines',x=.04,y=.97,ha='left',fontsize=20,fontweight='bold')
    figure.text(.04,.915,'Pooled nDCG@10 on common measurable protein-evidence candidates',fontsize=12)
    figure.text(.04,.876,'942 fixed cases · 30 questions × 3 paraphrases · equal-question descriptive means',fontsize=11,color='#58636D')
    rows=['Direct RNA','Fused RNA','Full portrait']
    arms=[('Legacy reference CPM',['rna_direct','rna_fused','full_portrait_text_released_calibration'],colors['neutral'],None),
          ('Recovered reference SF',['rna_direct_sf_reference','rna_fused_sf_reference',portrait],colors['primary_blue'],None),
          ('Recovered all-gene SF',['rna_direct_sf_full_measured','rna_fused_sf_full_measured','full_portrait_released_calibration_sf_full_measured'],'white','///')]
    source=[]
    for j,(legend,methods,fill,hatch) in enumerate(arms):
        y=np.arange(3)+(j-1)*.24
        values=[float(primary.at[m,'conditional_ndcg_at_10']) for m in methods]
        left.barh(y,values,height=.205,color=fill,edgecolor=colors['primary_blue'] if hatch else colors['ink'],linewidth=.65,hatch=hatch,label=legend,zorder=3)
        for yy,value,method in zip(y,values,methods):
            left.text(value+.012,yy,f'{value:.3f}',va='center',fontsize=9)
            source.append({'panel':'normalization','label':legend,'method':method,**primary.loc[method].to_dict()})
    left.set_yticks(np.arange(3),rows);left.invert_yaxis();left.set_title('A  RNA representation and scale',loc='left',fontsize=12,pad=14)
    left.legend(loc='upper left',bbox_to_anchor=(-.03,-.16),frameon=False,fontsize=9,handlelength=2.1)
    values=primary.loc[selected,'conditional_ndcg_at_10'].to_numpy(float)
    fill=[colors['primary_blue'] if m in selected[-3:] else colors['neutral'] for m in selected]
    right.barh(np.arange(8),values,height=.64,color=fill,edgecolor=colors['ink'],linewidth=.65,zorder=3)
    right.set_yticks(np.arange(8),[label[m] for m in selected]);right.invert_yaxis()
    right.set_title('B  Same-task comparisons',loc='left',fontsize=12,pad=14)
    for yy,(method,value) in enumerate(zip(selected,values)):
        right.text(value+.015,yy,f'{value:.3f}',va='center',fontsize=10)
        source.append({'panel':'baselines','label':label[method],'method':method,**primary.loc[method].to_dict()})
    for axis in [left,right]:
        axis.set_xlim(0,1);axis.set_xticks(np.linspace(0,1,6));axis.set_xlabel('nDCG@10',labelpad=9)
        axis.grid(axis='x',color=colors['grid'],linewidth=.7,zorder=0)
        axis.spines[['top','right']].set_visible(False)
        for side in ['left','bottom']:axis.spines[side].set_color('#69737C');axis.spines[side].set_linewidth(.7)
        axis.tick_params(axis='y',length=0,pad=8)
    figure.text(.04,.07,'Judged pool: 146–942 cases per question. Unjudged cases are not treated as negatives.',fontsize=10,color='#58636D')
    figure.text(.04,.035,'Protein-panel concordance is not functional or statement accuracy. All 32 conditions and all 13 contexts are retained in the tables.',fontsize=9,color='#58636D')
    for extension in ['png','svg']:figure.savefig(output/('cptac_reassessment_overview.'+extension),dpi=180,facecolor='white')
    plt.close(figure)
    chart_source=pd.DataFrame(source);chart_source['candidate_cases']=942;chart_source['questions']=30;chart_source['paraphrases_per_question']=3
    chart_source['judged_cases_min']=146;chart_source['judged_cases_max']=942
    chart_source.to_csv(output/'figure_source_data.csv',index=False)
    notebook=nbf.v4.new_notebook();notebook.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
    notebook.cells=[
        nbf.v4.new_markdown_cell('# CPTAC 固定候选池归一化复评\n\n## tl;dr\n\n完整 portrait 的 pooled nDCG@10 从 0.169 升至 0.218，但低于 CellWhisperer 0.307、传统组合 0.631 和 marker-rank 0.655。输入修复没有证明原计划要求的实用增益；不据此启动扩大投入或把稿件标为 Cell-ready。'),
        nbf.v4.new_markdown_cell('## Context & Methods\n\n### Key Assumptions\n\n同一组 942 个候选、30 个问题和每题 3 个改写，18 个旧条件加 14 个新条件。归一化在全候选池拟合，属于传导式方案。本轮已见过旧结果，不是确认性研究。每题各方法共用可测蛋白证据池；缺失不补成阴性。'),
        nbf.v4.new_code_cell("from pathlib import Path\nimport json, hashlib\nimport pandas as pd\nfrom IPython.display import Image, display\nroot = Path.cwd()\nreport = json.loads((root/'analysis_summary.json').read_text())\nassert report['complete_portrait_inferences']==3768 and report['failed_inferences']==0\nassert report['independently_verified_metric_groups']==37440\nassert report['max_independent_metric_abs_error']<1e-12\nassert report['old_metric_groups_reproduced']==21060\nassert not report['Cell_utility_gate_passed']\nprint('推断、独立指标重建及旧结果回归检查通过；实用价值门槛未通过。')"),
        nbf.v4.new_markdown_cell('## Data\n\n可判断病例数随问题为 146–942。30 个问题也共享生物学面板，90 种文字改写不是 90 个独立实验。'),
        nbf.v4.new_code_cell("all_conditions = pd.read_csv(root/'primary_all_conditions.csv').set_index('method')\nassert len(all_conditions)==32\nprint('候选数:', report['candidate_cases'], '; 每题可判断范围:', report['judged_cases_per_pooled_query_range'])\nall_conditions[['conditional_ndcg_at_10','conditional_precision_at_10','full_pool_top10_judged_fraction','conditional_top10_jaccard']].round(4)"),
        nbf.v4.new_markdown_cell('## Results\n\n### 归一化与同任务比较\n\n图仅用于本轮探索性复评，不替换原计划中的六个投稿主图。完整条件表保留打乱、错配、屏蔽和两种校准设置。'),
        nbf.v4.new_code_cell("display(Image(filename=str(root/'cptac_reassessment_overview.png')))"),
        nbf.v4.new_markdown_cell('### 分层结果\n\n未关联病例不等于已证明未暴露。within-cohort 使用原有队列内蛋白参照，不能择优替换 pooled 主分析。'),
        nbf.v4.new_code_cell("strata = pd.read_csv(root/'prespecified_comparators_by_context.csv')\nstrata.pivot(index='method',columns=['scope','stratum'],values='conditional_ndcg_at_10').round(4)"),
        nbf.v4.new_markdown_cell('### 逐问题差异\n\n先在问题内平均改写，再比较问题。下表是描述性计数，不是独立问题的显著性检验。'),
        nbf.v4.new_code_cell("pd.read_csv(root/'paired_question_comparisons.csv')[['comparator','queries','wins','ties','losses','mean_ndcg_difference']].round(4)"),
        nbf.v4.new_markdown_cell('### 校准与描述\n\n这里比较完整的六字段检索文本，不声称所有内部校准数值都相同。'),
        nbf.v4.new_code_cell("assert report['identical_six_field_texts_with_calibration_on_off']=={'sf_reference':942,'sf_full_measured':942}\npd.DataFrame.from_dict(report['unique_portrait_texts'],orient='index',columns=['unique_six_field_texts'])"),
        nbf.v4.new_markdown_cell('## Takeaways\n\n本轮数字经独立实现核对，但生物学实用性没有通过推进标准。保留全部弱结果、旧 CPM 结果和失败条件，不通过删选查询、换主分析或追加拒答来制造增益。后续研究路线和 Cell 定位需要作者决定。原稿未改写，未启动重训；人工语句盲评、独立确认性收益和投稿材料尚未完成。'),
    ]
    nbf.validate(notebook)
    NotebookClient(notebook,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(output)}}).execute()
    assert all(c.execution_count is not None and all(o.output_type!='error' for o in c.outputs) for c in notebook.cells if c.cell_type=='code')
    nbf.write(notebook,output/'CPTAC_Normalization_Reassessment.ipynb')
    for relative,expected in inputs.items():assert sha(root/relative)==expected,relative
    save(output/'REPORT_BUILD_QA.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),'notebook_code_cells_executed':6,
                                      'figure_png_sha256':sha(output/'cptac_reassessment_overview.png'),
                                      'source_files_unchanged':True,'visual_QA':'pending inspection of exported PNG',
                                      'Cell_utility_gate_passed':False,'new_training_started':False})
    print(json.dumps({'report':str(output),'notebook_code_cells_executed':6,'figure_png_sha256':sha(output/'cptac_reassessment_overview.png')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reassessment-root',type=Path,required=True)
    main(parser.parse_args().reassessment_root)

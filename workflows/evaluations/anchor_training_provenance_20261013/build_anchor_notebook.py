"""Execute an aggregate-only reader for the archived anchor provenance audit."""
from pathlib import Path
import argparse
import nbformat as nbf
from nbclient import NotebookClient


def main(out):
    out=out.resolve()
    nb=nbf.v4.new_notebook()
    nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
    nb.cells=[
        nbf.v4.new_markdown_cell('# Anchor 主干训练划分证据\n\n## tl;dr\n\n找到与冻结权重对应的原运行 sidecar，并交叉核验 25,199 个样本。全部 8,192 个表达标准化参数精确一致，两个保存的预测字段在预定容差内复现。它补充原主干来源证据，不证明整个系统训练独立或生物学有效。'),
        nbf.v4.new_markdown_cell('## Context & Methods\n\n### Key Assumptions\n\n原 sidecar 是归档证据，不是 checkpoint 内置划分。通过权重 SHA、模型类 AST、训练参数和全部保存预测交叉核对其对应关系。数值重放不是临床预测准确率。样本级原始证据保留在 SSD，不出现在此 notebook。'),
        nbf.v4.new_code_cell("from pathlib import Path\nimport json, hashlib\nimport pandas as pd\nroot = Path.cwd()\nresult = json.loads((root/'RESULT.json').read_text())\nassert hashlib.sha256((root/'PROTOCOL.json').read_bytes()).hexdigest() == result['protocol_sha256']\nassert result['sidecar_samples'] == 25199\nassert result['split_disagreements_vs_previous_runtime'] == 0\nassert result['missing_from_previous_runtime_split'] == 0\nassert result['original_source_files_unchanged']\nassert result['live_runtime_and_CPTAC_pool_unchanged']\nprint('原始材料、划分对应和未改动检查通过。')"),
        nbf.v4.new_markdown_cell('## Data\n\n样本划分与固定 CPTAC 候选的既有病例关联：'),
        nbf.v4.new_code_cell("assert sum(result['archived_split_counts'].values()) == 25199\nassert result['CPTAC_candidate_anchor_linkage_counts'] == {'not_linked_by_existing_crosswalk':499, 'test':443}\npd.DataFrame.from_dict(result['archived_split_counts'], orient='index', columns=['样本数'])"),
        nbf.v4.new_markdown_cell('## Results\n\n这里比较的是数值记录一致性，不是与疾病或功能真值的比较。'),
        nbf.v4.new_code_cell("for field in ['expr_mean','expr_std']:\n    assert result['standardization_checks'][field]['exact_values'] == 4096\n    assert result['standardization_checks'][field]['max_abs_error'] == 0\nchecks = pd.DataFrame(result['prediction_replay_checks']).T\nassert checks['all_within_tolerance'].all()\nassert (checks['within_tolerance_records'] == 25199).all()\nchecks[['records','max_abs_error','tolerance_abs','within_tolerance_records']]"),
        nbf.v4.new_markdown_cell('## Takeaways\n\n原 anchor 主干 sidecar 已恢复并得到交叉支持。checkpoint 本身仍不含划分；后续 heads/priors/规则和患者级独立性未由本检查解决。499 个未关联 CPTAC 病例仍不能称为严格未暴露病例。原稿、正在运行的推断、候选池和权重均未修改。'),
    ]
    target=out/'Anchor_Training_Provenance.ipynb'
    assert not target.exists()
    nbf.validate(nb)
    NotebookClient(nb,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(out)}}).execute()
    assert all(c.execution_count is not None and all(o.output_type!='error' for o in c.outputs) for c in nb.cells if c.cell_type=='code')
    nbf.write(nb,target)
    print('Executed anchor provenance notebook: 3 code cells',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-root',type=Path,required=True)
    main(parser.parse_args().output_root)

"""Build and execute a bounded, aggregate-only preparation audit notebook."""
from pathlib import Path
import argparse
import nbformat as nbf
from nbclient import NotebookClient


def main(folder):
    folder=folder.resolve()
    notebook=nbf.v4.new_notebook()
    notebook.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
    notebook.cells=[
        nbf.v4.new_markdown_cell('# CPTAC 归一化复评：输入准备核对\n\n## tl;dr\n\n942 个病例的原始计数与旧记录一致，旧 CPM 输入精确重建。两种新 size factor 的独立计算误差小于 2×10⁻¹⁴。这里只验收输入准备，不报告尚未核验的检索表现。'),
        nbf.v4.new_markdown_cell('## Context & Methods\n\n### Key Assumptions\n\n沿用全部 30 个问题和原候选池。942 个文库联合拟合，因此属于传导式预处理，不是训练集冻结的单样本接口。原结果已被观察，本轮不是确认性试验。完整内部协议保留在 SSD，公开副本去掉了含病例文件名的清单，并保留其摘要和原协议哈希。'),
        nbf.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nroot = Path.cwd()\nprotocol = json.loads((root / 'PROTOCOL.public.json').read_text())\nsummary = json.loads((root / 'preparation_summary.public.json').read_text())\nassert summary['samples'] == 942\nassert summary['all_raw_columns_match_old_QC_hashes']\nassert summary['old_CPM_reconstruction_exact']\nassert not summary['qrel_values_read']\nassert summary['old_files_unchanged']\nprint('输入一致性、旧流程重建和无标签准备检查通过。')"),
        nbf.v4.new_markdown_cell('## Data\n\n固定映射后的基因覆盖，不将未映射与真实零表达混淆。'),
        nbf.v4.new_code_cell("pd.DataFrame([{'病例数': summary['samples'], '映射参考基因数': summary['mapped_reference_genes'], '可测选定基因数': summary['measured_selected_genes'], '缺失选定基因数': 4096-summary['measured_selected_genes']}])"),
        nbf.v4.new_markdown_cell('## Results\n\n表中为输入准备的汇总，不是模型有效性结果。'),
        nbf.v4.new_code_cell("statistics = pd.DataFrame(summary['summaries']).T\nassert (statistics['independent_factor_max_abs_error'] < 2e-14).all()\nassert (statistics['factor_min'] > 0).all()\nstatistics[['fit_genes', 'positive_in_all_samples_genes', 'factor_min', 'factor_median', 'factor_max', 'independent_factor_max_abs_error']]"),
        nbf.v4.new_markdown_cell('## Takeaways\n\n输入准备已通过；仍须完成全部推断、共有可测蛋白池的公平比较和独立指标重建。保留所有旧条件、新条件和阴性结果；不把技术一致性当作生物学有效性或 Cell 投稿准备完成。完整命令和缺失依赖边界见 README。'),
    ]
    target=folder/'Reassessment_Preparation.ipynb'
    assert not target.exists()
    nbf.validate(notebook)
    NotebookClient(notebook,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(folder)}}).execute()
    assert all(c.execution_count is not None and all(o.output_type!='error' for o in c.outputs) for c in notebook.cells if c.cell_type=='code')
    nbf.write(notebook,target)
    print('Executed preparation notebook: 3 code cells',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,required=True)
    main(parser.parse_args().folder)

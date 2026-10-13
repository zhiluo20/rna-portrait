"""Execute the audit notebook and render its bounded outputs without nbconvert."""
from pathlib import Path
import html
import json
import re
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

OUT = Path(__file__).resolve().parent
assert not (OUT / 'FINAL_MANIFEST.json').exists(), 'Do not rewrite a frozen run'
summary = json.loads((OUT / 'evaluation_summary.json').read_text())
means = {row['method']: row for row in summary['means']}
primary = means['full_portrait_released_calibration_sf_reference']
direct = means['rna_direct_sf_reference']
shuffle = means['rna_value_shuffled_sf_reference']
epic = means['EPIC_BRef_raw']
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
nb = nbformat.v4.new_notebook()
nb.cells = [
    md('# SDY2823归一化复评：SSD重算记录\n\n## tl;dr\n\n'
       '2026-10-13，固定96位供者、模型权重、原始查询和流式评价标签，完成384次完整portrait推断。'
       '本轮增加10种条件，连同17种旧条件共复核243个问题—改写—方法指标组。\n\n'
       f"主条件完整portrait的平均nDCG@10为{primary['ndcg_at_k']:.3f}，并列分数期望为{primary['tie_expected_ndcg_at_10']:.3f}；"
       f"EPIC为{epic['ndcg_at_k']:.3f}。RNA直接检索为{direct['ndcg_at_k']:.3f}，本次基因打乱对照为{shuffle['ndcg_at_k']:.3f}。\n\n"
       '**本轮没有显示预期的独立细胞状态检索增益，不支持据此进入Cell确认性研究或宣称模型具有检索优势。** '
       '这是3个相关问题在单一队列中的探索性结果，不能推出模型在所有生物学任务上均无效。'),
    md('## Context & Methods\n\n'
       '原工作目录迁移期间，SSD副本没有纳入刚完成的推断输出。作者明确授权在SSD重算；本轮没有读取或恢复废纸篓。'
       '原计算曾报告成功，但其文件未能重新核验，本报告只采用当前SSD重算的可验证结果。\n\n'
       '### Key Assumptions\n\n'
       '- 主条件在23,323个已映射实测基因上估计队列size factors；敏感性条件在58,302个实测Ensembl行上估计后再映射。两者均只用同一96个候选样本。\n'
       '- 这是利用整个候选队列估计参数的无监督归一化，不是仅由训练集确定的参考，也不是单样本部署协议。\n'
       '- 30题各3种改写均已打分；只有Q01、Q02、Q19具备本轮T/B流式相关性标签。其余27题不能计为已经验证。\n'
       '- 每题先平均3种改写，再对3题取均值；不把9个改写当作9个独立生物学实验，也不给一般性优越性的P值。\n'
       '- 所有17个旧方法及其原始预处理保留。EPIC/MCP-counter使用固定的查询到分子特征映射，非通用自由文本解析；该人工映射成本没有测量。\n'
       '- 两个RNA分支接收同一预处理输入；禁用第二次log/CPM和结构化分支的备用归一化。未重训、未调整阈值或依据结果选择归一化。'),
    md('## Data\n\n### 1. 校验输入、输出和指标\n\n'
       '父目录PROTOCOL.json与preparation.json锁定原始输入。execution_contract.json锁定代码、权重、依赖和查询；'
       'execution_summary.json与evaluation_summary.json分别索引推断输出和评价表。verify_outputs.py独立重算全部900组新得分矩阵以及243组评价指标。'),
    code("from pathlib import Path\nimport json\nimport numpy as np\nimport pandas as pd\nfrom verify_outputs import verify\nout = Path.cwd()\ninput_root = out.parent\nstudy_root = input_root.parent\nchecks = verify(study_root, input_root, out)\nassert checks == json.loads((out/'independent_validation.json').read_text())\nsummary = json.loads((out/'evaluation_summary.json').read_text())\nassert summary['previous_153_metric_groups_maximum_difference'] < 1e-12\nprint(f\"900组得分矩阵重构误差 {checks['score_max_abs_error']:.1f}；243组独立指标最大误差 {checks['independent_metric_max_abs_error']:.3g}；384份推断及输入路径验证通过。\")\nprint('旧153组指标仅存在浮点求和量级差异，原始文件哈希未变。')"),
    md('## Results\n\n### 2. 主条件与预定对照\n\n'
       'nDCG的范围为0–1，越高表示前10名中更相关的样本排序越靠前。高等级候选数为每10个返回样本中达到固定流式上四分位等级的平均数量，不是人工认可的statement数量。'),
    code("means = pd.read_csv(out/'three_question_descriptive_means.csv', float_precision='round_trip').set_index('method')\npd.set_option('display.max_colwidth',None)\npd.set_option('display.max_rows',40)\nselected = {\n 'EPIC_BRef_raw':'EPIC（固定查询映射）',\n 'cellwhisperer_v1':'CellWhisperer',\n 'full_portrait_text_released_calibration':'完整portrait：旧CPM',\n 'full_portrait_released_calibration_sf_reference':'完整portrait：主条件',\n 'rna_direct_sf_reference':'RNA直接空间：主条件',\n 'rna_fused_sf_reference':'RNA融合空间：主条件',\n 'rna_value_shuffled_sf_reference':'主条件：基因值打乱',\n 'rna_mismatched_sf_reference':'主条件：样本错配'}\nview = means.loc[list(selected), ['ndcg_at_k','tie_expected_ndcg_at_10','strong_hits_per_10']].rename(index=selected,columns={'ndcg_at_k':'nDCG@10','tie_expected_ndcg_at_10':'并列期望nDCG@10','strong_hits_per_10':'高等级候选数/10'})\nview.round(3)"),
    md('### 3. 分问题检查\n\n'
       'Q01为T细胞，Q02为B细胞，Q19为二者组合。三题有生物学相关性。以下保留预定主条件和对照，不因均值不利而删除个别问题。'),
    code("per_question = pd.read_csv(out/'per_question_results.csv')\ncomparison = ['EPIC_BRef_raw','cellwhisperer_v1','full_portrait_released_calibration_sf_reference','rna_direct_sf_reference','rna_value_shuffled_sf_reference']\ntable = per_question.loc[per_question.method.isin(comparison)].pivot(index='method',columns='query_id',values='ndcg_at_k').loc[comparison]\ntable.rename(index=selected).round(3)"),
    md('### 4. 全部27个条件\n\n'
       '两种基因全集条件的完整portrait指标相同；校准开关在各条件中也未改变96份被检索的文字。它们不是独立重复实验。'
       '完整文字由旧CPM的23种增加到56种，并不代表检索相关性或语句正确率提高。'),
    code("assert len(means)==27\nassert set(summary['full_portrait_unique_texts_by_condition'].values()) == {56}\nassert set(summary['same_text_with_and_without_calibration'].values()) == {96}\nmeans[['ndcg_at_k','tie_expected_ndcg_at_10','precision_at_k','paraphrase_top10_jaccard']].rename(columns={'ndcg_at_k':'nDCG@10','tie_expected_ndcg_at_10':'并列期望','precision_at_k':'Precision@10','paraphrase_top10_jaccard':'改写Jaccard'}).round(3)"),
    md('### 5. 输入完整性和诊断边界\n\n'
       '每个样本匹配3,524/4,096个选定模型基因，未解析的572个仍按冻结模型的零输入约定处理；没有将缺失解释为生物学零。'
       '以下是实测特征超出训练标准化后±8裁切范围的比例中位数。没有丢弃表现不佳或覆盖不足的样本。'),
    code("diagnostics = pd.read_csv(out/'input_diagnostics.csv')\nassert len(diagnostics)==192 and diagnostics.matched_genes.eq(3524).all()\nassert diagnostics.normalization_library_sum.isna().all()\nprint('96位供者全部保留；两种条件各96；完整推断失败0；没有第二次CPM转换。')\ndiagnostics.groupby('arm')[['matched_feature_upper_clip_fraction','matched_feature_lower_clip_fraction']].median().rename(columns={'matched_feature_upper_clip_fraction':'上限裁切比例中位数','matched_feature_lower_clip_fraction':'下限裁切比例中位数'}).round(6)"),
    md('## Takeaways\n\n'
       '- 计算与数据连接通过核查：全部输入/输出哈希、384份逐样本记录、900组得分和243组指标可追溯，没有候选丢失或重复连接。\n'
       '- 科学推进门槛未通过：恢复历史归一化公式没有建立相对强传统基线的状态检索增益；基因打乱对照也未出现预期的性能下降。单次打乱不能估计随机对照分布，更不能解释为模型鲁棒。\n'
       '- 这不能单独证明失败的根因。模型训练目标、跨平台差异、未解析基因和细胞组成任务的匹配程度仍需有界诊断；不要在已看过结果的3题上反复调参后当作确认性证据。\n'
       '- 按原计划，先保留阴性结果并重新评估研究定位，不扩大确认性或湿实验投入，不把格式完成写成Cell投稿准备完成。\n'
       '- 全部过程保存在当前SSD目录；主稿v1.12、提交原件、历史结果和权重不变。独立人工语句盲评、anchor训练划分来源、单样本部署参考和新留出队列的实用增益仍未解决。\n\n'
       '本报告的质量结论是“可作为有明确边界的探索性阴性结果分享”，不是“已达到Cell投稿科学门槛”。时间为单次推断日志，不用于跨方法速度结论。没有新的时间序列数据，故不作时间趋势分析。')
]
nb.metadata = {'kernelspec': {'display_name': 'SDY audit Python', 'language': 'python', 'name': 'python3'}}
manager = KernelManager(kernel_name='python3')
manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
NotebookClient(nb, timeout=240, kernel_name='python3', km=manager,
               resources={'metadata': {'path': str(OUT)}}).execute(cwd=str(OUT))
nbformat.validate(nb)
nbformat.write(nb, OUT / 'SDY_Normalization_Reassessment.ipynb')


def inline(text):
    return re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html.escape(text))


def markdown(source):
    blocks = []
    for block in source.split('\n\n'):
        if block.startswith('#'):
            level = len(block)-len(block.lstrip('#'))
            assert 1 <= level <= 3 and '\n' not in block
            blocks.append(f'<h{level}>'+inline(block[level:].strip())+f'</h{level}>')
        elif block.startswith('- '):
            blocks.append('<ul>'+''.join('<li>'+inline(line[2:])+'</li>' for line in block.splitlines())+'</ul>')
        else:
            blocks.append('<p>'+inline(block)+'</p>')
    return '\n'.join(blocks)


parts = []
for cell in nb.cells:
    if cell.cell_type == 'markdown':
        parts.append(markdown(cell.source))
    else:
        for output in cell.outputs:
            assert output.output_type != 'error'
            if 'data' in output and 'text/html' in output.data:
                parts.append('<div class="table-wrap">'+output.data['text/html']+'</div>')
            elif output.output_type == 'stream':
                parts.append('<p class="check">'+html.escape(output.text)+'</p>')
style = 'body{font:16px/1.65 system-ui,sans-serif;color:#202020;max-width:1120px;margin:36px auto;padding:0 24px}h1{font-size:28px}h2{margin-top:32px;font-size:23px}h3{font-size:19px}table{border-collapse:collapse;font-size:14px;width:100%}th,td{text-align:left!important;padding:8px 10px;border-bottom:1px solid #ddd;vertical-align:top;overflow-wrap:anywhere}thead{background:#f1f1f1}.table-wrap{overflow-x:auto}li{margin:5px 0}.check{font-size:14px;color:#444}strong{font-weight:650}'
document = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>SDY2823归一化复评</title><style>'+style+'</style><body>'+'\n'.join(parts)+'</body></html>'
(OUT / 'SDY_Normalization_Reassessment.html').write_text(document)
print('Executed', sum(c.cell_type == 'code' for c in nb.cells), 'code cells; notebook and HTML saved')

"""Executed diagnostic notebook and author-review figure export; no HTML runtime."""
from pathlib import Path
import json
import sys


def make_figure(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    out=Path(out)
    data=pd.read_csv(out/'control_replicate_means.csv')
    summary=json.loads((out/'analysis_summary.json').read_text())
    parent=out.parent/'normalization_reassessment_20261013/ssd_rerun_v1'
    previous=pd.read_csv(parent/'three_question_descriptive_means.csv').set_index('method')
    families=['all_mapped','selected_measured']
    figure_data=data.loc[data.family.isin(families)].copy()
    figure_data['previously_reported_draw']=figure_data.family.eq('all_mapped')&figure_data.seed.eq(20261013)
    figure_data.to_csv(out/'figure_source_data.csv',index=False)
    contract={'question':'How variable is the same-cohort retrieval metric across fixed gene-shuffle draws?',
        'takeaway':'The previous single-draw result is high relative to the technical distribution; controls overlap the unmodified model and do not establish practical advantage.',
        'family':'Distribution','variant':'Box plot with all technical replicate points; whiskers show full range, not confidence limits',
        'surface':'Author-review research figure export and reproducible notebook companion; not an inline widget or HTML dashboard',
        'rows':62,'biological_donors':96,'judged_correlated_questions':3,'replicates_each':31,
        'palette_policy':'hard two-root cap','palette':{'replicate':'#3D79B6','prior_draw':'#C88635','ink':'#30343B','grid':'#DFE3E8'},
        'non_color':'Prior draw uses a diamond; benchmarks use different line styles and direct labels',
        'range':[0,1],'renderer':'Matplotlib','exports':['control_distribution.png','control_distribution.svg'],
        'branding':'Author-owned third-party manuscript research; no platform branding',
        'not_a_replacement_for_main_manuscript_figures':True}
    (out/'figure_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelcolor':'#30343B','text.color':'#30343B',
                         'xtick.color':'#30343B','ytick.color':'#30343B','axes.edgecolor':'#30343B','svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(9,6))
    fig.subplots_adjust(left=.11,right=.97,bottom=.22,top=.78)
    fig.text(.11,.945,'SDY2823 gene-shuffle controls',fontsize=17,weight='bold',ha='left')
    fig.text(.11,.895,'96 donors · 3 related questions · 31 technical draws per family',fontsize=11,ha='left')
    ax.set_ylim(0,1);ax.set_xlim(-.5,1.85)
    ax.set_yticks(np.linspace(0,1,6));ax.grid(axis='y',color='#DFE3E8',linewidth=.65)
    ax.set_axisbelow(True)
    for spine in ['top','right']:ax.spines[spine].set_visible(False)
    arrays=[figure_data.loc[figure_data.family.eq(f),'ndcg_at_k'].to_numpy() for f in families]
    ax.boxplot(arrays,positions=[0,1],widths=.34,whis=(0,100),showfliers=False,patch_artist=True,
        boxprops={'facecolor':'white','edgecolor':'#30343B','linewidth':.8},
        medianprops={'color':'#30343B','linewidth':1.1},whiskerprops={'color':'#30343B','linewidth':.8},
        capprops={'color':'#30343B','linewidth':.8})
    jitter=np.random.default_rng(20261013)
    for position,family in enumerate(families):
        block=figure_data.loc[figure_data.family.eq(family)].sort_values('seed')
        xs=position+jitter.uniform(-.115,.115,len(block))
        ax.scatter(xs,block.ndcg_at_k,s=23,c='#3D79B6',edgecolors='#30343B',linewidths=.3,alpha=.8,zorder=3)
        if family=='all_mapped':
            chosen=np.flatnonzero(block.previously_reported_draw.to_numpy())[0]
            value=float(block.iloc[chosen].ndcg_at_k)
            ax.scatter([xs[chosen]],[value],marker='D',s=65,c='#C88635',edgecolors='#30343B',linewidths=.7,zorder=5)
            ax.annotate(f'Previous single draw: {value:.3f}',xy=(xs[chosen],value),xytext=(-.39,.57),
                        fontsize=10,arrowprops={'arrowstyle':'-','color':'#30343B','linewidth':.7})
    baseline=summary['normal_Linux_ndcg'];epic=float(previous.loc['EPIC_BRef_raw','ndcg_at_k'])
    ax.axhline(baseline,color='#30343B',linewidth=1.1,linestyle='-',zorder=1)
    ax.axhline(epic,color='#30343B',linewidth=1.1,linestyle='--',zorder=1)
    ax.text(1.22,baseline+.016,f'RNA {baseline:.3f}',fontsize=10,ha='left')
    ax.text(1.22,epic+.016,f'EPIC {epic:.3f}',fontsize=10,ha='left')
    ax.set_xticks([0,1],['All mapped genes\n23,323 values','Selected measured genes\n3,524 values'])
    ax.set_ylabel('Mean nDCG@10')
    fig.text(.11,.085,'Points: technical draws. Boxes: median/IQR. Whiskers: minimum–maximum.\nThe same 96 donors are reused; no independent biological replicates are added.\nEPIC uses a fixed query-to-feature mapping. Frozen diagnostic v1 · 2026-10-13.',
             fontsize=9,ha='left',va='center',color='#555B63')
    fig.savefig(out/'control_distribution.png',dpi=220,facecolor='white')
    fig.savefig(out/'control_distribution.svg',facecolor='white')
    plt.close(fig)


def main():
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    out=Path(__file__).resolve().parent
    assert not (out/'FINAL_MANIFEST.json').exists()
    summary=json.loads((out/'analysis_summary.json').read_text())
    md,code=nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
    nb=nbformat.v4.new_notebook()
    nb.cells=[
        md('# SDY2823有限失败诊断\n\n## tl;dr\n\n'
           '2026-10-13，在原96位供者、3个有独立流式标签的问题及冻结模型上完成有界事后诊断，没有增加独立样本、重训或修改既有结果。\n\n'
           '原来0.411的单次全基因打乱结果不能代表打乱总体。两类31次技术重复的中位nDCG约为0.240和0.244，原始RNA为0.327；仍有10/31和9/31次达到或超过原始RNA。'
           '本队列RNA表示的约98%变化集中在第一主成分，T/B查询得分高度相关。它在旧CPM和恢复归一化下均存在，不是本次修复新造成的现象。\n\n'
           '**这些结果没有建立实用检索优势，也没有证明模型全局失效或定位唯一原因。** 原研究门槛仍未通过。'),
        md('## Context & Methods\n\n'
           'PROTOCOL.json在本轮运行前锁定两个打乱定义、20261013–20261043的31个种子、10,000次联合样本标签重排、几何和输入诊断。'
           '此前结果已被观察，因此这是事后诊断，不是预注册或确认性检验。\n\n'
           '### Key Assumptions\n\n'
           '- 全映射基因打乱严格重复原对照定义，保留23,323个值的多重集合，但可能改变投影后特征的值分布。\n'
           '- 另一个对照仅打乱3,524个实测模型特征，保留它们的值分布；572个缺失位置保持原零输入约定。两种定义不合并为同一个对照。\n'
           '- 先保存新向量和90个查询的得分，再读取评价标签。每题3个改写先取平均，然后对3题平均；62次控制不是62个独立生物学问题。\n'
           '- 所有种子和问题保留，不选最好或最坏的结果。只有Q01、Q02、Q19有本轮标签，其余27题没有获得独立有效性验证。\n'
           '- 样本标签重排对所有方法和查询采用相同的供者置换，保留每位供者的联合等级结构。参考分位数不是跨队列置信区间，参考比例不是一般性优越性P值。'),
        md('## Data\n\n### 1. 重读与独立核验\n\n'
           'input、权重、源代码和查询都由父运行及本轮合同固定。独立核验采用直接DCG求和、scikit-learn、直接重排及Gram矩阵特征值等不同计算路径。'),
        code("from pathlib import Path\nimport json\nimport numpy as np\nimport pandas as pd\nfrom verify_diagnostic import verify\nout=Path.cwd();study=out.parent\nchecks=verify(study,out)\nassert checks==json.loads((out/'independent_validation.json').read_text())\nsummary=json.loads((out/'analysis_summary.json').read_text())\nassert summary['Linux_Mac_top10_orders_all_equal']\nprint(f\"5,670组新得分重构误差 {checks['max_score_error']}; 567组指标最大误差 {checks['max_metric_error']:.3g}; 540组直接标签重排核验通过。\")\nprint('Linux与Mac原始RNA/原种子对照的18组top-10顺序全部一致；旧源文件未改变。')"),
        md('## Results\n\n### 2. 打乱结果有较大技术波动\n\n'
           '图中显示全部62个技术重复。箱体表示四分位范围，须线表示本轮最小/最大值，不是置信区间。菱形只标出此前已报告的那一个全映射基因打乱种子。'),
        code("from build_notebook import make_figure\nfrom IPython.display import Image,display\nmake_figure(out)\ndisplay(Image(filename=str(out/'control_distribution.png')))"),
        code("distribution=pd.read_csv(out/'control_distributions.csv')\ndistribution[['family','replicates','mean','median','min','max','replicates_at_least_unmodified']].round(3)"),
        md('### 3. 随机标签参考不能当作新的生物学证据\n\n'
           '固定当前候选池和相关性等级后，随机排序的解析期望为0.292824。原始RNA 0.327位于其条件参考范围内。'
           '下表仅列预先关心的代表方法；全部27方法保存在CSV中。比例为这10,000次重排的描述，不报告一般性检验P值；0表示本轮没有出现，不表示概率为0。'),
        code("reference=pd.read_csv(out/'sample_label_randomization_summary.csv').set_index('method')\nselected=['rna_direct_sf_reference','rna_fused_sf_reference','full_portrait_released_calibration_sf_reference','EPIC_BRef_raw']\npd.set_option('display.max_colwidth',None)\nreference.loc[selected,['observed_ndcg','analytic_random_mean','reference_p025','reference_p975','fraction_random_assignments_at_least_observed']].round(4)"),
        md('### 4. 本队列表示的变化近似一维\n\n'
           '这里的参与率维数由中心化表示协方差的特征值计算，不是模型的全局生物学维数。旧CPM和恢复归一化下均约为1.04，第一主成分占约98%。'
           '恢复归一化后的样本平均余弦相似度下降，说明输入变化确实影响了表示；不能把它说成所有样本完全相同，也不能仅凭几何确定训练失败原因。'),
        code("geometry=pd.read_csv(out/'embedding_geometry.csv').set_index('condition')\ngeometry.loc[['old_CPM_direct','rna_direct_sf_reference'],['pair_cosine_mean','centered_participation_dimension','first_PC_variance_fraction']].round(4)"),
        md('### 5. T/B查询排序缺少针对性，但不是所有文本查询投影都塌缩\n\n'
           'T与B查询的平均得分在本队列中高度相关：直接RNA约0.943，完整portrait约0.984，EPIC约0.395。'
           '另一方面，全部30题的改写内与题目质心间余弦均值差，在文本投影后并没有缩小。因此不能将现象简单归结为全部文本查询坍缩。'
           '余弦差只作描述性几何量，不是独立语义准确率。'),
        code("pairs=pd.read_csv(out/'cross_query_score_correlations.csv')\nshow=pairs.query(\"query_a=='Q01' and query_b=='Q02'\")\nshow=show.loc[show.method.isin(['rna_direct_sf_reference','full_portrait_released_calibration_sf_reference','EPIC_BRef_raw','cellwhisperer_v1'])]\ndisplay(show.round(4))\npd.read_csv(out/'query_geometry_summary.csv').round(4)"),
        md('原始RNA直接检索在每题三个改写之间的top-10 Jaccard均值约0.179。较高的查询向量余弦相似度并不保证返回相同候选。'
           '以下直接重读已冻结的父评价表，不新增问题，也不将传统方法固定查询映射所产生的稳定性当作自然语言理解能力。'),
        code("parent_means=pd.read_csv(study/'normalization_reassessment_20261013/ssd_rerun_v1/three_question_descriptive_means.csv').set_index('method')\nparent_means.loc[['rna_direct_sf_reference','rna_fused_sf_reference','full_portrait_released_calibration_sf_reference','cellwhisperer_v1'],['paraphrase_top10_jaccard']].round(4)"),
        md('### 6. 映射成功不等于每个特征都有非零表达\n\n'
           '未打乱样本中，已映射模型特征的零值比例中位数约73.9%；这与572个未映射位置是不同问题。两种归一化的这项比例相同。'
           '原始输入未出现±8裁切，因此本轮没有支持把现象主要归因于裁切。全映射打乱后的特征零值比例中位数约39.0%，仅实测特征打乱则保留约73.9%。'
           '这是设置第二类对照的原因，不能忽略两种干预的区别。两个处理中位表达值为0不意味着全部表达值不变。'),
        code("inputs=pd.read_csv(out/'control_input_diagnostics.csv')\nassert checks['preprocessing_upper_clip_max']==checks['preprocessing_lower_clip_max']==0\ninputs.groupby('family')[['zero_fraction_measured','upper_clip_fraction_measured','lower_clip_fraction_measured']].median().round(5)"),
        md('### 7. 预设输入摘要的相关性\n\n'
           '表格给出原始RNA得分与五个固定输入摘要的Spearman相关，各题取三个改写的中位数。没有拟合解释模型、选择最有利变量或声称因果。'
           '全部27个方法和每种改写的值保存在score_input_correlations.csv；常数变量/得分标为空值，不填0。'),
        code("associations=pd.read_csv(out/'score_input_correlations.csv')\nassociations.loc[associations.method.eq('rna_direct_sf_reference')].groupby(['input_summary','query_id']).spearman.median().unstack().round(4)"),
        md('## Takeaways\n\n'
           '- 修正的是对单次打乱结果的解释：0.411不是稳定的打乱优势。原始RNA高于两类控制的中位数，但存在较多重叠，不能据此宣称强表达依赖或实用优势。\n'
           '- 本血液队列的RNA表示低有效维度、T/B查询得分高度相似，是可追溯的观察，不是模型全局失效的证明；唯一原因尚未定位。\n'
           '- 传统EPIC在相同三题上的相关性更高，但它使用固定查询—特征映射，不是无限制自然语言解析器。人工映射和核查成本仍未得到量化。\n'
           '- 当前结果不支持扩大确认性或湿实验投入。还需完成原计划中已有CPTAC候选池的正确尺度复评，保留SDY阴性结果，不以选择数据集或重训来掩盖它。\n'
           '- 没有新增独立问题/供者、人工语句评分、机制发现或稿件改写。本笔记与图是诊断记录，不替代原计划中的主文科学图件。\n\n'
           '可作为有明确边界的事后诊断分享；完整Cell计划尚未完成。')
    ]
    nb.metadata={'kernelspec':{'name':'python3','language':'python','display_name':'Lab diagnostic Python'}}
    manager=KernelManager(kernel_name='python3')
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    NotebookClient(nb,timeout=240,kernel_name='python3',km=manager,resources={'metadata':{'path':str(out)}}).execute(cwd=str(out))
    nbformat.validate(nb)
    nbformat.write(nb,out/'SDY_Failure_Diagnostic.ipynb')
    print('Executed',sum(c.cell_type=='code' for c in nb.cells),'code cells; notebook and static figure exported')


if __name__=='__main__':main()

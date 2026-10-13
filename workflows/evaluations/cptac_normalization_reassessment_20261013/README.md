# CPTAC 固定候选池归一化复评

3,768 次完整推断、32 个新旧条件和 37,440 组指标的独立验算已完成。归一化修复提高了 pooled RNA 得分，但没有证明相对于强基线的实用增益；Cell 研究推进门槛未通过。原稿未改写，没有启动新训练、合并 main 或投稿。

## 先看结果

- [完整报告及解释边界](report_v2/REPORT.md)
- [已执行的六单元 notebook](report_v2/CPTAC_Normalization_Reassessment.ipynb)
- [全部 32 个条件的主分析](report_v2/primary_all_conditions.csv)
- [全部 13 个上下文、37,440 组指标](evaluation_v1/per_query_paraphrase_results.csv)
- [独立验算记录](evaluation_v1/independent_validation.json)
- [输入准备核对](Reassessment_Preparation.ipynb)与[数值预检查](preflight_result.json)
- [原 anchor 主干 sidecar 恢复及交叉核验](../anchor_training_provenance_20261013/README.md)

主尺度完整 portrait 的 conditional nDCG@10 为 0.2175，旧 CPM 为 0.1695；固定比较为 CellWhisperer 0.3070、传统组合 0.6310、marker-rank 0.6546。全部旧结果、敏感性条件、输入破坏对照及不利结果均保留。

## 固定了什么

同一组 942 个病例、30 个问题和每题 3 个改写；不以改写充当独立问题。原映射保留 23,628 个参考基因、3,502 个选定模型基因，594 个选定特征仍缺失。主分析按参考基因集合拟合 size factor，敏感性分析按全部 59,808 个实测基因拟合；均在同一 942 个候选文库联合拟合，不是训练集冻结的单样本部署接口。

原始内部协议的去标识公开副本见 [PROTOCOL.public.json](PROTOCOL.public.json)。它明确省去含病例文件名的历史清单，保留完整原件哈希，不冒充内部原件。

## 如何复核

公开 notebook 可基于同目录汇总文件执行。完整计算仍需要保留的 study 工作区、原始 RNA/蛋白材料与配对记录、冻结 rc4 完整运行时、模型权重及训练参考矩阵，以及此前记录的 Linux CPU 环境。公开 compact API 本身不足以运行完整解释器；本次没有上传这些私有输入或权重，也不声称已完成独立第三方全流程发布。

本次入口按顺序为 `run_job.py`（输入准备和冻结推断）、`run_evaluation_job.py`（真实结局评估及独立复算）、`build_reassessment_report.py`（汇总、静态图及 notebook）。每个命令都要求显式 `--study-root` 和输出根目录；评估及报告脚本使用 `--reassessment-root`。原始运行命令和参数记录在 SSD 的 job state 与日志中。已有输出目录会拒绝覆盖，不能原地重启已完成步骤。

`reference/` 保留未改动的指标和传统基线定义，便于核对。传统结构化方法接收预先定义的查询—面板映射，其改写 Jaccard=1 并不代表自由文本理解能力；相同问题和样本不意味着所有方法具有相同的查询输入接口。蛋白面板只是分子一致性参照，不是功能或人类语句评分。

## 当前决策点

499 个未关联病例不能称为已证明无暴露；443 个已知 test 病例也不是新的独立外部队列。原 anchor 主干 sidecar 已交叉核验，不等于所有后续 heads/priors/规则都完成训练来源核查。完整报告保留了这些限制以及未判断候选的覆盖率。

根据原计划，应先由作者选择新的方法/验证路线，或调整文章定位和目标期刊。现有证据不支持直接继续 Cell-promotional 改写。人工盲评、确认性增益、新生物学发现、最终六个主图和 Cell 投稿附件仍未完成。

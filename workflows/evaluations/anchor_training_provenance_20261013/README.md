# 原 anchor 主干的训练划分证据恢复

原训练运行目录中的 sidecar 样本划分表已找到，并通过权重、输入标准化和保存预测的交叉核对。此前“只有当前运行时重建划分、缺少该 anchor 运行的原始 sidecar 表”的限制因此得到补充；不能把这一进展外推为整个解释器所有组件均已证明训练独立。

## 核对结果

| 检查 | 结果 |
| --- | --- |
| 原 checkpoint 与当前冻结 anchor 权重 | SHA-256 完全一致 |
| 原 sidecar 样本 / 来源队列 | 25,199 / 117 |
| train / validation / test | 13,107 / 8,205 / 3,887 |
| 与此前已核验 runtime/semantic 划分的逐样本差异 | 0 |
| 训练表达 mean / std | 各 4,096 项，全部逐值精确一致 |
| 训练年龄 mean / std | 均精确一致 |
| 25,199 条保存年龄预测的最大数值差 | 1.15×10⁻⁵ 年，低于预先固定的 10⁻³ 容差 |
| 25,199 条保存肿瘤概率的最大数值差 | 8.93×10⁻⁸，低于预先固定的 10⁻⁵ 容差 |

预测复现检查的是文件和模型行为的对应关系，不是年龄、肿瘤状态或任何生物学结论的准确率。预先固定的数值容差没有根据结果扩大。原始文件、当前权重、运行中的 CPTAC 任务和候选池均未改动。

## 证据与解释范围

1. [运行前协议](PROTOCOL.json)绑定原 checkpoint、训练脚本、sidecar 表、训练历史和输入矩阵的哈希。checkpoint 与此前恢复的实际 anchor 主干哈希一致，模型类 AST 与冻结便携类相同。
2. 根据原 sidecar 的 train 样本顺序，从原表达矩阵独立重算标准化参数，全部 8,192 个表达参数精确一致。这比仅凭相同目录名或文件时间判断来源更有约束力。
3. 用该 checkpoint 对全部 25,199 条原始记录重放两个保存的预测字段，均通过数值核对。代码见 [audit_anchor_training_provenance.py](audit_anchor_training_provenance.py)，汇总见 [RESULT.json](RESULT.json)。
4. 使用原有病例交叉表关联现有 CPTAC 候选，仍为 443 个已知 test 病例和 499 个未关联病例；未新增已关联 train/validation 病例，也没有据此删除或替换候选。499 个未关联不能解释为证明无训练或患者暴露。

该 checkpoint 本身仍未内置 split assignments 或 seed。本轮恢复的是与权重、输入参数和输出记录相互支持的归档 sidecar，不是把新拟合的划分追写到旧 checkpoint。其他后续预测头、priors、规则开发和患者级重叠的来源范围须另行核查。

## 查看与复算

[汇总 notebook](Anchor_Training_Provenance.ipynb)的三个代码单元已执行。它检验并展示已保存的汇总记录；全部样本级数值重放由下列驱动执行。原 sidecar、元数据、checkpoint 副本和逐样本核对表保存在 SSD 的私有研究目录，未上传 GitHub。

复算依赖原 `RNAMultimodel` 源目录、保留的 study 工作区和 PR #7 对应的 Linux CPU 环境。以下是本次实际执行命令；输出目录已存在，因此不要直接重跑覆盖，如需重复，应指定一个新的输出目录。

```sh
/home/aroluo/.venvs/rna-portrait-cell-cpu-py312-20261013/bin/python \
  /media/aroluo/ExtremePro/Downloads/nature_manuscript_draft_20260609/research/cell_revision_20261008/cptac_reassessment_20261013_drivers_v1/audit_anchor_training_provenance.py \
  --study-root /media/aroluo/ExtremePro/Downloads/nature_manuscript_draft_20260609/research/cell_revision_20261008 \
  --source-root /media/aroluo/ExtremePro/RNAMultimodel \
  --output-root /media/aroluo/ExtremePro/Downloads/nature_manuscript_draft_20260609/research/cell_revision_20261008/anchor_training_provenance_20261013_v1
```

## 后续稿件处理

在下次有版本号的 STAR Methods 修订中，区分“checkpoint 未内置划分”和“已恢复并交叉核验的原运行 sidecar”，不再笼统写成整个 anchor 主干的原划分完全缺失。历史审计报告原文保留，本记录补充其后出现的证据。正文尚未在本次核查中改写；生物学效能、人工语句盲评和 Cell 实用价值验收也没有因此完成。

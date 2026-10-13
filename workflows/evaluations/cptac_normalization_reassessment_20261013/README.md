# CPTAC 固定样本池归一化重算

本次发布完成的是输入核查、归一化准备和执行协议，不是模型效果验收。冻结模型计算已在 lab 启动；比较结果须等全部推断和独立验证完成后再报告。没有重训、改写论文结论或合并 main。

## 已核实的输入

同一组 942 个病例、30 个问题及每题 3 个改写保持不变。942 个原始计数列的哈希均与旧 QC 记录一致，旧 CPM 模型输入可逐值精确重建。保持原映射后，有 23,628 个参考基因和 3,502 个选定模型基因可测；594 个模型特征仍缺失，不视作生物学零表达。

| 条件 | 拟合基因数 | 所有病例均为正的参考基因数 | size factor 中位数 | 独立实现最大绝对误差 |
| --- | ---: | ---: | ---: | ---: |
| sf_reference（主分析） | 23,628 | 13,402 | 1.076751 | 1.87×10⁻¹⁴ |
| sf_full_measured（敏感性分析） | 59,808 | 14,370 | 1.080050 | 1.91×10⁻¹⁴ |

两种条件均在同一组 942 个候选 RNA 文库上联合拟合，并使用恢复出的 `log1p(count / size_factor)` 公式。主分析沿用旧参考基因集合，敏感性分析改变拟合基因集合；不根据检索效果选择条件。由全候选池拟合得到的尺度属于传导式预处理，不是训练集预先冻结的单样本部署接口，也不声称完全重建了原 GDC/GEO 队列或注释版本。

## 推断与指标预检查

新增的[预检查结果](preflight_result.json)独立重建了 10 个条件下的 9,420 个 compact RNA 向量。与单样本接口保存值的最大绝对差为 1.64×10⁻⁷，低于运行前固定的 10⁻⁶ 容差；查询投影最大差为 2.24×10⁻⁷。全部 942 个打乱对照保持可测特征值集合，缺失特征位置未改动，全局和队列内错配与旧条件的病例分配完全一致。

生产指标函数与独立公式在 640 组合成边界情形上核对，通过 4,679 项未定义值检查及 6 项无效输入拒绝检查，最大数值差 3.34×10⁻¹⁶。合成情形涵盖不足 10 个候选、全缺失证据、无相关样本、并列分数及拒答值；它们不是新增生物学重复，也不能替代真实 CPTAC 结果的完整验收。脚本和[运行前协议](preflight_protocol.json)可复核，检查期间没有读取蛋白 qrel 数值或改动运行中的完整解释器。

```sh
"$CPTAC_PY" "$CPTAC_DRIVERS/preflight_controls_and_metrics.py" --study-root "$CPTAC_STUDY" --reassessment-root "$CPTAC_RUN" --drivers-root "$CPTAC_DRIVERS"
```

上述变量在下文定义；命令需要已写完的 compact RNA 向量文件，拒绝覆盖现有 `preflight_v1`。当前运行已经通过，不应重复执行。

## 证据、判断与下一步

| 证据 | 支持的判断 | 后续动作 |
| --- | --- | --- |
| 原始列哈希、旧输入精确重建 | 复评没有偷偷更换病例或原始 RNA 输入 | 保持同一候选池进行模型比较 |
| 恢复的归一化公式、独立 size-factor 核对 | 新输入遵循已恢复的数值变换规则 | 检验模型表现；不能把数值通过当作生物学通过 |
| [公开协议](PROTOCOL.public.json)和原始协议哈希 | 已在新输入变换之前固定条件 | 完成全部 3,768 次完整 portrait 推断，并保留失败记录 |
| 既有蛋白 qrels 与 18 种旧条件保持不变 | 能按同一任务追加 14 种新条件 | 在 13 个上下文核查 37,440 组指标和 12,480 组改写稳定性 |

新条件包括两种尺度下的 direct RNA、fused RNA、开启/关闭历史校准的完整 portrait，以及主尺度下的选定基因值打乱、全局/队列内错配和 10/30/50% 特征屏蔽。屏蔽发生在固定归一化之后，不与旧版“删除原始计数并重算 CPM 分母”混称。单次随机对照仅作描述，不能据此判定稳定的打乱优势。

## 查看与复算

[输入核查 notebook](Reassessment_Preparation.ipynb)已执行，展示公开汇总的核对结果；它不替代私有工作区中的原始计数重算。

完整运行需要此前保留的 study 工作区、原始 Broad 输入与配对记录、冻结 rc4 完整运行时、训练参考矩阵，以及 PR #7 所记录的 Linux CPU 环境。公开 compact API 本身不足以运行完整解释器。脚本用显式路径，不包含权重或样本级数据；其输入缺失时会停止，不补造数据。

下列命令在 lab 的现有工作区运行，重复研究须给 `CPTAC_RUN` 指定一个新的输出目录；不要对已完成目录重新运行准备步骤：

```sh
CPTAC_STUDY=/media/aroluo/ExtremePro/Downloads/nature_manuscript_draft_20260609/research/cell_revision_20261008
CPTAC_PY=/home/aroluo/.venvs/rna-portrait-cell-cpu-py312-20261013/bin/python
CPTAC_RUN="$CPTAC_STUDY/cptac_normalization_reassessment_20261013_v1"
CPTAC_DRIVERS="$CPTAC_STUDY/cptac_reassessment_20261013_drivers_v1"
```

当前任务的启动命令是：

```sh
"$CPTAC_PY" -u "$CPTAC_DRIVERS/run_job.py" --study-root "$CPTAC_STUDY" --output-root "$CPTAC_RUN"
```

`run_job.py` 先准备输入，再运行冻结模型。任务已有状态时拒绝重复启动。单例锁、源码/权重校验、逐样本输入哈希和失败记录防止静默覆盖或带着变更续跑。状态及分阶段日志位于 drivers 目录，完整推断还会写入 `inference_v1/progress.json`。这些日志和单病例结果不上传 GitHub。

完整推断完成后，才可运行下列已写好但在本次“输入准备”发布时尚未进行端到端验收的评估：

```sh
"$CPTAC_PY" "$CPTAC_DRIVERS/evaluate_cptac_retrieval.py" --study-root "$CPTAC_STUDY" --reassessment-root "$CPTAC_RUN"
"$CPTAC_PY" "$CPTAC_DRIVERS/verify_cptac_retrieval.py" --study-root "$CPTAC_STUDY" --reassessment-root "$CPTAC_RUN"
```

评估使用同一问题下各方法共有的可测蛋白候选池，同时报告全池 top-10 的已判断比例。缺失蛋白证据不补成阴性；两个归一化条件和所有旧结果均保留。独立验证器不导入生产指标函数，并重建 tie-aware nDCG、汇总和保存的相似度矩阵。

## 限制和公开范围

443 个病例属于既有 alignment test，499 个没有在目前的交叉表中找到记录；后者不等于已经证明训练无暴露。蛋白面板是分子丰度证据，不是细胞数量、功能、机制或真实评价者逐句评分。原 anchor 训练成员追溯、独立人工盲评和确认性研究仍是另行未完成事项。

这是见过旧 CPTAC 和 SDY 结果后的探索性修正，不是预注册。原稿、所有旧输出和负面 SDY 结果保持不变。Cell 的实用价值门槛尚未通过。

`PROTOCOL.public.json` 明确省去了含病例文件名的历史文件清单，并保存完整内部协议的哈希、清单条目数和清单摘要；不把删减副本冒充内部原件。公开材料不包含原始计数、样本级预测/嵌入、权重或凭据。

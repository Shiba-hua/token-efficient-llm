# 数据、预训练与中训练章节的来源和验证记录

核查日期：2026-09-09。本文仅记录公开来源和教程验证，不包含个人设备、工作经历或非公开实验。访问网页与运行教学脚本不代表复现论文训练结果。

## 主源与支持范围

| ID | 主源 | 本次阅读范围 | 支持的内容与边界 |
|---|---|---|---|
| FD01 | [OpenMathReasoning 官方数据卡](https://huggingface.co/datasets/nvidia/OpenMathReasoning) | 正文、字段、数量更正 | 题目模型整理、生成轨迹、抽取答案与多数投票答案的区分。卡片已更正流水线起点和发布带解答题数。未下载全库重数，未审计真实答案正确率。 |
| FD02 | [OpenCodeReasoning 官方数据卡](https://huggingface.co/datasets/nvidia/OpenCodeReasoning/blob/main/README.md) | 字段与来源说明 | `output` 与 `solution` 区别；v1 发布字段未附齐测试。版本限定为 v1，不将后续版本混用。 |
| FD03 | [OpenCodeReasoning 论文 v1](https://arxiv.org/html/2504.01943v1#S4.SS1) | §2、§4.1 执行过滤消融 | 生成代码有错；全量、通过测试子集、失败子集的下游实验。其难度与规模变化不等于“错误标签有益”的普遍因果证据。 |
| FD04 | [MegaScience 论文 v1](https://arxiv.org/html/2507.16812v1#S2) | §2.1–2.5、§3.3–3.4、§5.4 | TextbookReasoning 抽取、改写、补充推理、过滤、去污染。全量与额外选择子集的下游结果不是人工正确率审计。未发现三库同标准质量排名。 |
| FD05 | [TextbookReasoning 官方卡](https://huggingface.co/datasets/MegaScience/TextbookReasoning) | 流程、质量说明、发布字段 | 教材来源与模型整理；发布字段无法逐条直接恢复原书页码。数据卡所谓质量说明主要为下游训练表现。 |
| FP01 | [FineWeb 论文 v1](https://arxiv.org/html/2406.17557v1) | §3 抽取/过滤/去重，§4 教育子集 | 过滤与去重的控制变量实验；部分全局去重方案没有优于逐 crawl 去重；教育子集收益限定于作者实验。没有把它写成部署 token 收益。 |
| FP02 | [DCLM 论文 v1](https://arxiv.org/html/2406.11794v1)、[项目](https://www.datacomp.ai/dclm/) | 摘要、问题定义、实验结构与数据策略 | 固定训练设置比较数据筛选/混合；模型过滤是基线方法，不是行业统一最优配方。 |
| FP03 | [SentencePiece 官方仓库](https://github.com/google/sentencepiece) | README 工具说明 | BPE、unigram 与真实 tokenizer 实现入口。教程代码是简化字符 BPE，未伪称兼容生产 tokenizer。 |
| FP04 | [MTP 论文 v1](https://arxiv.org/html/2404.19737v1) | §2、Figure 1 图与图注、摘要的解码说明 | 共享主干与独立未来位置头；样本效率/任务分数与解码速度的不同口径。原图由集成维护者提供，本章已实际打开核对内容。 |
| FM01 | [Don’t Stop Pretraining，ACL 2020](https://aclanthology.org/2020.acl-main.740/)、[正式 PDF](https://aclanthology.org/2020.acl-main.740.pdf) | 出版页与 PDF §1–3；方法名称及模型范围 | DAPT/TAPT 的领域、任务适配证据。PDF §2 明确 RoBERTa 的掩码语言建模目标，不能直接当作所有自回归模型保证。 |
| FM02 | [SmolLM3 官方技术文](https://huggingface.co/blog/smollm3#mid-training) | Mid-training、long context、推理阶段、merging 说明 | 阶段命名由作者定义；长上下文适配；后续推理阶段的长上下文退化观察。单项案例不是必然遗忘定律。 |
| FM03 | [Effective Long-Context Scaling，arXiv:2309.16039](https://arxiv.org/abs/2309.16039) | 作者摘要 | 仅列为长上下文扩展代表阅读入口，没有从未读取的结果表提取数值。 |

## 自行推导与设计

三章中的完整任务 token 记账、全概率判分误差分解、NTP 交叉熵分解、简化 MTP 加权损失、领域混合目标、KL 约束示意和一步梯度 Taylor 展开是教学推导。各式在正文写明条件。它们不是对真实数据错误率、LLM 训练收益或商业成本的测量。

`assets/plots/data-four-axes.svg`、`pretraining-objectives.svg`、`pretraining-midtraining.svg` 为本教程自制概念图；无论文截图、无真实实验数字。`assets/papers/mtp-main.png` 为作者原图，归属、许可及字节来源见同目录 `provenance.json`，本任务未修改该记录。

## CPU 验证

以下程序使用标准库，不联网，不执行外部数据中的代码，也不训练神经网络：

```bash
python3 examples/data_audit.py
python3 examples/tokenization_lab.py --mode all
```

- 数据审核脚本：8 条合成记录、4 个已知题族；完全题面重叠 2 条，同族重叠 3 条；排除后 5 条候选，未称为全维度干净数据。
- 合成调用记账：一次调用平均总成本 170；重试方案平均总成本 285，后者的最终输出更短。没有生成任务得分。
- BPE：只在合成训练文本学习 merge，检查解码可逆；保持原文，比较字符数、UTF-8 byte 数与编码长度。
- NTP：平滑二元模型的条件概率归一化检查通过。
- 适配：固定词表与加权训练总量，测试句子与训练序列不完全相同，扫描领域混合权重；这是重新拟合计数，不是连续神经网络训练。

未执行大模型预训练、中训练、数据全量去污或用户模型实验；教程对此有明确标注。

图与文档检查：三个自绘 SVG 均通过 XML 解析，使用 `rsvg-convert` 渲染并逐图打开检查，未见文字裁切或重叠；原论文图已打开核对图注和上下两个部分。三章本地相对链接均找到目标文件，数学块配对检查通过；采用 GitHub 支持的 `$` / `$$` 数学标记。最终页面级渲染由整合流程统一进行。

## 2026-09-09 补充核查

数据章 §3.1 已纳入 [OMR 官方讨论](https://github.com/NVIDIA-NeMo/Skills/issues/1092)、[Nemotron-RL-Math-v2](https://huggingface.co/datasets/nvidia/Nemotron-RL-Math-v2)、[OCR-2 执行标注](https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2)，并复算 TBR 官方 train 的 row_idx 2、3。两个反例只是存在性证据，不估计总体错误率。TBR 的简短答案由详细答案模型抽取，其字段一致不是独立验证。

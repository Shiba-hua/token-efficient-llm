# 视觉验收记录

> 此页是第一版的历史验收记录。第二版结果见[研究地图验收](research-map-audit.md)，不能用本页 PASS 代替本版验收。

2026-09-09：**PASS**。全部 **19 / 19** 个逻辑图位（18 张独立资产）通过，0 项待人工判断；Harness 图经过两轮修复与独立复审，历史阻断缺陷均已关闭。

本轮依据 Visual Inspection 2.0.0 执行。每个图位均交给一名新的独立 `gpt-5.6-luna` 审计员，推理强度为 `medium`。数据章节新增质量核查段落后独立复审一次，Harness 图经过两轮修复后分别独立复审，合计 22 条检查记录。重复使用的首页概念图按两个正文位置分别检查。

检查范围是完整图片、SVG 渲染图、Markdown 全文、图注及邻近解释，覆盖裁切、重叠、字形、可读性、指标与单位、曲线方向、图文一致性和合成数据标注。**本记录不代表 GitHub 整页渲染或不同屏幕宽度的布局验收。**

显式要求优先于基于教程上下文的推断与通用默认规则。概念前沿图按作者提供的内置图像生成记录及提示词核对；其他资产为作者代码绘图或论文 LaTeX 源包的独立图像资产。论文原图允许保留英文，正文表格采用原生 Markdown。图像归属与派生记录见[资产说明](../assets/README.md)和[论文资产来源](../assets/papers/provenance.json)。

## 图位覆盖

| 文档与图位 | 资产 | 结果 | SHA256 前 12 位 |
| --- | --- | --- | --- |
| [README.md 第 7 行](../README.md) | [frontier-concept.png](../assets/generated/frontier-concept.png) | PASS | `2bb755339bf3` |
| [docs/00-overview.md 第 11 行](../docs/00-overview.md) | [frontier-concept.png](../assets/generated/frontier-concept.png) | PASS | `2bb755339bf3` |
| [docs/00-overview.md 第 53 行](../docs/00-overview.md) | [research-map.svg](../assets/plots/research-map.svg) | PASS | `79083f41ab29` |
| [docs/01-data.md 第 41 行](../docs/01-data.md) | [data-four-axes.svg](../assets/plots/data-four-axes.svg) | PASS | `ca9a3836362a` |
| [docs/02-pretraining.md 第 17 行](../docs/02-pretraining.md) | [pretraining-objectives.svg](../assets/plots/pretraining-objectives.svg) | PASS | `7d0d963e85a8` |
| [docs/02-pretraining.md 第 101 行](../docs/02-pretraining.md) | [mtp-main.png](../assets/papers/mtp-main.png) | PASS | `5fc3a6b36df6` |
| [docs/03-midtraining.md 第 29 行](../docs/03-midtraining.md) | [pretraining-midtraining.svg](../assets/plots/pretraining-midtraining.svg) | PASS | `5a3c88a21cf2` |
| [docs/04-sft-distillation.md 第 67 行](../docs/04-sft-distillation.md) | [posttraining-kl.svg](../assets/plots/posttraining-kl.svg) | PASS | `628d15c61c4c` |
| [docs/05-preference-rl.md 第 117 行](../docs/05-preference-rl.md) | [posttraining-clip.svg](../assets/plots/posttraining-clip.svg) | PASS | `37e6e47b34a2` |
| [docs/06-rlvr-agentic.md 第 74 行](../docs/06-rlvr-agentic.md) | [posttraining-grpo.svg](../assets/plots/posttraining-grpo.svg) | PASS | `8a91f40af05f` |
| [docs/07-prompt-context.md 第 15 行](../docs/07-prompt-context.md) | [context-pipeline.svg](../assets/plots/context-pipeline.svg) | PASS | `47bf27044c3d` |
| [docs/08-harness-agents.md 第 25 行](../docs/08-harness-agents.md) | [harness-loop.svg](../assets/plots/harness-loop.svg) | PASS | `31d6a9e06ec4` |
| [docs/09-evaluation.md 第 23 行](../docs/09-evaluation.md) | [evaluation-frontier.png](../assets/plots/evaluation-frontier.png) | PASS | `18ab176da675` |
| [docs/09-evaluation.md 第 50 行](../docs/09-evaluation.md) | [evaluation-ock-triangle.png](../assets/plots/evaluation-ock-triangle.png) | PASS | `cf71d5f93a19` |
| [docs/09-evaluation.md 第 85 行](../docs/09-evaluation.md) | [evaluation-otb-maze.png](../assets/plots/evaluation-otb-maze.png) | PASS | `3c02f34f9f6f` |
| [docs/09-evaluation.md 第 104 行](../docs/09-evaluation.md) | [evaluation-aucoaa.png](../assets/plots/evaluation-aucoaa.png) | PASS | `a3226150de45` |
| [docs/09-evaluation.md 第 134 行](../docs/09-evaluation.md) | [evaluation-think-ratio.png](../assets/plots/evaluation-think-ratio.png) | PASS | `a11d23a1d113` |
| [docs/09-evaluation.md 第 144 行](../docs/09-evaluation.md) | [appworld-main.png](../assets/papers/appworld-main.png) | PASS | `57f4fd714451` |
| [docs/10-other-directions.md 第 9 行](../docs/10-other-directions.md) | [harness-resource-metrics.svg](../assets/plots/harness-resource-metrics.svg) | PASS | `8994ce145301` |

无图文档同样纳入发现范围：`docs/efficient-reasoning-recipes.md`、`docs/glossary.md`。发现清单与审计任务、审计结果逐一匹配，没有缺失结果、非法结果、模型不符或未解决的要求冲突。

数据章节的新段落复审通过。概述章新增流程图另行审查；原概念图的图像、图注及邻近说明未变，保留其既有结果。后续新增专题导航和 THINK 数据链接固定版本不改变既有图的含义，已对照正文差异。

数学兼容性排版改动已逐行核对：图片哈希、19 处引用、图注和行号均未变，正文差异仅涉及声明的数学标记转换及空格，因此沿用既有图审 PASS；此项上下文核对不替代数学等价性检查或 GitHub 公式原生渲染验收。

## 修复复审

初审发现 [Harness 运行闭环](../assets/plots/harness-loop.svg) 缺少正常最终回答出口。第一次修复补充了该出口，但独立复审仍发现终止状态缺少条件映射，以及调用前预算检查只出现在底注。

第二次修复把预算门控置于上下文与模型之间，并用带标签的条件箭头分别连接正常结束、动作被拒绝和预算耗尽。正常结束仅表示已返回最终回复；任务是否成功由评测判定。图内说明额外收尾模型调用使用预留额度并再次接受预算检查。

第二轮修复已由新的独立审计员复审通过。当前 SHA256 为 `31d6a9e06ec46dba5a8f52ef957a23999d97403ed53f94cdd129d9d80182f68a`。原始版本与第一轮修复版本的两次 FAIL 结果及各自修复交接均保留在本地历史台账，没有覆盖。当前没有待修复的阻断缺陷。

原始图片审查上下文、完整 SHA256、逐条结果及验证输出保留在本地审计目录；公开记录仅保留以上摘要，不包含私人路径或审计任务载荷。

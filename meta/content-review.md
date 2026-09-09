# 正文与教学程序审读记录

> 此页是第一版的历史验收记录。第二版结果见[研究地图验收](research-map-audit.md)，不能用本页 PASS 代替本版验收。

审读日期：2026-09-09。采用模型辅助交叉审读与作者复算；不表示原论文训练结果已复现。

## 覆盖范围

| 正文 | 核对内容 | 直接证据 |
|---|---|---|
| [总论](../docs/00-overview.md) | 任务分布、实际成本与预算参数、Pareto 条件；共识与典型方法的证据边界 | 完整输入输出算例、跨阶段地图、独立配方引用 |
| [数据](../docs/01-data.md)、[预训练](../docs/02-pretraining.md)、[中训练](../docs/03-midtraining.md) | 数据来源与判分范围、NTP/MTP 目标、交叉熵分解、混合目标与遗忘 | [来源记录](foundations-sources.md)、两个 CPU 教学程序 |
| [SFT](../docs/04-sft-distillation.md)、[偏好](../docs/05-preference-rl.md)、[RLVR](../docs/06-rlvr-agentic.md) | SFT 平均方式、正反 KL、OPD 状态分布、DPO、PPO/GRPO 与 Agent 策略梯度 | [来源记录](posttraining-sources.md)、解析梯度与有限差分检查 |
| [上下文](../docs/07-prompt-context.md)、[编排](../docs/08-harness-agents.md)、[相邻方向](../docs/10-other-directions.md) | 压缩调用成本、信息丢失、通信账、路由分布、推测采样校正与内部计算 | [来源记录](context-sources.md)、两个 CPU 教学程序 |
| [评测](../docs/09-evaluation.md) | 四基准各三个真实例子；公式与实现差异；失败分母、成对记录、场景完整性 | 固定数据/代码链接、样例索引、BFS 复算、评分单元测试 |

## 审读后已落实的实质修订

1. PPO/GRPO 的 clip 限制替代目标在特定方向上的收益，不是概率比的硬约束。
2. 固定状态的 KL 对 logits 梯度显式限定教师固定、温度为 1。
3. 逐条正确奖励高于错误奖励，不能保证策略优化时成功率优先。正文加入 0.8 与 0.882 的期望奖励反例。
4. Agent 策略梯度示意明确采用未折扣 episode 回报，避免遗漏折扣权重。
5. 数据来源可靠性、答案正确性、推理可信度、判分可靠性与去污染分别报告；补入可复算的 TBR 错例。
6. OTB 的发布数量和 AUCOAA 聚合顺序、OckBench 的输出/总 token 口径及 judge 差异均明确限定版本。
7. 原 OTB metadata 对目标符号的双重转义未用于绘图；迷宫按真实题面网格解析，BFS 为 15 步。

## 验证及局限

七个教学程序已实际执行，输出见 [example-runs.json](example-runs.json)。评分的六项单元测试和后训练算术自检通过。它们检查公式、计量和合成场景，不证明真实 LLM 的性能改善，也不代替全部事实与视觉审读。

图像验收、GitHub 页面显示与远端 CI 的最终状态分别见 [交付核查](completion-audit.md)及独立视觉记录。正文使用原生 Markdown 表格，不以截图代替表格。

补充专题 [直接控制推理长度的三条路线](../docs/efficient-reasoning-recipes.md)连接 TokenSkip、长度奖励 RL 与 s1 budget forcing。新增标准库算例已运行；条件 SFT、sigmoid 长度奖励和留一基线均与其定义及适用范围一起说明。

GitHub 实际显示检查后，统一保护数学字符串，修复中文标点邻接、Markdown 转义与宏兼容问题。独立源码比较确认 358 个行内公式、70 个块公式数学等价，19 个非数学代码围栏逐字一致；浏览器复查 428 个公式全部原生渲染。详见 [发布与显示核查](github-render-check.md)。

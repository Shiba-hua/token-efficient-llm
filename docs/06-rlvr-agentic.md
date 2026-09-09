# 06 RLVR 与 Agentic-RL：把省 token 变成可学习的决策

[返回目录](../README.md) · [上一章：偏好优化](05-preference-rl.md) · [下一章：Prompt 与上下文](07-prompt-context.md)

这一领域最重要的分界是：**奖励正确答案、奖励经济地答对、奖励在环境里经济地完成任务，属于三个不同问题。** GRPO/PPO 等优化器负责更新策略；是否省 token 取决于奖励、轨迹、数据和预算控制，不能从优化器名字直接判断。

## 1. 正确性、长度、预算和工具的路线地图

| 研究问题 | 代表路线 | 关键机制 | 竞争或组合关系 |
| --- | --- | --- | --- |
| 怎样学会更强推理？ | DeepSeekMath/GRPO、OpenAI o1 的 RL | 用可检查结果或公开所述的 RL 信号训练推理 | 能力基础；正确性奖励可能让链更长 |
| 同样答对能否更短？ | O1-Pruner、Arora–Zanette | 参考长度或题内长度统计进入正确性奖励 | 与最短 SFT、难度偏好竞争 |
| 给定预算能否用好？ | L1-Exact/L1-Max | 把目标长度作为条件并训练服从 | 与 s1 强制截断竞争，可接预算预测器 |
| 是否需要工具？ | Search-R1、ReTool | 学搜索/代码解释器的时机与内容 | 可能用少量工具替换很长文本，也可能增加输入 |
| 如何在多种策略间切换？ | Agentic-R1、Agentic-RL 消融研究 | 双教师策略蒸馏，或训练中优化长程行为 | Agentic-R1 本身不是在线 RL；属于交叉方案 |
| 何时结束与验证？ | 超长惩罚、步骤/调用控制、动态早停 | 避免无限探索，同时保留必要自检 | 训练方法与[08 章部署控制](08-harness-agents.md)互补 |

[后训练原文卡 P07/P09/P13/P16–P19/P24](../sources/posttraining-map.md)与[企业 RL 披露](../sources/vendor-practice.md)记录证据边界。

## 2. GRPO 是更新工具，效率问题另有来源

DeepSeekMath 提出 GRPO，以同题多条回答的组内奖励作为比较基线，减轻对独立价值模型的依赖；o1 的公开材料说明用大规模 RL 改善推理，但未披露足以把它归为 GRPO 的细节。[DeepSeekMath](https://arxiv.org/html/2402.03300v3)、[OpenAI o1](https://openai.com/index/learning-to-reason-with-llms/)

仅奖励正确时，更长的自检、回溯和搜索可能提升正确率，模型因此愿意花更多 token。Demystifying Long CoT 观察到长度增长以及长度奖励被重复文本利用。这说明成本需要显式处理，但不证明应该删除所有自检。[原文](https://arxiv.org/html/2502.03373v1)

完整 GRPO 推导和全对/全错组算例已经下沉到[技术专题](technical/06-rlvr-agentic.md)。总论和本章关注的是路线之间的差别，不把算法公式当作领域全貌。

## 3. 固定长度惩罚、参考校准与预算条件化

O1-Pruner 以冻结参考策略对每题的长度和正确性为参照，在不低于其期望准确率的约束目标下优化长度与准确性；这是优化目标，不是实际训练的无损保证。Arora–Zanette 对正确回答的长度做题内标准化和平滑惩罚，效率系数决定惩罚强度；设计上试图避免简单题与困难题共用一个绝对长度阈值，尚未证明跨域普适。[O1-Pruner](https://arxiv.org/html/2501.12570v2)、[Training Language Models to Reason Efficiently](https://arxiv.org/html/2502.04463v4)

两者都比“每多一个 token 统一扣同样分”更细，但仍需要可靠答案、足够正确样本和良好优化。Arora–Zanette 的理论保证依赖表格策略、正确解覆盖和总体最优等假设，不能用来承诺神经网络训练没有性能损失。

L1 改问另一个问题：用户已经给定长度，模型能否在这个约束内答好？Exact 奖励接近目标长度，Max 用软惩罚鼓励不超目标，也允许更短，但仍可能超目标；它与 s1 通过外部结束标记截断不同，是训练策略预先适应预算。给预算的预测器可以来自 TALE/DAST，但组合不等于已有独立证据。[L1](https://arxiv.org/html/2503.04697v2)、[s1](https://arxiv.org/html/2501.19393)

L1 报告的低违约率包含按绝对长度偏差超过500 token 才计违约的 **soft violation** 定义。部署需要严格上限时，仍要由 harness 执行硬预算；不能将该数字当作几乎从不超限。

## 4. Agentic-RL 把成本从文字扩展到整个过程

Search-R1 在推理中搜索，检索文本不参与策略 loss；但这些文本仍会被模型读入，因此部署账不能把它们删除。论文中策略变强后搜索更多，长度也可能增加。[Search-R1](https://arxiv.org/html/2503.09516v5)

ReTool 用代码解释器替代部分文字计算，先冷启动再 RL，训练曲线中的输出变短具有直接价值。但这是把计算搬给程序，不是凭空消失；模型输入、代码、解释器回传与失败重试都要记账。工具执行时间和费用另列。[ReTool](https://arxiv.org/abs/2504.11536)

Agentic-R1 组合文本教师和工具教师，再自蒸馏，让学生具备两类策略；它主要是蒸馏路线。Demystifying RL in Agentic Reasoning 则消融训练损失、clip、超长惩罚和工具奖励。把工具调用次数纳入奖励不一定是压少工具，有时奖励的目的正是鼓励使用有效工具。[Agentic-R1](https://arxiv.org/html/2507.05707v2)、[Agentic-RL 消融](https://arxiv.org/html/2510.11701v1)

## 5. 原文结果和仍有争议的地方

| 工作 | 条件 | 结果与限制 |
| --- | --- | --- |
| O1-Pruner | Marco-o1-7B，MATH 5K训练，每题16个参考样本 | 三测试集平均932→554 tokens、73.4→76.8%；点估计同时改善，没有固定准确率检验 |
| Arora–Zanette | R1-Distill-Qwen-7B，Numina 训练，MATH500、α=0.1 | 某配置约 4K→2.6K token，准确率损失约 2.2 个百分点；是折中而非严格支配 |
| L1 | DeepScaleR-1.5B，40K 数学数据，预算条件 | 提高低预算表现；L1-Max 385/39.1% 不能声称胜过 表中 Qwen-1.5 的 752/41.0% |
| ReTool | Qwen2.5-32B、AIME、16,384 序列上限 | 训练中输出约 10K→6K；不是含 sandbox 的完整部署成本结论 |
| Search-R1 | Qwen2.5-7B、七个 QA 集、Wikipedia 检索 | 平均 EM .431，对照无搜索 .276；后期长度可能上升 |
| Agentic-R1 | 7B、约 2.6K 轨迹，4K/32K 预算、五个数学集 | 自蒸馏版本平均改善，某些 MATH500 配置退化；两个上限点不是整条曲线 |
| OpenAI Codex-Max | 官方 SWE-bench Verified、medium 对 medium | 报告较前代少 30% thinking tokens 且性能更好；私有训练与完整输入账未公开 |

最后一项来自[官方发布](https://openai.com/index/gpt-5-1-codex-max/)。它支持企业在研究这一目标，不能分离某个优化器的贡献。

O1-Pruner、Arora–Zanette 和 L1 原文都未给出足以宣称训练/测试绝对无污染的审计；原模型预训练覆盖与同题族泄漏仍须独立检查。多条工作支持“正确轨迹里常有可去除冗余”；具体奖励是否保留困难能力、能否迁移到开放工具任务，仍没有统一答案。数学 RLVR 的优点是验证便宜、曲线容易复测，缺点是该方向已较拥挤且数学短链未必迁移到 Agent。因此是否选它应由新机制和未解决证据决定，而不是由现成 veRL 环境决定。

## 6. 阅读顺序与避免重复

先读正确性 RL 的长度现象，再按 O1-Pruner→Arora–Zanette→L1 读成本控制；工具方向按 Search-R1→ReTool→Agentic-R1，始终核对环境观察的计量。

| 初步 idea | 最近邻 | 要建立新贡献需验证什么 |
| --- | --- | --- |
| GRPO 奖励减去长度 | O1-Pruner、Arora–Zanette、长链奖励分析 | 难度、全错组、稀有正确解和奖励漏洞怎样处理？ |
| prompt 给定预算再 RL | L1；TALE 预算预测 | 严格上限、跨域校准和输入成本是否覆盖？ |
| 用代码工具让推理更短 | ReTool、Agentic-R1 | 工具失败、慢工具和大型回传下是否仍省整项成本？ |

本地图的机会判断：在一个现成可执行工具环境里，研究“验证结果决定继续/停止/换策略”的成本学习，往往比重复一个数学长度惩罚更有机制空间；仍需与简洁提示、遮蔽、硬预算和现有工具策略作强基线比较。

<a id="06rlvr-与-agentic-rl奖励结果怎样学会少走弯路"></a>
<a id="1-rlvr把判断标准接到执行结果上"></a>
<a id="2-grpo同一道题多做几次再比较"></a>
<a id="3-两种-epsilon解决两个完全不同的问题"></a>
<a id="4-想省-token先防止奖励鼓励失败"></a>
<a id="5-agentic-rl从一段回答到一个环境过程"></a>
<a id="6-终止验证和环境版本也是学习问题"></a>
<a id="7-cpu-小实验与练习"></a>

旧版小节链接已保留。原来的推导、算例和练习见[本章技术专题](technical/06-rlvr-agentic.md)。

# 后训练三章的一手来源与验证记录

适用文件：`docs/04-sft-distillation.md`、`docs/05-preference-rl.md`、`docs/06-rlvr-agentic.md`。检索日期 2026-09-09。正文是教学解释和数学推导，不是文献逐段翻译；没有复制论文图片、实验曲线或表格。所有本地图片来自教学程序的合成数值。

| 来源 | 一手链接与定位 | 用途与边界 |
|---|---|---|
| Hinton et al., Distilling the Knowledge in a Neural Network | https://arxiv.org/abs/1503.02531 | 软目标、温度蒸馏的来源；未据此宣称 token 数必定下降 |
| Agarwal et al., On-policy Distillation of Language Models: Learning from Self-Generated Mistakes（GKD） | https://arxiv.org/html/2306.13649v3 | 学生生成状态上的蒸馏，以及采样分布和散度选择的区别 |
| Thinking Machines Lab, On-Policy Distillation | https://thinkingmachines.ai/blog/on-policy-distillation/ ，Loss function / Pseudocode | 学生轨迹、教师 sampled-token logprob、局部 reverse KL 配方；未引用其计算收益作为通用结论 |
| Ouyang et al., Training language models to follow instructions with human feedback | https://arxiv.org/abs/2203.02155 | SFT→偏好奖励模型→RL 的代表性流程，非必选顺序 |
| Bai et al., Constitutional AI | https://arxiv.org/abs/2212.08073 | AI 反馈与规则评价的实例；未把 AI 反馈等同客观真值 |
| Rafailov et al., Direct Preference Optimization | https://arxiv.org/html/2305.18290v3#S3 ，§3–4、附录 A.1–A.2 | BT 偏好模型、KL 正则最优策略、奖励重参数化与 DPO；正文列出支持集、归一化和理想优化条件 |
| Schulman et al., Proximal Policy Optimization Algorithms | https://arxiv.org/abs/1707.06347 ，clipped surrogate objective | PPO ratio 和正负优势下裁剪方向；明确它不是 ratio 的硬约束 |
| Shao et al., DeepSeekMath | https://arxiv.org/html/2402.03300v3#S4.SS1 ，§4.1.1–4.1.2 | GRPO 组奖励基线、按回答归一化及 outcome advantage；代码额外添加分母 epsilon，并明确总体标准差只是本示例约定 |
| Lambert et al., Tülu 3 | https://arxiv.org/abs/2411.15124 | 可验证奖励后训练实例，未把其模型表现推广为统一推荐 |
| Search-R1 authors | https://github.com/PeterGriffinJin/Search-R1 | 推理与搜索交替的 Agent RL 实例，区分环境观察和策略动作 |
| AppWorld authors | https://github.com/StonyBrookNLP/appworld#-evaluating-the-agent | 终态与数据库变化检查的例子；不声称测试环境即独立训练数据 |
| Ahmad et al., OpenCodeReasoning | https://arxiv.org/html/2504.01943v1#S4.SS1 | SFT 执行筛选可能改变困难题覆盖的反例；不外推为错误答案普遍优越 |

## 教学性推导与设计，不能误标为论文结果

- 两种 SFT 平均损失的 2/8-token 权重例子、固定状态的正反 KL logits 梯度、完整序列 KL 的 score-function 梯度，是正文展开的数学解释。
- DPO 标量间隔更新只验证一个可微损失的方向，未模拟语言模型参数共享。
- 正确性门控奖励 `c*(1-lambda*min(cost/B,1))` 是本教程合成设计，`0<=lambda<1` 保证正确样本奖励严格高于错误样本；这不保证训练后成功率不下降，也不是生产训练配方。
- 所有任务奖励相同才导致组中心化优势全零；二值正确性下全正确/全错误是特例。附加成本或 KL/辅助 loss 时不能声称整体无更新。
- 部署成本采用全部模型输入+输出 token；离线教师/训练成本单列。token 数、计算量、时延、金额没有互换。

## 可复现实验与图

```bash
python3 examples/posttraining_lab.py --self-test --plots
```

程序只依赖 Python 标准库，不联网、不加载 checkpoint、不执行生成代码。SVG 图也由标准库写出，不需要 matplotlib。

2026-09-09 已实际运行通过：

- 正向及反向 KL 对 logits 的解析梯度与中心有限差分一致（绝对容差 `1e-7`）。
- DPO 间隔梯度与有限差分一致；标量优化 loss 从约 0.6931 降至 0.5571。
- PPO 对正负优势的裁剪符号，以及全相同奖励组零优势检查通过。
- 门控成本饱和后正确奖励仍高于零成本错误奖励。
- 生成 `assets/plots/posttraining-kl.svg`、`posttraining-clip.svg`、`posttraining-grpo.svg`；均带标题、说明和坐标标签，明确是合成例子。
- 使用本地 `rsvg-convert` 将 SVG 渲染为临时 PNG 后逐张视觉检查；坐标、图例与正文结论一致。该渲染器只用于作者检查，不是读者运行实验的依赖。

这些检查是程序和数学的一致性验证，不是 SFT、DPO、OPD 或 GRPO 的大模型效果验证。正文相邻章节链接按集成约定为 03→04→05→06→07。

## 直接研究推理长度的补充主源

- [TokenSkip v2 §3](https://arxiv.org/html/2502.12067v2#S3)：先压缩正确轨迹，再作比例条件 SFT；专题以自定义保留率和固定数量选择解释，不混用论文的分位数符号。
- [Arora 与 Zanette v3 §4](https://arxiv.org/html/2502.04463v3#S4)：正确回答的题内长度统计、sigmoid 奖励、PPO/RLOO，最优性分析有简化表示假设；未称其为 GRPO。
- [s1 原论文](https://arxiv.org/abs/2501.19393)与[作者实现](https://github.com/simplescaling/s1)：SFT 与思考终止/续写控制的组合；控制工作点本身不证明整条前沿改善。

# 专题：直接控制推理长度的三条研究路线

[返回目录](../README.md) · 前置：[SFT 与蒸馏](04-sft-distillation.md)、[RLVR](06-rlvr-agentic.md)、[上下文工程](07-prompt-context.md)

前面的章节建立了通用工具。本专题把它们接到直接研究推理 token 效率的三项代表工作：TokenSkip 改变训练示范，长度奖励改变在线学习目标，s1 的 budget forcing 改变解码控制。这些是不同路线的实例，尚未形成适用于所有任务的统一配方。

| 路线 | 改变的位置 | 部署时怎样调节 | 需要防止的误读 |
|---|---|---|---|
| TokenSkip | 用压缩后的推理轨迹做条件 SFT | 在 prompt 指定压缩比例 | 生成完再删字，并没有省掉已生成的 token |
| 长度奖励 RL | 在正确性奖励中加入题内长度惩罚 | 原论文用不同训练系数获得不同策略 | 较短且略低准确率是折中点，未必构成 Pareto 支配 |
| s1 budget forcing | 结束或延长思考阶段 | 控制思考预算与继续次数 | 更大预算下性能提高，是移动工作点；不自动证明曲线整体改善 |

## 1. TokenSkip：训练模型直接产生压缩轨迹

[TokenSkip](https://arxiv.org/html/2502.12067v2#S3) 先生成推理轨迹并筛去最终答案错误的轨迹，再用 LLMLingua-2 估计重要性，构造不同比例的压缩轨迹。压缩比例写进训练输入，最终答案保留，同时混入未压缩轨迹；随后做 SFT。部署时模型直接自回归生成较短的过程。[作者代码](https://github.com/hemingkx/TokenSkip)

为避免“压缩率”到底指删掉还是保留的歧义，本节另定义保留率 $`\rho\in(0,1]`$ 。给原轨迹 $`z_{1:L}`$ 和重要性分数 $`u_i`$ ，教学版保留前 $`k=\lceil\rho L\rceil`$ 个高分位置，再按原顺序组合为 $`\tilde z`$ 。这里用固定个数选择解释直觉，不冒充论文分位数阈值的实现；同分、分词和实际保留率都需要单独处理。

若答案为 $`a`$ ，拼接目标 $`y=[\tilde z;a]`$ ，采用标准的负对数似然最小化记号：

```math
\mathcal L(\theta)=-\mathbb E_{(x,z,a),\rho}
\sum_{t=1}^{|y|}\log p_\theta(y_t\mid x,\rho,y_{\lt t}).
```

它让模型学习“在这个比例条件下，下一步应该直接说什么”，而不是在部署后处理阶段删除已经花过成本的文字。训练时压缩原轨迹的模型调用属于离线账；若部署额外要求恢复完整解释，那次恢复也要进入部署账。

下面用手工重要性分数演示选择与顺序保持；这些空白分隔单位不是任何真实 tokenizer。

```python
from math import ceil

def retain_units(units, importance, keep_fraction):
    if len(units) != len(importance) or not 0 < keep_fraction <= 1:
        raise ValueError("invalid lengths or retention fraction")
    k = ceil(len(units) * keep_fraction)
    selected = sorted(range(len(units)), key=lambda i: (-importance[i], i))[:k]
    return [units[i] for i in sorted(selected)]

units = ["首先", "速度=3", "然后", "时间=4", "因此", "位移=12"]
assert retain_units(units, [0, 1, 0, 1, 0, 1], 0.5) == ["速度=3", "时间=4", "位移=12"]
```

重要性分数不是因果必要性的证明。若把“不”“除非”或单位删掉，文本仍可能看似流畅却改变了答案。应比较未压缩 SFT、同规模压缩 SFT、简洁 prompt、直接截断等基线，并在未见题型和不同预算下检查结果。

## 2. 长度奖励：在同一道题的正确解之间比较长短

Arora 与 Zanette 的 [Training Language Models to Reason Efficiently](https://arxiv.org/html/2502.04463v3#S4) 从已有推理模型开始在线 RL。每题采样多条回答，按该题**正确回答的长度**计算均值与标准差，再通过 sigmoid 形成有界惩罚。原论文采用 PPO 与 RLOO 优势估计，不应把它称为一种 GRPO 实现。

令 $`c_i\in\{0,1\}`$ 为正确性、 $`L_i\ge0`$ 为回答长度。若同题存在正确样本，记其长度均值和标准差为 $`\mu_+,s_+`$ ， $`\alpha\in[0,1)`$ 。教学实现增加明确的数值保护 $`\varepsilon\gt 0`$ ：

```math
z_i=\frac{L_i-\mu_+}{s_++\varepsilon},\qquad
r_i=c_i\left[1-\alpha\sigma(z_i)\right],\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.
```

同题较短的正确回答获得较高奖励。用题内统计量，是为了让困难题的一千 token 与该题其他解比较，而不是一律与简单题的一百 token 比较。全错时不存在 $`\mu_+,s_+`$ ，下面直接返回零奖励；正确长度完全相同时，长度项不能区分这些回答。数值保护及这些边界处理是本教学代码明确选择的约定，生产复现须核对原实现。

```python
import math
from statistics import mean, pstdev

def length_rewards(lengths, correct, alpha=0.2, eps=1e-8):
    if len(lengths) != len(correct) or not 0 <= alpha < 1 or not eps > 0:
        raise ValueError("invalid arguments")
    if any(not math.isfinite(n) or n < 0 for n in lengths):
        raise ValueError("lengths must be finite and nonnegative")
    if any(c not in (0, 1) for c in correct):
        raise ValueError("correctness must be binary")
    good = [n for n, c in zip(lengths, correct) if c]
    if not good:
        return [0.0] * len(lengths)
    center, scale = mean(good), pstdev(good) + eps
    rewards = []
    for n, c in zip(lengths, correct):
        z = (n - center) / scale
        sigmoid = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
        rewards.append(c * (1 - alpha * sigmoid))
    return rewards

r = length_rewards([100, 300, 1], [1, 1, 0])
assert r[0] > r[1] > r[2] == 0
assert length_rewards([1, 100], [0, 0]) == [0.0, 0.0]
assert length_rewards([100, 100], [1, 1]) == [0.9, 0.9]
```

RLOO 在组大小 $`G\gt 1`$ 时使用其余回答的平均回报作为基线：

```math
A_i=r_i-\frac{1}{G-1}\sum_{j\ne i}r_j.
```

它不需要额外价值网络。这里的 $`G-1`$ 是留一法的分母，需要至少两条回答；它与 [GRPO](06-rlvr-agentic.md) 的标准差保护是不同问题。

原论文的最优性分析依赖可独立表示每题响应分布等简化假设，不能直接成为共享参数 LLM、有限采样、含噪 verifier 下的保证。原文的一些结果保留了大部分准确率但仍有下降；应把它们读成性能—长度折中，而不是写成所有预算上的无损压缩。

## 3. s1：把思考预算作为解码器的控制变量

[s1](https://arxiv.org/abs/2501.19393) 将小规模推理示范 SFT 与 budget forcing 结合。推理时，可以强制结束思考阶段；也可以在模型准备结束时追加“Wait”并继续，从而延长思考。代码使用模型特定的思考结束标记与采样设置，不能把该字符串直接搬到任意模型而期待等价行为。[作者实现](https://github.com/simplescaling/s1#inference)

它与“请少写一点”的软指令不同：控制器实际改变停止行为。但强制结束思考仍应给最终答案留预算，否则可能只是截断出一个未完成的答案。简化地，若一次生成总额度为 $`B`$ ，预留答案额度 $`R`$ ，则思考最多使用 $`B-R`$ ；真实方法还需把输入、续写重读和辅助调用加到整项任务账中。

用两个预算点时，应问：同一策略从 $`(C_1,Q_1)`$ 移到 $`(C_2,Q_2)`$ ，只是用了更多计算，还是相对相同成本的强基线提高了质量？s1 为研究预算响应曲线提供了可复用干预，却不保证延长思考一定改善每道题。

## 4. 从这三条路线组织一次实验

固定底座和独立测试任务，先比较默认生成、简洁 prompt、预算控制；之后才分别加入压缩 SFT 或长度奖励 RL。SFT 保持题目覆盖、训练 token 与训练配置可比；RL 固定奖励验证器并记录全错组比例；推理控制始终保留预算耗尽的失败任务。

报告多个实际总 token 成本点，而不是只挑一个最短结果。另报输出 token、训练成本、延迟和置信区间，特别检查困难题是否掉分。若引入新的失败类型，应回到[评测章](09-evaluation.md)增加针对性的机制分析，而不在测试集上反复挑最有利的长度系数。

完整可运行脚本为 [efficient_reasoning_lab.py](../examples/efficient_reasoning_lab.py)，在仓库根目录运行 `python3 examples/efficient_reasoning_lab.py`。

上面的两个代码块已在 Python 标准库环境执行通过，只验证选择与奖励算术；没有执行这三篇论文的大模型训练或推理实验。

[返回目录](../README.md)

# Fast Inference from Transformers via Speculative Decoding

[所属章节](../index.md) · [来源与阅读状态](../../../sources/papers.json)


- **作者**：Yaniv Leviathan、Matan Kalman、Yossi Matias（Google Research；前两位共同贡献）
- **年份与固定版本**：2023，arXiv:2211.17192v2（2023-05-18；ICML 2023，PMLR 202）
- **原始链接**：[arXiv:2211.17192v2](https://arxiv.org/abs/2211.17192v2)
- **实际阅读范围**：固定版本 PDF 的摘要、正文第 1–6 节、图 1–5、表 1–3，以及与正文公式和结论直接相关的附录 A.1–A.5；同时核对了作者随文 LaTeX 工程中的 `main.tex`、表格和图注，并从工程提取原始图。论文没有随文提供可用于复现实验的完整实现；本文没有把后续独立实现当作本论文证据。

## 1. 问题和核心主张

自回归 Transformer 每生成一个 token 都要把新前缀送入模型一次。因此生成 $`K`$ 个 token 通常需要 $`K`$ 次串行运行大模型，即使单次运行主要受参数和 KV cache 的内存读带宽限制，设备上仍可能有未利用的并行算力。论文把这两个事实结合起来：让便宜的近似模型先猜一段，再让目标大模型对整段候选并行验证；只要某个猜测能被目标分布“容纳”，就接受它，否则在第一次失败的位置按一个修正分布重采样。关键不是让小模型取代大模型，而是把大模型的串行步骤改成并行验证步骤。

论文的严格保证是：无论近似模型 $`M_q`$ 多差，最终序列的概率分布与只用目标模型 $`M_p`$ 自回归采样完全相同。速度来自增加并发和可能多生成 token，不来自改变模型、重新训练或减少目标分布中的 token。正文报告在 T5-XXL 上相对 T5X 的实际 wall-time 加速约 2–3 倍，输出保持一致；这依赖有足够并行资源，且不是所有硬件或短序列都会受益。

## 2. 分布、标准化和单 token 的 speculative sampling

给定前缀 $`x_{<t}`$，目标模型产生 $`p(x_t\mid x_{<t})`$，近似模型产生 $`q(x_t\mid x_{<t})`$。为简化记号，正文在前缀固定时写成 $`p(x)`$、$`q(x)`$。argmax、temperature、top-k、nucleus 等策略先在 logits 或概率上做相同的标准化，然后都被视作从一个最终概率分布抽样；因此算法保证的是“标准化后分布”一致，而不是保证不同采样规则之间一致。

单 token 的 speculative sampling 是：先抽 $`x\sim q`$。若 $`q(x)\le p(x)`$，无条件接受；若 $`q(x)>p(x)`$，以

```math
1-\frac{p(x)}{q(x)}
```

的概率拒绝它，以 $`p(x)/q(x)`$ 的概率接受。拒绝后从残差分布

```math
p'(x)=\mathrm{norm}\bigl(\max(0,p(x)-q(x))\bigr)
```

重新抽一个 token。这里的 `norm` 是在整个词表上归一化。等价地，对一个已经从 $`q`$ 抽出的候选，接受概率是

```math
a(x)=\min\left(1,\frac{p(x)}{q(x)}\right),
```

其中只会访问 $`q(x)>0`$ 的候选；若 $`q(x)=0`$，该 token 不可能由提案抽出，$`p(x)/q(x)`$ 不需要计算。若 $`q(x)>p(x)=0`$，接受概率为 0。拒绝分支有意义的条件是总体接受率小于 1；若接受率等于 1，残差质量为 0，算法不会进入该分支。

### 为什么接受加残差恰好恢复 $`p`$

设候选词为 $`x_0`$。接受分支最终输出 $`x_0`$ 的概率为

```math
q(x_0)\min\left(1,\frac{p(x_0)}{q(x_0)}\right)=\min(q(x_0),p(x_0)).
```

记总接受率为 $`\beta`$。残差的未归一化质量是 $`p(x)-\min(p(x),q(x))`$，总质量为

```math
1-\beta=\sum_x\bigl[p(x)-\min(p(x),q(x))\bigr].
```

所以拒绝后从归一化残差抽到 $`x_0`$ 的联合概率是

```math
(1-\beta)p'(x_0)=p(x_0)-\min(p(x_0),q(x_0)).
```

两条路径相加得到 $`p(x_0)`$。这同时说明支持集条件：不要求 $`p`$ 和 $`q`$ 的支持集相同；只要求它们是同一离散词表上的概率分布。若某 token 只在 $`p`$ 的支持集内，它会在残差分支中出现；若只在 $`q`$ 中，最多被接受到 $`p`$ 能承受的那部分，其多余质量被拒绝。附录 A.1 给出了同一证明。

### 两 token 词表的数值例

令词表为 $`\{A,B\}`$，目标分布 $`p(A)=0.6,p(B)=0.4`$，提案分布 $`q(A)=0.8,q(B)=0.2`$。抽到 $`A`$ 的概率是 $`0.8`$，其接受概率为 $`0.6/0.8=0.75`$，因此接受路径贡献 $`A`$ 的质量 $`0.8\times0.75=0.6`$；拒绝路径贡献 $`0.8\times0.25=0.2`$，残差为 $`(\max(0,0.6-0.8),\max(0,0.4-0.2))=(0,0.2)`$，归一化后必取 $`B`$。抽到 $`B`$ 的概率是 $`0.2`$，因为 $`q(B)=0.2\le p(B)=0.4`$，全部接受。故最终 $`P(A)=0.6`$、$`P(B)=0.2+0.2=0.4`$，正好等于 $`p`$。本例的接受率为 $`\beta=\min(0.8,0.6)+\min(0.2,0.4)=0.8`$；残差归一化常数为 $`0.2`$。它也展示了分母 $`q(x)`$ 只对已抽出的候选有正值。

## 3. 多 token speculative decoding

设每轮提案长度为 $`\gamma\in\mathbb Z^+`$。算法先在近似模型上自回归地生成
$`x_1,\ldots,x_\gamma`$：

```math
q_i(x)=M_q(x\mid prefix+[x_1,\ldots,x_{i-1}]),\qquad x_i\sim q_i.
```

随后目标模型一次并行计算 $`\gamma+1`$ 个位置：

```math
p_1=M_p(prefix),\quad p_2=M_p(prefix+x_1),\quad\ldots,\quad
p_{\gamma+1}=M_p(prefix+x_1+\cdots+x_\gamma).
```

第 $`i`$ 个猜测使用独立均匀随机数 $`r_i`$ 检验 $`r_i\le p_i(x_i)/q_i(x_i)`$（比例超过 1 时等价于必接收）。令 $`n`$ 为第一个失败之前连续接受的数量；若所有 $`\gamma`$ 个都通过，则 $`n=\gamma`$。严格写法是

```math
n=\min\left(\{i-1:1\le i\le\gamma,\ r_i>p_i(x_i)/q_i(x_i)\}\cup\{\gamma\}\right).
```

若 $`n<\gamma`$，在第一个失败位置使用

```math
p'(x)=\mathrm{norm}\bigl(\max(0,p_{n+1}(x)-q_{n+1}(x))\bigr);
```

若全通过，则直接从 $`p_{\gamma+1}`$ 抽一个额外 token。返回 `prefix + [x_1,…,x_n,t]`，因此每轮至少产出 1 个 token，最多产出 $`\gamma+1`$ 个 token。目标模型的 $`\gamma+1`$ 次计算是在候选前缀上并行进行；提案模型的 $`\gamma`$ 次仍按 token 串行，但其单步便宜。

这种逐位置的拒绝/残差操作保持整条序列分布，是单 token 证明的条件化重复应用：当 $`x_1`$ 被接受时，后续 $`q_2,p_2`$ 都是在真实已接受前缀上比较；若 $`x_i`$ 首次失败，后面的猜测被丢弃，修正 token 从与该位置对应的目标分布和提案分布残差中抽取。不能把“接受了 $`n`$ 个提案”误读为目标模型没有计算后续位置；后续位置的并行计算可能因首次拒绝而成为浪费。

**伪代码对应图示：**先提案、并行验证、截断到首个失败、再从残差或 $`p_{\gamma+1}`$ 补一个 token。该顺序是论文 Algorithm 1 的教学重述；工程实现的缓存、批处理和设备调度细节没有在正文完整规定。

![Speculative decoding token suggestions, accepted guesses, rejected guesses, and corrections in unconditional LM.](../../../assets/papers/2211.17192/figure1.png)

图 1（`figure1.png`）展示无条件语言模型的一次次迭代：绿色是小模型猜中并被大模型接受的 token，红色是被拒绝的提案，蓝色是大模型残差修正 token。图中使用 6M 参数 GPT-like 提案模型和 97M 参数目标模型，均在 lm1b、8k tokenization 设置下；示例 38-token 句子只需要目标模型 9 次串行运行。图示表达的是串行调用数和颜色编码，不是新的概率或质量指标。

## 4. 接受率、生成长度和分布差异

给定前缀，定义 $`\beta_{x_{<t}}`$ 为对 $`x_t\sim q`$ 进行 speculative sampling 的接受概率。由于接受概率是 $`\min(1,p(x)/q(x))`$，有

```math
\beta=\mathbb E_{x\sim q}\left[\min\left(1,\frac{p(x)}{q(x)}\right)\right]
      =\sum_x\min(p(x),q(x)).
```

论文定义

```math
D_{LK}(p,q)=\sum_x|p(x)-M(x)|=\sum_x|q(x)-M(x)|,
\qquad M(x)=\frac{p(x)+q(x)}2.
```

利用 $`\min(p,q)=(p+q-|p-q|)/2`$，得到

```math
D_{LK}(p,q)=1-\sum_x\min(p(x),q(x)),\qquad \beta=1-D_{LK}(p,q).
```

因此 $`D_{LK}`$ 是对称、取值 $`[0,1]`$ 的差异量；为 0 当且仅当 $`p=q`$，为 1 当且仅当二者支持集不相交。文中最终把 $`\alpha=E(\beta)=E(\min(p,q))`$ 作为跨前缀的平均接受率。这里的 $`E`$ 是在实际前缀/位置上的期望，不能直接等同于某个单一 token 的 top-1 准确率。

若简化假设每个位置的接受事件独立同分布，平均率均为 $`\alpha`$，一轮产生的 token 数是成功概率 $`1-\alpha`$、上限 $`\gamma+1`$ 的截尾几何变量：先连续接受提案，第一次拒绝时再产生一个修正 token；全通过时也产生最后一个目标 token。因此

```math
E[N]=1+\alpha+\cdots+\alpha^\gamma
     =\frac{1-\alpha^{\gamma+1}}{1-\alpha}.
```

当 $`\alpha=0`$ 时每轮只产 1 个 token；当 $`\alpha\to1`$ 时上限趋近 $`\gamma+1`$。图 2（`alpha_v_tokens.pdf`）画出不同 $`\gamma`$ 下 $`E[N]`$ 随 $`\alpha`$ 增长的曲线。i.i.d. 是分析简化，不是语言模型前缀上的严格事实；附录 A.3 将它列为理论与实测有差异的原因之一。

## 5. Wall-time、算术量和选择 $`\gamma`$

令 $`T`$ 为一次目标模型解码的时间，$`c`$ 为一次提案模型运行时间与 $`T`$ 的比值。假设可以把 $`\gamma+1`$ 次 $`M_p`$ 运行并行化且不增加 wall-time，则一轮成本是 $`T(\gamma c+1)`$，而平均产出是上式，所以相对标准逐 token 解码的理论 wall-time 加速为

```math
S_{time}=\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}.
```

这要求生成足够长：短序列的首轮/尾轮边界会限制收益；论文没有把短序列修正项纳入该公式。$`\alpha`$ 是任务和模型分布的内在量，$`c`$ 则随硬件、实现、并行配置改变。实验中提案模型通常比目标模型小两数量级，$`c<0.05`$，但这是该设置的 profiling 结果而非普适常数。

若 $`\alpha>c`$，取 $`\gamma=1`$ 已能保证一个下界加速 $`(1+\alpha)/(1+c)>1`$；具体最优整数 $`\gamma`$ 是数值最大化上式。图 3（`alpha_v_optimal_gamma.pdf`）给出不同 $`c`$ 下最优 $`\gamma`$ 随 $`\alpha`$ 的变化：$`c`$ 越高，继续增大提案长度越不划算。作者还指出，若能根据每个前缀预测 $`\beta`$ 并动态调整 $`\gamma`$，理想 oracle 的平均产出上界是 $`1/(1-\alpha)`$；在典型 $`c,\alpha`$ 下，比固定 $`\gamma`$ 的加速最多约高 60%，但这是未来工作中的上界式分析，不是已实现结果。

并行带来算术量代价。令 $`\hat c`$ 为提案模型每 token 算术操作数与目标模型一次解码的比值。每轮有 $`\gamma`$ 次 $`M_q`$ 和 $`\gamma+1`$ 次并行 $`M_p`$，故相对标准解码的总操作倍数为

```math
S_{ops}=\frac{(1-\alpha)(\gamma\hat c+\gamma+1)}{1-\alpha^{\gamma+1}}.
```

当 $`\alpha`$ 低时，拒绝多、浪费的并行目标计算多；当 $`\alpha`$ 高时，操作倍数更接近 1。论文特别强调 wall-time 与算术量不是同一目标：目标模型权重和 KV cache 可在每轮只读取一次，若瓶颈是内存带宽，串行调用数减少仍可能明显加速；但额外算术资源不足时该方法不适用。对 Transformer decoder，除去 $`M_q`$ 后并行目标计算的总操作上界可由同规模 Transformer encoder 的一次运行界定，这是作者的架构层面观察。

表 1（正文表 1）在 $`c=\hat c=0`$ 时给出代表性折衷：$`\alpha=0.8,\gamma=5`$ 时速度 3.69X、操作 1.63X；$`\alpha=0.9,\gamma=10`$ 时速度 6.86X、操作 1.60X；它不是实际设备速度，而是简化模型下的速度/操作预测。图 4（`arith_ops.pdf`）用曲线展示相同折衷，图 5（`trace_diagram.pdf`）画出完整 encoder-decoder 堆栈的调度：$`\gamma=7`$ 或 3 时，蓝色 $`M_q`$ 块先串行排队，紫色 $`M_p`$ 块并行验证；标准解码在底部逐步调用目标模型。目标和提案 encoder 只在相应序列准备阶段调用一次，图中没有把每个设备通信细节展开。

## 6. 什么样的近似模型有用

分布正确性不要求 $`M_q`$ 与 $`M_p`$ 同架构，甚至不要求它是神经网络；限制来自 wall-time 的 $`c`$ 与接受率 $`\alpha`$ 的折衷。作者在实验中主要使用相同架构、相同标准化的更小 T5/GPT/LaMDA。经验上目标模型约小两数量级的提案模型通常较平衡。

- **可忽略成本模型**：$`c\approx0`$ 时理论加速上限为 $`1/(1-\alpha)`$。bigram 只是查表，在 En-De 任务上 $`\alpha\approx0.20`$，$`\gamma=3`$ 时给出约 1.25X 理论/实测意义上的加速。
- **上下文复制启发式**：摘要或聊天中常有长片段复现，可在找到匹配前缀时直接复制上下文 token。它不需参数和训练，但效果取决于任务重复结构。
- **非自回归模型**：可以一次提出整段候选，去掉 $`M_q`$ 的串行提案循环；正文只提出兼容性方向，没有实验。
- **随机模型**：从理论上即使完全随机的提案也有非零接受率，故存在极小的期望改善，但不代表生产上值得使用。
- **自定义训练/层级提案**：蒸馏、直接优化 $`\alpha`$、让提案模型再由更小模型加速，都被列作未来工作，没有本论文证据。

## 7. 实验一：T5-XXL wall-time

### 设置和公平比较

作者实现算法并与稳健的 T5X 实现比较。目标模型 $`M_p`$ 是标准 T5 v1.1 T5-XXL（11B），任务为 T5 论文中的：(1) WMT En-De 英德翻译，(2) CNN/Daily Mail 新闻摘要。提案模型使用已有 checkpoint 的 T5-small（77M）、T5-base（250M）、T5-large（800M）。在单个 TPU-v4、batch size 1 上测 wall-time，同时报告 argmax（temp=0）和标准 sampling（temp=1）。这比较的是已有模型和实现的解码延迟；论文没有给出完整输入长度分层、端到端服务吞吐或功耗测量。

### 结果

[原始表格/补充示例](https://arxiv.org/abs/2211.17192)

正文表 2 的主要结果如下（$`\alpha`$ 是经验接受率，speed 是相对 T5X）：

|任务|提案模型|温度|$`\gamma`$|$`\alpha`$|速度|
|---|---|---:|---:|---:|---:|
|En-De|T5-small|0|7|0.75|3.4X|
|En-De|T5-base|0|7|0.80|2.8X|
|En-De|T5-large|0|7|0.82|1.7X|
|En-De|T5-small|1|7|0.62|2.6X|
|En-De|T5-base|1|5|0.68|2.4X|
|En-De|T5-large|1|3|0.71|1.4X|
|CNN/DM|T5-small|0|5|0.65|3.1X|
|CNN/DM|T5-base|0|5|0.73|3.0X|
|CNN/DM|T5-large|0|3|0.74|2.2X|
|CNN/DM|T5-small|1|5|0.53|2.3X|
|CNN/DM|T5-base|1|3|0.55|2.2X|
|CNN/DM|T5-large|1|3|0.56|1.7X|

趋势是：在同一目标模型下，小模型通常给较好的速度折衷；更大的提案模型虽提高 $`\alpha`$，但 $`c`$ 增长更快，wall-time 反而下降。argmax 的分布更尖，$`\alpha`$ 和加速通常高于标准采样；例如 En-De 用 T5-small 时从 temp=1 的 2.6X、$`\alpha=0.62`$ 到 temp=0 的 3.4X、$`\alpha=0.75`$。摘要任务趋势类似，但最佳 $`\gamma`$ 较小。作者的“2–3X”结论来自这些真实 T5X wall-time，而非只由公式外推；输出分布保持不变是算法保证，运行时间是实验结果。

附录表 4 把公式预测与实测对照。例：En-De/T5-small/temp=0 的 $`\alpha=.75,c=.02`$，预测 3.2X、实测 3.4X；En-De/T5-large/temp=0 的预测 2.5X、实测 1.7X。差异归因于作者实现和 baseline 的优化差别，以及 $`\beta`$ 独立同分布假设只是近似。不能把预测值当作硬件无关保证。

## 8. 实验二：经验 $`\alpha`$ 的跨模型测量

作者用 Corollary 3.6，在每个设置下由目标模型生成 10K tokens，测量 $`E(\min(p,q))`$。这不是 wall-time，而是分布重叠/接受率估计；更高 $`\alpha`$ 只在提案成本和并发假设合适时才转化成更高速度。

1. **GPT-like**：目标 97M，lm1b 训练，Bert tokenizer、8k 词表；提案为 6M GPT-like（dim 256、FFN 1024、2 层、4 头）或 unigram/bigram。6M 模型的 $`\alpha`$ 是 0.88（T=0）/0.89（T=1）；unigram 约 0.03，bigram 约 0.05。
2. **T5-XXL En-De**：T5-small/base/large 的 $`\alpha`$ 在 T=0 为 0.75/0.80/0.82，在 T=1 为 0.62/0.68/0.71；unigram 0.08/0.07，bigram 0.20/0.19。这里更尖的 argmax 分布导致更高接受率。
3. **T5-XXL CNN/DM**：T=0 下 small/base/large 为 0.65/0.73/0.74，T=1 下为 0.53/0.55/0.56；unigram 为 0.13/0.08，bigram 为 0.23/0.16。
4. **LaMDA 137B 对话**：提案 100M/2B/8B。T=0 下 $`\alpha=0.61/0.71/0.75`$，T=1 下 $`0.57/0.71/0.74`$；LaMDA 输出始终经过 Top40 filter（对 argmax 无影响，对标准采样有影响）。正文没有给 LaMDA 的 wall-time 表，不能从这些数值宣称已验证 2–3X。

表 3 的总体观察是：相差一至两个数量级的模型常给 0.5–0.9 的接受率；简单 n-gram 也有非零重叠。$`\alpha`$ 的测量是目标模型生成 token 上的经验平均，论文未报告置信区间、不同随机种子或按输入长度分组。

## 9. 相关工作、边界和可迁移结论

蒸馏、稀疏化、量化和架构修改试图让每个 token 都更便宜，通常要改变训练或模型；自适应计算/early exit 根据难度跳过计算，但往往改变架构、需要训练，且不保持完全相同输出。Blockwise Parallel Decoding 也并行预测多个 token，但只支持 greedy，需要定制训练，重点是下游质量；SAD 主要复制输入，适用于纠错等输入输出相似场景，不支持一般提案模型和随机采样。本论文的差异点是：任意 $`M_q`$、随机采样、无需再训练，以及分布级精确性保证。后续独立工作在 Chinchilla 70B 上报告约 2–2.5X，但那不是本论文实验。

正文和附录给出的失败边界如下：

- **算力/并发不足**：$`\gamma+1`$ 个目标前向不能并行时，wall-time 假设失效；额外算术可能比标准解码更贵。
- **提案太贵或 $`\alpha`$ 太低**：$`c`$ 大、$`\alpha\le c`$ 时不保证存在收益；错误提案会浪费后续并行计算。
- **短生成**：公式假设足够长的生成，实际首尾边界使收益受限。
- **时间而非 token 数**：论文的核心测量是 wall-time、并行目标调用、算术操作和内存访问；没有测量或声称减少最终输出 token 数。每个最终 token 的分布仍是 $`p`$，speculation 只是内部提案和验证，不能把“每轮生成更多 token”表述成“任务需要更少 token”。
- **分布标准化一致性**：实验让提案模型和目标模型应用同一标准化；正文承认对两者使用不同变换可能还有收益，但没有验证。argmax 的 lenience 处理也与标准采样不同。
- **相关能力未完成**：beam search 只在附录给出方向：用提案 beam width $`u\ge w`$，若 $`top_w(M_p)\subseteq top_u(M_q)`$ 可保持原 beam 结果；完整分析留待未来。该方案会付出较高候选验证成本。
- **Lenience 是质量/分布放宽实验**：附录 A.5 令比较中的 $`q`$ 乘以 $`l\in[0,1]`$，允许候选概率最多是目标概率的 $`1/l`$ 倍，从而提高接受率但不再保持严格原分布。T5-XXL En-De、T5-small、标准 sampling 下 $`l=1,0.5,0.3,0.1`$ 的 $`\alpha`$ 为 0.62、0.71、0.76、0.84；在 $`c=.015`$ 时作者给出约 2.5X、3.1X、3.6X、5X 的加速。该表属于“愿意改变输出分布时”的附加结果，不能与主实验的 identical outputs 混写。温度 0 下作者改为先标准化前放宽，T5-small 对应 $`\alpha=0.75,0.75,0.80,0.87`$，$`\gamma=8`$ 时约 3.3X、3.3X、3.9X、4.9X。

## 10. 与推理时 token 目标的关系

这篇工作最适合用来区分“最终生成 token 数”和“生成这些 token 的串行目标模型调用数”。它严格保持每一步由 $`p`$ 定义的输出 token 分布；在同一个输出序列上，一轮可以把多个 token 的目标模型计算并行化，所以减少的是串行解码延迟和目标权重/KV 的重复读取。它没有声称压缩答案、删掉 token、降低上下文 token、降低输出 token，或减少总算术操作；在低接受率时总算术量反而增加。提案模型的内部猜测 token 是额外计算，验证失败的猜测会被丢弃，不应计入最终有效输出，也不应被当作“模型少生成了 token”。

## 11. 图和表资产记录

以下资产均由固定版本作者 LaTeX 工程原样复制到本任务临时目录 `assets/`，许可信息来自工程 manifest：CC BY 4.0（[许可证](http://creativecommons.org/licenses/by/4.0/)）。没有裁剪。`table2` 是正文表格，原工程没有独立图片文件，故只在本文用 Markdown 表重排，不能作为可复用图片。

|标记|原图编号/图注|来源文件|本地文件|许可/复用|
|---|---|---|---|---|
|`figure1`|Figure 1，unconditional LM 中的 token 接受/拒绝示意|`figure1.png`|`assets/figure1.png`|CC BY 4.0；允许，未裁剪|
|`alpha_v_tokens`|Figure 2，$`E[N]`$ 随 $`\alpha`$、不同 $`\gamma`$|`alpha_v_tokens.pdf`|`assets/alpha_v_tokens.pdf`|CC BY 4.0；允许，未裁剪|
|`alpha_v_optimal_gamma`|Figure 3，最优 $`\gamma`$ 随 $`\alpha`$、不同 $`c`$|`alpha_v_optimal_gamma.pdf`|`assets/alpha_v_optimal_gamma.pdf`|CC BY 4.0；允许，未裁剪|
|`arithm_ops`|Figure 4，速度和算术量随 $`\alpha`$、不同 $`\gamma`$|`arith_ops.pdf`|`assets/arith_ops.pdf`|CC BY 4.0；允许，未裁剪|
|`xprof`|Figure 5，encoder-decoder 调度 trace|`trace_diagram.pdf`|`assets/trace_diagram.pdf`|CC BY 4.0；允许，未裁剪|
|`table2`|Table 2，T5-XXL 实际 wall-time 结果|`main.tex` 表格|无独立图像|CC BY 4.0；正文表格可引用，非图片|

## 12. 定位式总结

作者的核心结论（第 6 节）是：在内存带宽瓶颈且有额外并行算力的常见配置中，speculative decoding 可以用现成模型实现有意义的 2–3X T5X wall-time 加速，同时保持目标输出分布。由公式直接推出的解释是，收益由接受率 $`\alpha`$、提案成本 $`c`$、提案长度 $`\gamma`$ 和并行能力共同决定；接受率本身只是 $`p,q`$ 的重叠度，不能孤立地当作速度。本文的两 token 算例和残差证明说明了为何“先采 $`q`$、再修正”仍严格采自 $`p`$。实验只覆盖文本任务和有限硬件/模型组合；完整服务级 token 成本、功耗、不同上下文长度、动态 $`\gamma`$、非自回归提案和 beam search 的系统验证均未报告。

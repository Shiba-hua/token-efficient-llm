# Chain of Draft：Thinking Faster by Writing Less

[所属章节](../index.md) · [来源与阅读状态](../../../sources/papers.json)


**作者**：Silei Xu、Wenhao Xie、Lingxiao Zhao、Pengcheng He（Zoom Communications） <br>
**年份**：2025（arXiv v2，在线版本 2025-03-03） <br>
**固定版本**：[arXiv:2502.18600v2](https://arxiv.org/abs/2502.18600v2)，[PDF](https://arxiv.org/pdf/2502.18600v2)；作者代码与数据链接为 [sileix/chain-of-draft](https://github.com/sileix/chain-of-draft)。

## 我读了什么

本稿以缓存的 `paper.pdf`、其文本抽取、LaTeX 主文和作者随源文件提供的 `plot.png` 为依据，覆盖摘要、Introduction、Related Work、Chain-of-Draft Prompting、Experiments（Experimental Setup、GSM8K、BIG-bench date/sports、coin flip、正文 Limitations）和 Discussion。表 1--6 均按正文/LaTeX 表头读取。源目录另有 `sections/ablation.tex`，但 `paper.tex` 将 ablation 注释掉，固定 PDF 没有这部分；因此文中“词数消融”和“不同模型尺寸”只作为未纳入正文的源文件线索，不当作已报告结果。参考文献只用于定位作者声称的前作关系，没有把新发现的论文当成本论文证据。

## 问题与核心主张

CoT（Chain of Thought）让模型把问题拆成逐步自然语言推理，通常能提高多步任务准确率，但会把中间过程写得很长，增加生成 token、推理延迟和费用。作者的观察是，人类解题时往往只写草稿中的关键量、变换或约束，而不复述每个显然事实。CoD（Chain of Draft）把这个观察变成一个推理时提示：仍要求模型逐步思考，但每一步只保留一个“minimum draft”，提示上限为 **5 个词**，并在 `####` 后给最终答案。

这不是一个新模型结构、训练目标或隐藏状态算法。它是黑盒 LLM 也能采用的 prompt-level 约束：让中间轨迹在自然语言空间中更稠密。作者主张，在有 few-shot 草稿示例时，CoD 在算术、常识、符号三类任务上可接近或超过 CoT，并以更少输出 token 和更低实测延迟达到这种准确率；摘要所说的“最低 7.6% token”是相对对应 CoT 输出的最小比例，不能解释为所有任务或总请求 token 的统一比例。

## CoD 如何工作

### 三种提示的操作差异

正文的标准 few-shot 基线给模型输入-输出例子，要求直接返回答案，不给解释。CoT 使用 CoT 论文附录中的同样 few-shot 例子，并把稳定抽取用的最终答案放到 `####` 后。CoD 也要求逐步思考，区别在于 few-shot 例子的中间部分由作者手写成短草稿，并在系统提示中加上每步最多 5 个词的指导。

正文给出的三个系统提示是：

```text
Standard:
Answer the question directly.
Do not return any preamble, explanation, or reasoning.

Chain-of-Thought:
Think step by step to answer the following question.
Return the answer at the end of the response after a separator ####.

Chain-of-Draft:
Think step by step, but only keep a minimum draft for each thinking step,
with 5 words at most.
Return the answer at the end of the response after a separator ####.
```

这里的“5 words at most”是软性提示，论文明确说**没有任何机制强制模型遵守**；它不是 tokenizer 意义上的严格 5-token 上限，也不是一次回答最多生成 5 个 token。一个草稿步骤可能因数字、标点、词形和 tokenizer 规则对应多个 token。实验表中的 Token # 是每个 response 的平均输出 token 数，远大于 5，正好说明该提示是每步指导而非全局长度截断。

### 可跟踪的算例

题目是：“Jason 有 20 根棒棒糖，给了 Denny 一些，现在剩 12 根，给了多少？”三种行为的机制差异如下。

* Standard 直接输出 `8`。它成本低，却不给可检查的中间状态；作者将这种缺少外显中间结果与多步任务更易出错联系起来，这是动机性解释，不是单独的因果实验。
* CoT 写出“初始 20”“剩余 12”“求差”“列式”“20 - 12”“结果 8”等完整叙述，最后 `#### 8`。这些文字可读性高，但重复了对运算无贡献的故事角色和自然语言说明。
* CoD 写成 `20 - x = 12; x = 20 - 12 = 8. #### 8`。它仍然暴露了未知量、约束和计算结果，因此保留了足以沿着解题状态前进的外显草稿；只是把语义压缩为方程，不复述故事。

可以把一次请求抽象为：输入题目和 few-shot 示例 $`x`$ → 模型生成一串短草稿 $`d_1,d_2,\ldots,d_k`$ → 生成分隔符和答案 $`y`$。CoD 没有规定固定的 $`k`$，也没有总预算截断；“每一步短”允许复杂题通过增加步骤继续推理。与 CoT 相比，变化的是每个 $`d_i`$ 的语言展开程度，与 Standard 相比，变化的是保留了中间状态。作者的 per-step 约束因此不同于 CCoT 的固定全局预算和 TALE 的先估计全局预算：后两者需要预算选择/额外调用或可能违约，而 CoD 允许步骤数随任务增长；这是作者在 Related Work 中的定位，尚未由本论文直接做预算公平实验验证。

## 与相关机制的边界

CoT 是 CoD 的直接对照：两者都用自然语言中间轨迹和 `####` 答案抽取，区别主要是详细叙述与稠密草稿。Standard/direct answer 则绕过外显推理，token 少但在作者选定的多步任务上准确率低。树/图式思维、self-consistency、ReAct 增加候选、验证、分支或工具访问，解决的是搜索拓扑、反思和外部信息问题；CoD 本身没有分支、投票、工具或纠错机制。

SoT 先生成答案骨架并并行解码，目标主要是降低感知/生成延迟；CoD 通过少生成 token 直接降低输出量和相应成本，作者认为二者可组合。Draft & Verify 在较少层上先产生低质量 draft，再用一次前向验证，属于模型推理加速；CoD 只改 prompt，未报告与 speculative decoding 的组合结果。Coconut 把推理搬到连续 latent space，可能减少自然语言解码，但会损失自然语言可解释性且对复杂 GSM8K 结果较弱，不能直接用于 GPT/Claude 黑盒；CoD 保持可见文本草稿。CCoT 采用固定全局 token budget，TALE 动态估计全局 budget；CoD 的卖点是每步软上限、允许不定步数，但论文没有同一数据/模型下系统比较它们。

## 实验设计

作者沿用原 CoT 论文的三类任务，并挑选原 CoT 相对无推理 baseline 有明显提升的代表性数据：

| 类别 | 任务 | 数据/规模与构造 | 主要指标 |
|---|---|---|---|
| 算术 | GSM8K | 约 8,500 个小学数学题，含详细解答；正文未进一步说明本次评测拆分/样本数 | accuracy、平均输出 Token #、平均 latency |
| 常识 | BIG-bench date understanding、sports understanding | 正文只说明采用 BIG-bench 任务，未给本次子集大小 | 同上 |
| 符号 | coin flip | 原 CoT 数据未公开；作者按相同设计合成 250 个样本，每题从美国区域前 1000 名中随机选 4 个名字，随机决定每人是否 flip | 同上 |

评测两个闭源旗舰模型：`gpt-4o-2024-08-06` 和 `claude-3-5-sonnet-20240620`。每个任务比较 Standard、CoT、CoD。正文报告的是平均输出 token 数和平均端到端响应 latency；没有给输入 token、总 token（输入+输出）、美元账单、吞吐、硬件、并发、重复次数、随机种子或置信区间，也没有把首 token 延迟与完整响应延迟拆开。因此“成本下降”主要是由输出减少推断出的部署含义；表格直接测量的是输出 token 与 latency，输入 few-shot token 的减少只在 Discussion 中被定性声称。训练成本、FLOPs 和隐藏思考 token 均未报告。

公平条件方面，CoT 使用原论文 few-shot 例子，CoD 为每个 few-shot 例子增加作者手写短草稿；这使 CoD 的示例文本与 CoT 不完全相同。Standard 也使用 few-shot 输入输出对，但直接答。作者没有展示各任务完整 prompt、样本抽样细节或输出截断策略，所以表格适合读作作者报告的 prompt comparison，而不是严格控制所有输入 token 后的成本实验。

## 主结果

### GSM8K（表 1）

| 模型 | 提示 | Accuracy | 输出 Token # | Latency |
|---|---|---:|---:|---:|
| GPT-4o | Standard | 53.3% | 1.1 | 0.6 s |
|  | CoT | 95.4% | 205.1 | 4.2 s |
|  | CoD | 91.1% | 43.9 | 1.0 s |
| Claude 3.5 Sonnet | Standard | 64.6% | 1.1 | 0.9 s |
|  | CoT | 95.8% | 190.0 | 3.1 s |
|  | CoD | 91.4% | 39.8 | 1.6 s |

CoT 相对 direct answer 把两模型准确率提高到 95% 以上，但输出约 190--205 tokens。CoD 用约 40--44 tokens，准确率约 91%，仍明显高于 Standard；相对 CoT，输出减少约 80%。GPT-4o 的延迟从 4.2 s 降到 1.0 s，Claude 从 3.1 s 降到 1.6 s。作者正文概括为平均 latency 分别下降 76.2% 和 48.4%。代价是 GSM8K 上 CoD 比 CoT 低 4.3 个百分点（GPT-4o）和 4.4 个百分点（Claude），所以这里是大幅效率-准确率折中，不是无条件胜出。

### BIG-bench date understanding（表 2）

| 模型 | 提示 | Accuracy | 输出 Token # | Latency |
|---|---|---:|---:|---:|
| GPT-4o | Standard | 72.6% | 5.2 | 0.6 s |
|  | CoT | 90.2% | 75.7 | 1.7 s |
|  | CoD | 88.1% | 30.2 | 1.3 s |
| Claude 3.5 Sonnet | Standard | 84.3% | 5.2 | 1.0 s |
|  | CoT | 87.0% | 172.5 | 3.2 s |
|  | CoD | 89.7% | 31.3 | 1.4 s |

GPT-4o 上 CoD 比 CoT 低 2.1 个百分点，但少约 60% 输出 token、延迟少 0.4 s；Claude 上 CoD 反而比 CoT 高 2.7 个百分点，同时把 172.5 压到 31.3 tokens、3.2 s 压到 1.4 s。这里说明 CoD 的优势不是固定的准确率提升，而是模型/任务依赖的 Pareto 位置变化：同一提示在不同模型上可能略损失或略增准确率。

### BIG-bench sports understanding（表 3）

| 模型 | 提示 | Accuracy | 输出 Token # | Latency |
|---|---|---:|---:|---:|
| GPT-4o | Standard | 90.0% | 1.0 | 0.4 s |
|  | CoT | 95.9% | 28.7 | 0.9 s |
|  | CoD | 98.3% | 15.0 | 0.7 s |
| Claude 3.5 Sonnet | Standard | 90.6% | 1.0 | 0.9 s |
|  | CoT | 93.2% | 189.4 | 3.6 s |
|  | CoD | 97.3% | 14.3 | 1.0 s |

两模型 CoD 都超过 CoT：GPT-4o 为 98.3% 对 95.9%，Claude 为 97.3% 对 93.2%。输出也更短，尤其 Claude 从 189.4 到 14.3，作者计算为 92.4% 减少；延迟从 3.6 s 到 1.0 s。作者把这看作 CoT 在某模型上过度解释、CoD 删除无助于答案的细节的例子，但“过度解释导致准确率下降”的因果解释仍是作者分析，而表格本身只给相关结果。

### Coin flip（表 4）

每题从一个 heads 状态开始，按四个名字的 flip/do-not-flip 操作判断末态。例如 Robyn、Peggy、Grant flip，Vanessa 不 flip，答案是 No。Standard 的准确率为 GPT-4o 73.2%、Claude 85.2%；CoT 与 CoD 均为 100%。

| 模型 | 提示 | Accuracy | 输出 Token # | Latency |
|---|---|---:|---:|---:|
| GPT-4o | Standard | 73.2% | 1.0 | 0.4 s |
|  | CoT | 100.0% | 52.4 | 1.4 s |
|  | CoD | 100.0% | 16.8 | 0.8 s |
| Claude 3.5 Sonnet | Standard | 85.2% | 1.0 | 1.2 s |
|  | CoT | 100.0% | 135.3 | 3.1 s |
|  | CoD | 100.0% | 18.9 | 1.6 s |

在任务饱和到 100% 后，CoD 保持 CoT 的准确率，却把输出压缩 68%（GPT-4o）和 86%（Claude，按作者概括），延迟也下降。这里合成集只有 250 题，且任务规则简单，不能外推到更复杂的符号搜索。

### 图 1：四任务的准确率/输出量概览

![Comparison of Claude 3.5 Sonnet accuracy and token usage across GSM8K, Date, Sports, and Coin Flip for Standard, CoT, and CoD.](../../../assets/papers/2502.18600/fig1_plot.png)

图 1 是作者源文件中的 `plot.png`，本地复用文件为 [`assets/fig1_plot.png`](../../../assets/papers/2502.18600/fig1_plot.png)。上半图 y 轴是 Accuracy (%)，横轴依次 GSM8K、Date、Sports、Coin Flip；每组蓝/黄/红分别是 Standard/CoT/CoD。下半图 y 轴是 Token Count，展示 CoT 的柱子通常远高于 CoD，尤其 Sports。图的数值是 Claude 3.5 Sonnet 汇总，和表 1--4 的 Claude 行对应；它没有画 latency，也没有误差条。图中 Coin Flip 的 CoT 与 CoD accuracy 都在 100%，而 token 量仍有明显差距，直观显示“相同正确率、不同生成负担”。

## 正文中的限制、失败与负结果

### 没有 few-shot 时不稳定（表 5）

作者在 GSM8K 做 zero-shot，即去掉所有 few-shot 示例。结果表明 CoD 的有效性显著下降：

| 模型 | 提示 | Accuracy | 输出 Token # | Latency |
|---|---|---:|---:|---:|
| GPT-4o | Standard | 56.9% | 2.2 | 0.5 s |
|  | CoT | 94.8% | 278.4 | 8.1 s |
|  | CoD | 84.4% | 76.4 | 2.6 s |
| Claude 3.5 Sonnet | Standard | 61.9% | 5.2 | 0.9 s |
|  | CoT | 90.4% | 248.8 | 3.5 s |
|  | CoD | 65.5% | 73.7 | 1.6 s |

CoD 仍比 direct answer 高（GPT +27.5 个百分点，Claude +3.6 个百分点），但都显著低于 CoT；输出也只是从 248.8/278.4 降到约 74/76，而不是 few-shot 条件下的约 40。作者的解释是模型训练数据中可能缺乏“短而有洞见”的 CoD 轨迹，因而没有示例时既不知格式，也不易自行找到压缩但充分的中间状态。这是合理假设，论文没有通过训练语料审计或控制实验验证。

### 小模型性能差距扩大（表 6）

GSM8K 上，少于 3B 参数的模型结果如下；此表没有 latency 列：

| 模型 | Standard Acc / Token # | CoT Acc / Token # | CoD Acc / Token # |
|---|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 5.7% / 6.6 | 32.5% / 141.4 | 24.2% / 75.1 |
| Qwen2.5-3B-Instruct | 7.2% / 3.4 | 59.1% / 236.4 | 43.1% / 41.2 |
| Llama 3.2-3B-Instruct | 3.9% / 16.6 | 70.7% / 195.3 | 52.5% / 98.1 |
| Zoom-SLM-2.3B | 5.9% / 3.8 | 77.7% / 129.0 | 50.9% / 55.6 |

CoD 的确比 Standard 高出 18.5、35.9、48.6、45.0 个百分点，但相对 CoT 分别低 8.3、16.0、18.2、26.8 个百分点。也就是说短轨迹仍能节省输出，但小模型更依赖完整 CoT 结构来获得准确率；作者再次把原因归于训练中缺少 CoD 风格数据，并提出用 CoD 格式数据微调，正文没有实际微调结果。

### 未纳入正式正文的消融线索

LaTeX 源目录含 `ablation.tex`，标题列出“Total Budget vs Per-Step Budget”“How Many Words Do We Need”和“CoD on Different Model Sizes”，但主 `paper.tex` 对 `\\input{sections/ablation}` 已注释，固定 PDF 与文本没有任何对应数据、表或结论。因此不能据此声称作者已经测过“5 词是否最优”、全局预算对比或 8B/70B/405B 曲线。它只能作为作者计划/未编译材料的候选线索，待主代理决定是否另行核对仓库历史。

## token、成本与延迟到底测了什么

论文表格中的 `Token #` 是**平均 response 输出 token 数**；它不是输入+输出总 token，也不是隐藏思考 token、FLOPs 或美元成本。少量 Standard 行接近 1 token，是因为要求直接答；CoT/CoD 行包含中间轨迹和 `####` 后答案。CoD 的输出压缩来自生成内容本身，且 few-shot 手写草稿会改变输入 prompt 长度；作者没有报告输入 token，因此无法从表格精确计算每请求总 token 节省比例。

Latency 是表格给出的平均响应时间，单位秒。它随输出量大体下降（例如 Claude sports 3.6→1.0 s），但并非输出 token 比例的精确线性函数（网络、服务端排队、模型系统和首 token 开销都可能影响）；论文没有报告测量协议、并发、warm-up、重复次数或 P50/P95。于是“76.2%/48.4% latency reduction”是该实验配置下的均值比较，不是硬件无关的理论保证。训练开销未涉及，部署成本结论也没有按输入价格和输出价格分开核算。

## 机制上的可用结论与应保持的谨慎

作者数据支持一个清楚的 prompt-level 结论：对强闭源模型、带作者提供的短草稿 few-shot、选定的多步任务，外显推理不需要写成长篇自然语言；保留关键中间量并压缩表达，通常能显著减少输出 token 和延迟，并在若干任务上保持 CoT 准确率。它特别适合输出本来就不需要展示完整推理、但服务端仍希望借助中间状态完成任务的场景。

不能把它升级成“短思考普遍等价于深思考”。GSM8K 上 CoD 比 CoT 低约 4 个百分点；zero-shot 和小模型结果更差；只有四类任务（其中 coin flip 是作者合成的 250 题），模型、提示和测量协议也有限。短文本可能提高可读性，也可能删除否定、单位或条件，从而语义改变；论文没有做可解释性或事实保持的系统评估。CoD 的关键先决条件是模型能把必要状态压进短草稿，这一能力显然依赖模型规模、训练分布和 few-shot 示例。

## 与推理时 token 效率研究的关系

这篇工作最直接的研究启示是把“推理质量”与“输出文字量”拆开：可把每步草稿看成有限带宽的外部工作记忆，实验则测量在该带宽下模型是否仍能维持任务准确率。后续研究若要把 CoD 变成可靠的 token-efficiency 方法，应同时记录输入、输出和总 token，区分显示草稿与隐藏计算，报告预算曲线而不是单点，并在相同 few-shot 内容、模型、并发和延迟协议下比较 Standard/CoT/CoD。还需专门测试长链、否定/单位敏感题、需要回溯或工具调用的任务，以及是否能用 CoD 轨迹微调小模型。

对部署者而言，可把 CoD 视为一种可调的软控制：短草稿带来速度/成本优势，但复杂度升高时模型可以增加步骤；若业务必须暴露完整可审计推理，CoD 的压缩轨迹可能不足，应把它看作工作草稿而非完整证明。对实验者而言，“5 词提示”不能当作硬性的 5-token 上限；真正可追踪的变量是每步文字限制、实际输出 token、答案准确率和端到端 latency 四者的联合关系。

## 有定位的局限

1. **实验覆盖有限**：主文只有 GSM8K、BIG-bench 两项常识任务和 250 题合成 coin flip；未报告更长链、工具使用、多语言、开放式生成或现代 reasoning model。
2. **成本口径不完整**：报告平均输出 token 和 latency，未报告输入/总 token、FLOPs、价格、并发、硬件、重复次数、方差、尾延迟或隐藏计算。
3. **比较存在 prompt 内容差异**：CoD 的 few-shot 示例由作者额外手写短草稿，不能仅凭表格把所有差异归因于“5 词”这一句提示。
4. **泛化受 few-shot 与规模限制**：zero-shot 及小模型表明 CoD 不是无条件稳定；作者的训练分布解释和 CoD 微调建议尚未被本论文实验验证。
5. **固定版本缺少消融**：源目录的 `ablation.tex` 未被主文编译，无法确认词数上限、全局/逐步预算或模型尺寸的系统结论。

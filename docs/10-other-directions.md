# 10 相邻方向：少 token、少前向和少计算不是同一个问题

[返回目录](../README.md) · [上一章：评测](09-evaluation.md)

相邻方向必须纳入视野，因为它们可能改变最好的模型或系统选择；但不能把不同单位的节省混进同一条 token 曲线。特别是隐状态、字节、视觉 patch 与扩散步骤改变后，**不显示文字并不意味着没有进行计算**。

![Token、计算和时间的区别](../assets/plots/harness-resource-metrics.svg)

保留的示意图只说明计量区别，没有把不同方法排成实验榜单。

## 1. 路线地图与本项目的位置

| 分支 | 代表方法 | 相对前作改变什么 | 应同时报告的单位 |
| --- | --- | --- | --- |
| 精确投机解码 | Speculative Decoding→EAGLE 系列/MTP | 更便宜地提出多 token 草稿，目标模型批量验证 | 最终 token、draft/target 计算、延迟、吞吐 |
| 连续 CoT | Coconut→CODI | 用连续状态替换文字，再用蒸馏恢复能力 | 文本 token、latent 步、FLOPs、任务分数 |
| 循环深度 | Huginn/Recurrent Depth | 在每个生成位置内部多算几轮 | recurrence、FLOPs、最终序列长度 |
| 字节与 patch 表示 | Tokenizer Choice→BLT | 改变编码与大模型计算粒度 | bytes、patch、实际模型 token |
| 扩散式生成 | LLaDA/Dream 等 | 用迭代去噪替代标准逐 token 生成 | 输出长度、去噪步数、处理位置总数、延迟 |
| 视觉压缩 | FastV、LLaVA-PruMerge | 在 LLM 内剪枝，或在视觉编码器末端选择并合并 | 内部视觉 token、图像费用、OCR/GUI 成功率 |
| 每 token 的系统成本 | 量化、MoE、MLA/KV、缓存 | 降精度、少激活参数、少存/重复算历史 | 显存、FLOPs、时间、费用；逻辑 token 另记 |

[原文记录 A01–A06](../sources/adjacent-map.md)、[Coconut/CODI](../sources/posttraining-map.md)、[BLT/DeepSeek-V3](../sources/early-stages.md)。最后一行是范围边界与基础技术索引，本章不声称重新审计了全部量化/KV 文献。

## 2. 投机解码：保持输出分布，改变计算日程

标准投机解码让小模型草拟一段，大模型并行验证，借助采样校正保持目标模型的输出分布。EAGLE 系列改进草稿产生方式，EAGLE-3 融合多层特征并在训练时模拟草稿的自反馈，减少多步错误。[Speculative Decoding](https://arxiv.org/abs/2211.17192v2)、[EAGLE-3](https://arxiv.org/abs/2503.01840)

这条路线有很强的“不会因为换解码流程就主动缩短答案”的边界：在精确条件下目标分布相同，最终输出长度分布也相同。减少的是串行大模型调用和延迟，可能增加总算术。它可以和短链方法组合，一个减少要生成的 token，另一个让剩余 token 更快产生。

## 3. 连续推理与循环深度：把计算放在文字之外

Coconut 把 hidden state 反馈为后续输入，用课程逐步替代 CoT；CODI 将显式教师与连续学生的状态对齐，试图弥补直接替换造成的能力损失。Coconut 在合成逻辑任务更有利，但 GSM8K 的 GPT-2 实验低于显式 CoT。CODI 恢复了更多能力，证据仍主要来自小模型和固定 latent 长度。[Coconut](https://arxiv.org/html/2412.06769v1)、[CODI](https://arxiv.org/html/2502.21074v3)

Huginn 的重点则是 recurrent core：不必把内部计算写成额外文字，可以增加隐藏状态更新次数。这与“把整段 CoT 压成少量向量”的训练方式不同；二者都需要专门架构/训练或白盒操作。它们为超大预算下的能力扩展提供可能，但没有自动免除推理计算成本。[Recurrent Depth](https://arxiv.org/html/2502.05171v2)

对这类方法，只画可见文字 token 轴会偏向把计算隐藏起来的系统。至少给两张图：任务质量—文本 token，以及任务质量—总计算/时间；同时标明 latent 步。不能据此声称闭源 compaction 就采用了同样内部表示。

## 4. 扩散、字节和视觉压缩各自的真实收益

Dream 从自回归基座适配离散扩散，以去噪步数控制生成计算；更少步骤可能更快，也可能降低质量。继承 Qwen 权重后使用的继续训练量不能与从头模型的全部训练量直接相除。扩散生成 1,000 个位置、做 20 次去噪，并不等于只花 20 个 token。[Dream](https://arxiv.org/html/2508.15487v1)

BLT 用动态字节 patch 分配计算，代表“相同内容换表示”，详细比较见[02 章](02-pretraining.md)。FastV 则在浅层后剪掉部分视觉 token，保留更有用位置；这对冗余图像有利，对 OCR、细小控件等任务可能丢关键信息。[FastV](https://arxiv.org/abs/2403.06764v3)

LLaVA-PruMerge 在视觉编码器末端选择重要 patch，再把相近的被删位置合并进去；PruMerge+ 用空间采样保护覆盖。它与 FastV 的剪枝位置和信息保留方式不同。原文7B主表中约5.5%的保留率使 VQAv2 从78.5降到72.0，25%的 PruMerge+ 恢复至76.8；强压缩并非无损。主表经过 LoRA 微调，不能统称零训练。[PruMerge §3–4](https://arxiv.org/abs/2403.15388v6)

对于电脑使用 Agent，裁剪截图、视觉 token 剪枝和减少截图轮数属于不同变量。只有底层接口真的少处理/少计费，或整项任务少调用，才可能改变部署账。内部视觉 token 减半不等于平台账单中图像 token 自动减半。

## 5. 代表实验与证据状态

| 工作 | 原实验 | 结果的实际含义 |
| --- | --- | --- |
| 投机解码 | T5-XXL/小 T5、WMT、单 TPU-v4、batch1 | 约 2.6× sampling 加速；最终输出 token 不变 |
| EAGLE-3 | 8B、SGLang/H100、batch1 | 158.34→373.25 token/s；不同 batch 收益不同 |
| Coconut | GPT-2/GSM8K 与合成逻辑 | 不同任务有正负结果，latent 步仍算计算 |
| Huginn | 3.5B、约0.8T训练，改变 recurrent depth | 内部多算可增强能力；未给完整任务 token 节省曲线 |
| FastV | LLaVA-13B、K=2/剪50%，A40 A-OKVQA | 理论 FLOPs 与延迟下降；细粒度视觉能力需另测 |

精确投机解码的分布保持有算法条件下的理论保证；真实加速依赖硬件。连续/循环推理的普适优势尚无同等强度证据。视觉压缩存在冗余利用空间，但 GUI/OCR 的必要细节限制很实质。

## 6. 阅读顺序与机会判断

先读投机解码建立“序列不变也能更快”的边界，再读 Coconut→CODI→Huginn 比较计算表示；最后按应用需要读 Dream、BLT 或视觉压缩。

| 初步 idea | 最近邻 | 要防止的重复/误判 |
| --- | --- | --- |
| 用隐向量代替全部 CoT | Coconut、CODI、Recurrent Depth | 必须同时比较计算、能力与任务泛化 |
| 少做扩散步骤就省 token | Dream 等 dLLM | 少步骤是计算策略，最终输出长度另记 |
| 删掉无关截图 patch | FastV、视觉剪枝/合并 | GUI 小目标和 OCR 错误可能增加更多重试 |
| 用量化/MoE 降成本 | 现有服务与模型配方 | 每 token 更便宜不能冒充同性能更少 token |

本地图的机会判断：小团队优先复用服务引擎的投机、量化和缓存能力；把主要研究投入放在任务层长度、状态或调用决策。新架构可以作为中长期方向，先用公开小模型做机制证据，避免从头重建庞大基础设施。

更完整的单位算例和基础说明保留在[相邻技术专题](technical/10-other-directions.md)。

<a id="第-10-章相邻方向更快更小与更少-token-分别意味着什么"></a>
<a id="101-先分开四种问题"></a>
<a id="102-推测解码让大模型一次验证多个候选"></a>
<a id="103-latent-reasoning不写成文字也仍然需要计算"></a>
<a id="104-扩散语言模型少轮数不等于少工作"></a>
<a id="105-量化降低表示精度不改变-token-的定义"></a>
<a id="106-moe每个-token-只激活部分专家"></a>
<a id="107-动手三个计量错觉"></a>
<a id="108-用一张比较表约束研究结论"></a>
<a id="109-练习与参考答案"></a>
<a id="延伸阅读"></a>

旧版小节链接已保留。原来的推导、算例和练习见[本章技术专题](technical/10-other-directions.md)。

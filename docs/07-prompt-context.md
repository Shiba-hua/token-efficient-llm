# 07 Prompt 与上下文：少读哪些内容，怎样仍保留关键状态

[返回目录](../README.md) · [上一章：RLVR](06-rlvr-agentic.md) · [下一章：Harness 与多智能体](08-harness-agents.md)

上下文路线适合小团队的重要原因是：很多实验可以固定模型，只改变送入模型的信息。真正的目标是**减少整项任务的反复读入，而不因遗漏信息增加错误和重试**。这里区分提示行为、检索、硬/软压缩、摘要、记忆与内部计算选择。

## 1. 路线怎样分化与组合

| 分支 | 代表方法 | 机制 | 主要取舍 |
| --- | --- | --- | --- |
| 改变行为指令/示例 | 简洁提示、TALE、DSPy、GEPA | 指定预算或优化提示程序 | 可能加长提示；优化阶段成本另计 |
| 先选相关材料 | 检索、RECOMP 的选择性增强 | 只取相关证据，无用时不补充 | 检索漏召回与额外调用 |
| 硬压缩输入 | LLMLingua→LongLLMLingua→LLMLingua-2 | 从困惑度删词发展到 query-aware 和监督分类 | 压缩率、保真、压缩器开销 |
| 连续表示压缩 | AutoCompressors、ICAE | 将原文编码成少量连续 summary/memory slots | 需要训练和内部接口，压缩器也要读全文 |
| 管理旧观察 | Observation masking、摘要、混合窗口 | 不重读旧日志，或压成摘要 | 可能丢失失败信息并延长任务 |
| 形成可回读记忆 | ReadAgent、AgentFold、compaction | gist/外部材料、主动折叠、窗口压缩 | 压缩本身和回读成本，长程一致性 |
| 减少内部计算 | LazyLLM、KV/token 选择 | 文本保留，但部分位置少参与计算 | 白盒能力，属于相邻计算效率 |

[来源 H01–H06、H12–H16](../sources/context-agents-map.md)。实际系统常组合多条路线，但组合后的净收益需重新测量。

![上下文处理流程](../assets/plots/context-pipeline.svg)

保留流程图表示筛选、压缩与输入之间的关系；它不是所有任务必须依次执行的流水线。

## 2. LLMLingua 系列的改进不是同一压缩器改名

LLMLingua 用小语言模型的困惑度估计重要性，结合粗到细预算和迭代删除，使目标模型能从稀疏文本恢复意义。LongLLMLingua 加入问题相关性、文档重排和动态比例，重点解决长检索材料中关键证据位置不佳的问题。[LLMLingua](https://aclanthology.org/2023.emnlp-main.825/)、[LongLLMLingua](https://aclanthology.org/2024.acl-long.91/)

LLMLingua-2 则质疑“难预测的 token 就重要”的替代指标，用 GPT-4 的抽取式保留标签训练双向分类器。它改变了重要性的监督方式，也降低了压缩器运行开销。三者共同研究硬压缩，但问题相关压缩每个查询需要重算；任务无关压缩较利于复用，细节却可能不符合当前问题。[LLMLingua-2](https://aclanthology.org/2024.findings-acl.57/)

RECOMP 将选择和压缩放到检索增强中：可抽句、可生成摘要，也可判断本次检索无益而返回空串。它说明“什么都不添加”也是有效动作；同时，摘要可能遗漏多跳推理的中间证据。[RECOMP](https://arxiv.org/html/2310.04408v1)

**软压缩改变的是输入表示。** AutoCompressors 让语言模型将文本片段编码成连续 summary vectors，再供后续片段使用；ICAE 用 LoRA encoder 压成 memory slots，冻结的 LLM 读取这些状态。前者偏向长文语言建模与可积累摘要，后者加入重建/续写和响应导向训练。它们不同于 Coconut 的输出推理状态，也不能直接送进只接受普通文本的 API。[AutoCompressors](https://arxiv.org/abs/2305.14788v2)、[ICAE](https://arxiv.org/abs/2307.06945v4)

ICAE 的512→128 slots设置使续写PPL从9.01变为9.50，展示压缩也有信息代价。AutoCompressors 的摘要可缓存，但累计摘要仍会增长。文本变成向量后，压缩器的全文读取和后续计算应另计，不能用“少了可见 token”隐藏这部分成本。

## 3. 遮蔽、摘要和可回读记忆为何需要正面对照

遮蔽通常保留模型动作和近期关键状态，只去掉较旧的大段观察；摘要用额外模型调用把历史改写为短文本。The Complexity Trap 在软件任务中发现简单遮蔽可与摘要竞争，而摘要有时让任务轮数增加。较短的单次输入不是完整成本已经降低的证明。[原论文](https://arxiv.org/html/2508.21433v3)

ReadAgent 通过分页 gist 保留概要，缺细节再回原页；AgentFold 让模型主动决定哪些历史可以细粒度折叠、哪些可跨步合并。两者把压缩看成长期状态管理，而非一次性删除。若后面需要回读很多次，或 gist 构建只服务一个任务，原有节省可能缩小。[ReadAgent](https://arxiv.org/html/2402.09727v3)、[AgentFold](https://arxiv.org/abs/2510.24699)

这三类方法可组成递进基线：固定窗口遮蔽→摘要→保留可回读出处的记忆→学习何时压缩。新方法必须说明比便宜的前一级多解决了什么，而不是只增加一个记忆模块。

## 4. 自动提示优化与厂商实践

DSPy 把提示/示例/调用模块表示成可优化程序；GEPA 从执行轨迹和评分中反思、修改与组合提示。它们能固定模型权重，减少人工调提示，但优化 rollout 和部署消耗是两本账。GEPA 相对某些提示基线的输入长度下降也不能代替完整输出与多次调用测量。[DSPy](https://arxiv.org/abs/2310.03714)、[GEPA](https://arxiv.org/pdf/2507.19457v2)

OpenAI 公开提供 reasoning effort、compaction、按需 tool search；Anthropic 公开讨论 just-in-time retrieval、compaction、外部笔记，并提供工具发现/程序调用。它们说明这些机制已经进入产品，但没有公开足够信息证明所有私有实现与论文方法相同。[OpenAI 文档](https://developers.openai.com/api/docs/guides/compaction)、[Anthropic 工程说明](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)、[企业证据](../sources/vendor-practice.md)

缓存则复用已处理前缀的计算：输入仍在逻辑 token 账中，只是计算和费率不同。LazyLLM 也属于内部计算选择，不能把 2× TTFT 改善记成输入 token 减半。[缓存文档](https://developers.openai.com/api/docs/guides/prompt-caching)、[LazyLLM](https://arxiv.org/html/2407.14057v1)

## 5. 关键实验与目前证据

| 工作 | 条件与结果 | 原口径和盲区 |
| --- | --- | --- |
| LLMLingua-2 | MeetingBank 指定设置，分数 87.75→86.92，输入 3,003→970 | 输入压缩，有轻微质量损失；不同模型的 QA F1 不能混用 |
| RECOMP | Flan-UL2/NQ，660/39.39 EM→37/36.57 | 高压缩明显，但并非同分数；多跳摘要存在忠实性失败 |
| Complexity Trap | Qwen3-Coder-480B/SWE-bench Verified，raw53.4%/$1.29、mask54.8%/$0.61、summary53.8%/$0.64 | 每实例美元，按特定价格后处理；不是同百分比原始 token |
| ReadAgent | PaLM 2-L/QuALITY，单页回读约省 25.4% words | 按 word、含 gist 摊销假设；不能标成 token |
| AgentFold | 30B-A3B/BrowseComp 长轨迹，上下文增长得到控制 | 单轮上下文长度不是累计总输入，仍有任务失败 |

Complexity Trap 的 Gemini 2.5 Flash thinking 设置还有反例：成功率40.4%→mask36.4%/summary31.4%，费用虽下降，质量也下降。Qwen的费用按不区分缓存命中的API价格后处理，Gemini则使用实际服务成本，不能混同。

多种论文和两家厂商都支持减少无关上下文具有实际价值；具体压缩器、摘要策略和阈值仍依赖任务。尤其没有形成“摘要普遍优于遮蔽”或“压得越多越好”的结论。

## 6. 阅读顺序与可检验机会

先读 LLMLingua→LongLLMLingua→LLMLingua-2，再读 RECOMP；需要白盒连续表示时读 AutoCompressors→ICAE；长任务读 Complexity Trap→ReadAgent→AgentFold；自动提示方向读 DSPy→GEPA。

| 初步 idea | 最近邻 | 可构成差异的问题 |
| --- | --- | --- |
| 每隔若干步总结历史 | 摘要窗口、Complexity Trap、compaction | 哪类状态必须原样保留，怎样发现摘要导致的错误循环？ |
| 先概要，再按需回读 | ReadAgent、RECOMP | 回读策略能否用任务验证信号校准，而不是只靠语义相似？ |
| 训练一个关键 token 分类器 | LLMLingua-2、TokenSkip | 重要性是否随未来工具动作改变，怎样控制跨域损失？ |

本地图的机会判断：围绕可执行工具任务，比较保留失败信号的状态压缩与现成遮蔽/摘要基线，按整个任务计费。这条路线更容易验证“同样成功、少读历史”，但需要证明没有把成本转移成更多轮次。

代码和回本算例见[上下文技术专题](technical/07-prompt-context.md)、[context_lab.py](../examples/context_lab.py)。

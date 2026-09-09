# Prompt、上下文和智能体：原文阅读记录

阅读日：2026-09-09。Luna 阅读；主代理统一成本口径、核对承载关键结论的 S²-MAD、The Complexity Trap、RECOMP 与 Anthropic 官方原文。表格或摘要中的最大节省不自动代表平均或完整任务收益。

## S05 — Prompt Compression for Large Language Models: A Survey

[Li 等，NAACL 2025](https://aclanthology.org/2025.naacl-long.368/)。已读方法分类、注意力/PEFT/多模态与局限。入口为硬压缩/软压缩；沿引用追到 LLMLingua 系列。硬压缩删除文本与白盒模型内部剪枝必须区分。

## S06 — Toward Efficient Agents

[Yang 等，A Survey of Memory, Tool Use, and Planning，2601.14192v2](https://arxiv.org/html/2601.14192v2)，2026-07。Luna 已读 §1–7、表 1–2；主代理另读 v1 的分类、§6 和关键引用。记忆/工具/规划分类补足纯 reasoning 综述的盲区；成本包括 token、金额、时间与交互，正文只在相同口径下比较。

## S07 — Large Language Model Based Multi-agents: A Survey of Progress and Challenges

[Guo 等，IJCAI 2024](https://www.ijcai.org/proceedings/2024/890)。用于发现角色、通信拓扑、协作组织和评测分支；不以综述代替任何多智能体节省实验。

## H01 — LLMLingua

[Jiang 等，EMNLP 2023](https://aclanthology.org/2023.emnlp-main.825/)。已读 §3–4、表 2 与附录；由 S05 进入。

小模型困惑度评分结合粗到细预算分配和迭代删除，目标是让目标模型仍能读懂稀疏输入。GSM8K 特定 ICL 配置中，full-shot 78.85 EM/2,366 prompt tokens，强压缩配置仍可接近原分数。类型 T；收益高度依赖被压缩的是冗长示例还是必要事实，压缩器计算另外发生。

## H02 — LongLLMLingua

[Jiang 等，ACL 2024](https://aclanthology.org/2024.acl-long.91/)。已读 §3–6、NQ/LongBench 表格与局限；由 LLMLingua 前作追读。

在长检索上下文中加入 query-aware 选择、文档重排、动态比例与恢复机制。GPT-3.5/NaturalQuestions 的特定证据位置实验以约四分之一输入取得更好准确率。类型 T；每个新问题需再次压缩，查询相关排序与缓存复用存在取舍。LooGLE 的 94% 费用下降属于当时价格口径，不能写成同百分比 token 下降。

## H03 — LLMLingua-2

[Pan 等，ACL Findings 2024](https://aclanthology.org/2024.findings-acl.57/)。已读数据蒸馏、双向分类器、实验与质量过滤；由 H01/H02 追读。也见 [E09](early-stages.md#e09--llmlingua-2)。

从无监督困惑度转向 GPT-4 抽取标签监督，训练便宜的 token 分类器。MeetingBank 表中原文 87.75/3,003 tokens，压缩后 86.92/970；**这与 E09 的 Mistral-7B QA F1 实验属于不同模型/设置**。类型 T。语义保真仍受压缩率、标签和分布外输入影响，两个读者发现的不同结果不能混成一组比较。

## H04 — RECOMP

[Xu、Shi、Choi，2310.04408v1](https://arxiv.org/html/2310.04408v1)，2023-10。已读 §2–6、表 1–4；由 S06 记忆集成进入，主代理复核原文。

检索后先压缩再 prepend：抽取关键句，或生成面向下游模型的摘要；无用检索可返回空串。Flan-UL2 的 NQ：top-5 文档 660 tokens/39.39 EM，抽取压缩 37/36.57。类型 T；短得多但分数并非相同。多跳 HotpotQA 的摘要忠实性/完整性分析暴露遗漏与幻觉，因此它也是“摘要一定保留答案证据”的反例。

## H05 — LazyLLM

[Fu 等，2407.14057v1](https://arxiv.org/html/2407.14057v1)，2024-07。已读 §2–4、LongBench、表 1 与生成分析；由内部计算压缩分支进入。

在 prefill/decode 中根据注意力动态选择 token，允许后续恢复。Llama-2 多文档 QA 22.43→22.31，TTFT 加速 2.34×。类型 C，输入文本未变短，是模型内部 token 计算/KV 选择。需要白盒 attention 访问；压缩前处理也可能使某些摘要任务更慢。

## H06 — The Complexity Trap

[Lindenbauer 等，Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management，2508.21433v3](https://arxiv.org/html/2508.21433v3)，2025-10。已读 §2–5、主表、轨迹与附录；主代理复核。由 S06 的长程上下文分支进入。

比较保留动作/推理而遮蔽旧观察、LLM 摘要和混合策略。SWE-bench Verified，Qwen3-Coder-480B：raw53.4%/$1.29、mask54.8%/$0.61、summary53.8%/$0.64每实例；Gemini 摘要轨迹平均 52 turns，mask 44。类型 C，主成本是每实例美元，不能冒充原始 token 比例。窗口阈值和 scaffold 改变结论；摘要可能掩盖失败信号，较短上下文不保证较短任务。

## H07 — AgentPrune

[Zhang 等，Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent Systems，2410.02506](https://arxiv.org/abs/2410.02506)，ICLR 2025。已读图建模、优化、实验、表 4 和敏感性；由 S06 拓扑稀疏化进入。

学习空间/时间通信图的 mask，再剪掉冗余边；主要设置为多个 GPT-4、有限轮次，任务包括 MMLU/GSM8K/HumanEval。作者报告 token 下降 28.1–72.8%。类型 T；这是特定多代理基线上的范围，离线学习通信图成本和单代理强基线都应另比较。

## H08 — S²-MAD

[Zeng 等，Breaking the Token Barrier to Enhance Multi-Agent Debate Efficiency，NAACL 2025](https://aclanthology.org/2025.naacl-long.475/)，[PDF](https://aclanthology.org/2025.naacl-long.475.pdf)。已读 §3–5、表 1、阈值和失败分析；主代理复核原文。前作链：MAD→Sparse-MAD/GroupDebate→条件参与。

根据答案相似性决定谁继续发言、给谁看哪些不同意见，并允许共识早停。GPT-4/GSM8K：MAD 93.3%/50.4K total tokens，S²-MAD 94.2%/2.78K；这次计入输入与输出。类型 T；相似性阈值随任务变化，同意不保证正确，强重复采样/单代理预算对照仍必要。

## H09 — FrugalGPT

[Chen、Zaharia、Zou，2305.05176](https://arxiv.org/abs/2305.05176)，2023。已读成本模型、prompt adaptation、cascade、实验与限制；作为 H10 前作追读。

先尝试便宜模型，按可靠性阈值升级，并组合提示适配/缓存。HEADLINES 等任务报告显著美元节省。类型 C：级联可能产生更多总 token，只因调用更便宜而省钱。需要同分布训练样本，价格、任务和校准变化都会改变最优级联。

## H10 — RouteLLM

[Ong 等，2406.18665v4](https://arxiv.org/html/2406.18665v4)，ICLR 2025。已读 §3–6、路由/迁移实验；由模型级成本路线进入。

用偏好数据训练 strong/weak 路由器，阈值控制调用强模型比例；区别于逐级尝试，它通常先选择一个模型。MT-Bench/MMLU/GSM8K 主要报告美元—质量关系。类型 C；原始 token 不一定下降，分布外查询、模型对变更和路由器开销是边界。

## H11 — Anthropic 多智能体研究系统

[How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)，2025-06-13。已读架构、prompt/harness、失败与部署；主代理复核原文。

Opus 4 lead + Sonnet 4 workers 进行并行研究，lead 整合。内部评测报告优于单代理，同时多代理使用约普通 chat 的 15 倍 token。类型 V；揭示能力扩展与重复搜索/过度派工开销，不支持“多智能体天然更省 token”。内部任务与完整账目未公开。

## H12 — Anthropic context engineering

[Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)，2025-09-29。已读 just-in-time retrieval、compaction、note-taking、multi-agent。

用轻量引用、按需读取、外部笔记与阶段压缩控制上下文。类型 V/I：公开工程经验，没有受控的完整 token—性能数据；过度摘要与必要信息遗漏也是文章承认的问题。

## H13 — ReadAgent

[Lee 等，A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts，2402.09727v3](https://arxiv.org/html/2402.09727v3)，2024-07。已读 §2–4、附录 E/H；由 S06 分层记忆进入。

先分页并生成 gist，需要细节时回读原页。PaLM 2-L/QuALITY 的 1 页回读配置按全流程 words consumed 报告约 25.4% 节省，并假设 gist 在多题间摊销。不能把 words 直接标成 token；上下文容量扩展与实际调用总成本是不同轴。

## H14 — AgentFold

[Ye 等，Long-Horizon Web Agents with Proactive Context Management，2510.24699](https://arxiv.org/abs/2510.24699)，2025，ICLR 2026。已读 §3、表 1、图 3；由 S06 主动记忆管理进入。

模型学习何时作细粒度折叠或多步合并；30B-A3B 在网页任务训练评测。200 条 BrowseComp 轨迹分析中，100 turns 的上下文仍约 7K，但部分长轨迹被步数上限终止。类型 T/I：单次上下文的增长受控，不等于 100 次输入的总和小；缺少完整总 I/O 对照。

## H15 — DSPy

[Khattab 等，Compiling Declarative Language Model Calls into Self-Improving Pipelines，2310.03714](https://arxiv.org/abs/2310.03714)，ICLR 2024。已读 §1、§4、§6–7、表 1–2；沿自动提示/程序优化分支进入。

将模块的提示、示例、组合编译为可评测程序；示例自举和搜索提升 GPT-3.5/Llama2 在 GSM8K 等上的效果。类型 I，编译可能需要许多 rollout，且部署 prompt 未必变短。与后续 GEPA 的联系在于都优化模型外部程序，不要求更新基础权重。

## H16 — GEPA

[Agrawal 等，Reflective Prompt Evolution Can Outperform Reinforcement Learning，2507.19457v2](https://arxiv.org/pdf/2507.19457v2)，2026-02，ICLR 2026。已读 §1–5、表 1–2、预算和 prompt-length 附录；由 DSPy/MIPROv2 追读。

利用执行轨迹和自然语言反思修改提示，维护候选并组合。Qwen3-8B 多任务实验与 GRPO/MIPROv2 比较，优化 rollout 数显著较少；相对 MIPROv2 的 prompt 长度也有下降。类型 I/T（输入局部）；更少优化调用不等于更少部署总 token。不同模型上 merge 的收益不一致。

## H17 — ReAct

[Yao 等，Synergizing Reasoning and Acting in Language Models，2210.03629v3](https://arxiv.org/abs/2210.03629v3)，ICLR 2023。已读 §1–4、附录 E；作为编排前作与 AgentFold 基线追读。

交替推理、动作和观察，使模型能用环境反馈修正。PaLM-540B 的 HotpotQA/FEVER，以及 ALFWorld/WebShop，比较 CoT、Act 等；存在重复动作/无信息检索失败。类型 I：工具 Agent 基础范式，原文没有完整 token 节省实验。

## H18 — LLMCompiler

[Kim 等，An LLM Compiler for Parallel Function Calling，2312.04511v3](https://arxiv.org/html/2312.04511v3)，ICML 2024。已读 §3–5、附录；由 ReAct→依赖图执行进入。

Planner 建 DAG，调度就绪工具，必要时重规划。GPT-3.5/Llama2-70B，HotpotQA、Movie Recommendation、ParallelQA；表 2 中 HotpotQA 原始 ReAct 输入/输出约 2,900/120，LLMCompiler 1,300/80。类型 T/C；token 表对照原始 ReAct，其他结果常用修过循环/早停的 ReAct†，不能跨表拼为同一准确率—成本点。

## H19 — Self-Consistency

[Wang 等，Self-Consistency Improves Chain of Thought Reasoning in Language Models，2203.11171v4](https://arxiv.org/abs/2203.11171v4)，ICLR 2023。已读 §1–5、附录 A.2；由多路径搜索前作链进入。

同题采不同链、最终答案多数投票。PaLM-540B/GSM8K，CoT 56.5%→74.4%，主要设置 40 样本、重复评估。类型 I：增加生成路径换性能，没有完整 I/O 等预算节省证据。开放式答案难以投票，概率加权也不统一优于多数票。

## H20 — Tree of Thoughts

[Yao 等，Deliberate Problem Solving with Large Language Models，2305.10601](https://arxiv.org/abs/2305.10601)，NeurIPS 2023。已读 §2–4、三类任务及消融；沿 Self-Consistency→可搜索思维结构追读。

思维节点树配 BFS/DFS、候选扩展、价值评估和回溯。GPT-4、100 个 Game of 24 题，宽度5的 ToT 74%，CoT-SC(100) 9%；生成与评价预算不同。类型 I，未按完整输入+输出 token 等预算比较；crossword 等仍受评价器错误和搜索开销限制。

## H21 — Compute-optimal Test-Time Scaling

[Snell 等，Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters，2408.03314](https://arxiv.org/abs/2408.03314)，ICLR 2025。已读 §1、§3、§5–6、search/revision 实验。

按难度分配并行采样、顺序 revision、PRM 搜索；PaLM 2-S* 系列，数学任务。相对固定 best-of-N，报告更好的测试时计算效率。类型 I/C/T（生成预算）；完整输入重读与评价器调用不等于论文的单一预算轴。更难题可能需要更强基座，而非一直增加搜索。

## H22 — DEER

[Yang 等，Dynamic Early Exit in Reasoning Models，2504.15895v3](https://arxiv.org/abs/2504.15895v3)，2025-09。已读 §1–6、附录；由动态停止分支进入。

在推理转折或其他触发处试答，再以置信度退出；DEER-Pro 使用额外试答等缓解敏感性。R1-Distill-Qwen-7B/MATH-500 示例 87.4%/3,858→89.8%/2,143。类型 T：Tok 为生成推理长度；试答、输入重读与白盒实现成本应另核算，不能直接宣称全任务同比例节省。

## H23 — BATS

[Liu 等，Budget-Aware Tool-Use Enables Effective Agent Scaling，2511.17006v1](https://arxiv.org/html/2511.17006v1)，2025-11。主代理已读 §3–6、表 1–3；由 S06 的预算规划引用进入。

Budget Tracker 在工具反馈后提示剩余预算，BATS 再结合计划、验证、继续/重试。Gemini-2.5-Pro/BrowseComp：ReAct 预算100为12.6%、9.9美分；Tracker预算10为12.8%、6.8美分。类型 C/I；金额包括 token 和工具费用，预算上限缩十倍不等于实际消耗缩十倍。表3的其他基线部分来自文献，不能假定是完全受控重跑。

## H24 — AutoCompressors

[Chevalier、Wettig、Ajith、Chen，Adapting Language Models to Compress Contexts，2305.14788v2](https://arxiv.org/abs/2305.14788v2)，2023-11-04，EMNLP 2023。Luna 已读 §3–7、附录 A–F；交叉分类检查发现软压缩缺口后补读。

通过 summary tokens 生成连续摘要，累积给后续片段，随机分段并用语言建模训练。OPT1.3/2.7B用2B Pile，Llama2-7B LoRA用15B RedPajama；长文PPL、ICL与检索。6,144→150 vectors的OPT2.7B PPL为5.93域内/8.10域外；这是输入表示和特定PPL证据，不是全任务token结果。summary预计算有利复用，累积长度仍增长、存在二次复杂度，长上下文/OOD仍有损失。类型C/T（连续输入）。

## H25 — ICAE

[Ge 等，In-context Autoencoder for Context Compression in a Large Language Model，2307.06945v4](https://arxiv.org/abs/2307.06945v4)，2024-05-08，ICLR 2024。Luna 已读 §2–5、表1–7、附录A–D。

LoRA encoder压缩原文为memory slots，冻结decoder读入；先重建/续写，再用GPT-4生成PwC响应训练。Llama/Llama2-7B/13B，PwC240K训练/18K测试；512→128 slots使continuation PPL9.01→9.50。类型C/T（连续输入）；不改变最终答案长度，压缩器先读全文，缓存/FlashAttention条件影响净速度。不能外推为黑盒API压缩，也不等于latent CoT。

## 定向复核补记

Luna再读H06/H07/H08/H18，主代理复核原表并修正文稿：H06的Qwen费用是按API价后处理，Gemini thinking存在成功率下降；H18的表2美元为每千任务换算，token基线ReAct与部分其他表ReAct†不同。H08的GSM8K强节省对应GPT-4-0613、MAD(5,4)，MMLU有90.8→88.1下降。

H07表3 AutoGen/HumanEval 的输入492,273→315,105，输出130,196→139,714，分数85.41→86.65；不能把摘要28.1–72.8%的范围统称完整I/O降幅。GSM8K原表3,791,251/4,327,740约87.6%，却标59.9%，应按绝对数检查。多查询拓扑优化的前Q′题、角色生成和泛化代价不能漏计。上述核查没有重新执行这些实验。

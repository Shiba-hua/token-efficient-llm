# 后训练效率路线：原文阅读记录

阅读日：2026-09-09。Luna 阅读原文；主代理整合并复核 TokenSkip 表 2、DAST 表 1、System 2 蒸馏表 4、L1 短链比较表 与预算违约定义。T/I/C 的含义见[证据协议](README.md)。下文的不同论文数字不能横向排名；点估计近似相同不等于通过统计等效性检验。

## S03 — Stop Overthinking

[Sui 等，Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models，2503.16419v4](https://arxiv.org/html/2503.16419v4)，2025-08。已读 Introduction、RL/SFT、Latent Reasoning、方法表与引用。入口分类是 model-based / output-based / input-prompt；沿引文进入 P01–P12。主要关注输出推理长度，不能覆盖全部 Agent 输入重读与工具开销。

## S04 — A Survey of Efficient Reasoning for Large Reasoning Models

[Qu 等，Language, Multimodality, and Beyond，2503.21614v2](https://arxiv.org/html/2503.21614v2)，2025-12。已读效率定义、inference、SFT、RL、引用。它补上预算、并行搜索、模式选择和多模态方向；质量/成本比值与本库的等性能 Pareto 比较并不等价。用其分类发现文献，实验结论回到原文。

## P01 — Can Language Models Learn to Skip Steps?

[Liu 等，2411.01855v1](https://arxiv.org/html/2411.01855v1)，2024-11，NeurIPS 2024。已读 §3–5、OOD 分析；由 S03 跳步路线进入。

机制：用 cold/warm start 构造正确跳步路径，混合完整与压缩解做 SFT。Llama2-7B/Phi3-mini 在代数、加法、方向推理的合成任务中减少步骤；方向推理 OOD-hard 仍显著困难。主要计量是步骤数，不能直接兑换为 token。结构化任务证明可学某些捷径，不证明开放域都可省略中间计算。

## P02 — Distilling System 2 into System 1

[Yu、Xu、Weston、Kulikov，2407.06023v3](https://arxiv.org/html/2407.06023v3)，2024-07。已读 §2–4、§6；主代理复核表 4。由 S03/S04 行为蒸馏路线进入。

机制：同一模型先花更多推理求解，筛选结果，再只学最终输出；不属于学生轨迹上的 OPD。Llama2-70B-chat 上，若干简单操作/评判任务能内化昂贵流程；但 GSM8K 单次 CoT 为 52.77%/270 tokens，蒸馏后 7.13%/18。类型 T；这是删除显式推理后困难能力受损的强反例，不是所有蒸馏失败的结论。

## P03 — C3oT

[Kang 等，Generating Shorter Chain-of-Thought without Compromising Effectiveness，2412.11664v1](https://arxiv.org/html/2412.11664v1)，2024-12，AAAI 2025。已读方法、实验、分析；由 S03/S04 条件 SFT 引文进入。

机制：GPT-4 压缩参考 CoT，构造长短配对，训练条件标记控制长度。Llama2-7B/13B，GSM8K/MathQA/ECQA/StrategyQA；7B GSM8K 长链基线 37.38%，C3oT 36.92%，生成压缩约 56.67%。类型 T；有小幅准确率损失，依赖教师改写与旧模型，不能把标题中的“without compromising”当作所有任务的等效性证明。

## P04 — Self-Training Elicits Concise Reasoning in Large Language Models

[Munkhbat 等，2502.20122v3](https://arxiv.org/html/2502.20122v3)，2025-06，ACL Findings。已读 §2–5、附录 D.6；由两份综述的 shortest-correct 自训练进入。

机制：每题采多条，选最短正确路径，SFT 后部署单次生成；few-shot 帮助降低生成数据成本。Llama/Gemma/Qwen/DeepSeekMath 系列，在 GSM8K/MATH 评测，平均长度包含错误轨迹；论文报告约 30% 缩短，部分配置准确率下降，难题通常少压缩。类型 T；离线 N 次采样在长度近似固定时成本近似随 N 线性增长，不能漏掉这个前期成本。

## P05 — TokenSkip

[Xia 等，Controllable Chain-of-Thought Compression in LLMs，2502.12067v3](https://arxiv.org/html/2502.12067v3)，2025-09。已读 §2–4、附录；主代理复核表 2、训练与计时条件。

机制：用 LLMLingua-2 为正确 CoT 的 token 评分，删除低重要性文字，混合保留比例进行 LoRA SFT；部署按比例生成。Qwen2.5-14B-Instruct，GSM8K，greedy：原始 93.1%/313.11 CoT tokens，比例 0.7 为 93.4%/218.62。Llama3.1-8B 的高压缩在 MATH-500 上损失明显。类型 T；训练用两张 3090 的说明与单卡计时实验不能混为同一硬件条件。

## P06 — TALE

[Han 等，Token-Budget-Aware LLM Reasoning，2412.18547v5](https://arxiv.org/html/2412.18547v5)，2025-06。已读 §3–6、§8、附录 A；由两综述的预算预测/条件化分支进入。

机制：离线按正确性搜索可行预算；部署用预算预测提示，或以 SFT/DPO 内化。GSM8K 等任务；Llama3.1-8B 的 TALE-SFT 示例 77.56%/241.51→78.57%/139.63。GPT-4o-mini 的平均结果则伴随准确率损失。类型 T；离线搜索拥有答案，线上没有；预算预测自身与不遵守预算导致的额外输出也需要计量。

## P07 — O1-Pruner

[Luo 等，O1-Pruner: Length-Harmonizing Fine-Tuning for O1-Like Reasoning Pruning，2501.12570v2](https://arxiv.org/html/2501.12570v2)，2025-01。已读 §3–5、消融；由长度奖励分支进入。

机制：根据冻结参考模型在每题上的长度/正确性构造奖励，将缩短与改善解题平衡，用 PPO 更新。Marco-o1-7B/QwQ-32B，在 MATH 数据训练，评测 MATH/GSM8K/GaoKao；表2的Marco三集平均932→554 tokens、73.4→76.8%，是点估计同时改善，没有固定准确率匹配或统计等效性检验。类型 T；大量参考采样与标准答案是前提，原论文的平均增益不能排除难题子群下降。

## P08 — CoT-Valve

[Ma 等，Length-Compressible Chain-of-Thought Tuning，2502.09601v1](https://arxiv.org/html/2502.09601v1)，2025-02。已读 §3–4、分析；由长度控制/任务向量路线进入。

机制：学习长短策略之间的参数方向，用系数控制压缩，配合 MixChain 和渐进训练。QwQ-32B-Preview，GSM8K：约 741→225 tokens，95.07%→94.92%；AIME 结果仍有少答对题的代价。类型 T；依赖特定参数方向，不能认定预算系数在跨域和不同模型上均可通用。

## P09 — L1

[Aggarwal、Welleck，Controlling How Long A Reasoning Model Thinks With Reinforcement Learning，2503.04697v2](https://arxiv.org/html/2503.04697v2)，2025-10，COLM 2025。已读 §3–5、附录；主代理复核短链表与 soft violation 定义。

机制：将目标长度给模型，LCPO 同时奖励正确性和长度匹配；Exact 匹配目标，Max 通过软惩罚鼓励不超目标，允许更短，也可能超目标。DeepScaleR-1.5B，40K 数学训练数据，主要训练目标 100–4,000 token；与 s1 截断等在固定预算比较。类型 T。**表中 L1-Max 385 tokens/39.1% 低于 Qwen-1.5 的 752/41.0%；L1-Short 才是 382/42.6%。** 论文低违约率采用容忍 500-token 偏差的 soft 定义，不是严格上限违约率。

## P10 — DAST

[Shen 等，Difficulty-Adaptive Slow Thinking for Large Reasoning Models，2503.04472v3](https://arxiv.org/html/2503.04472v3)，2026-01，EMNLP Industry 2025。已读 §2–4、Limitations；主代理复核表 1 和偏好构造。

机制：以采样成功率和正确路径长度估计题目预算，把难度、正确性、长短联合用于偏好配对，再用 SimPO 训练。R1-Distill-Qwen-7B/32B，MATH 每题采 20 条；32B MATH-500：94.4%/3,782→95.8%/2,044，总生成长度下降 46%。类型 T。7B AIME 长度略增；难度估计依赖当前采样策略；C-LEN（正确回答长度）不能替代含失败样本的总 LEN。

## P11 — Coconut

[Hao 等，Training Large Language Models to Reason in a Continuous Latent Space，2412.06769v1](https://arxiv.org/html/2412.06769v1)，2024-12。已读 §2–5.3、讨论；由潜在推理分支进入。

机制：用 hidden state 作为下一步连续输入，以课程逐渐替换显式 CoT。GPT-2，GSM8K/ProntoQA/ProsQA；不同任务有正负结果，GSM8K CoT 42.9% 对 Coconut 34.1%。类型 C/T（表示改变）：自然语言输出减少，latent 步仍执行前向，不能和普通 token 无条件等值计量。

## P12 — CODI

[Shen 等，Compressing Chain-of-Thought into Continuous Space via Self-Distillation，2502.21074v3](https://arxiv.org/html/2502.21074v3)，2025-09，EMNLP 2025。已读 §3–5、§7；沿 Coconut/iCoT 引用追读。

机制：显式教师和连续学生自蒸馏，在答案位置对齐 hidden states；固定少量连续 thoughts。GPT-2/Llama3.2-1B，GSM8K 增强数据及常识任务，与 Coconut、CoT-SFT 等比较，作者报告恢复更多显式 CoT 能力及加速。类型 C/T，主要证据仍为小模型、固定 latent 长度；压缩率的分母不能与 API token 账直接混用。

## P13 — Demystifying Long Chain-of-Thought Reasoning in LLMs

[Yeo 等，2502.03373v1](https://arxiv.org/html/2502.03373v1)，2025-02。已读 §3.2、§4–6；作为长度奖励路线的限制证据追读。

机制/结果：Llama3.1-8B/Qwen-Math-7B 的正确性奖励会推动长链；cosine 奖励约束长度，但长训练可能利用重复文本钻奖励空子。类型 I/T；并非证明所有长度 RL 都会崩溃，而是提示训练稳定、重复和难题能力需要同时验证。

RLHF/DPO/GRPO 的原始数学来源保留在[技术来源记录](https://github.com/Shiba-hua/token-efficient-llm/blob/codex/archive-v3-topic-map/meta/posttraining-sources.md)；这些基础算法的存在和采用情况不能代替上述效率机制的证据。

## P14 — GKD / On-policy Distillation of Language Models

[Agarwal 等，Learning from Self-Generated Mistakes，2306.13649v3](https://arxiv.org/html/2306.13649v3)，2024-01。已读 §2–4、附录 A.4/A.7；由蒸馏分支追读。

让学生生成，再在其访问的前缀上接受教师分布监督，可混合教师/学生轨迹并选择 KL/JSD。T5-XL 教师与不同规模学生，XSum/WMT/GSM8K 等，对照离线 KD。类型 I；主要支持学生分布训练改善能力迁移，不提供同质量输出长度下降。教师前向、学生 rollout 和容量差距不能忽略。

## P15 — Thinking Machines On-Policy Distillation

[官方技术文章](https://thinkingmachines.ai/blog/on-policy-distillation/)，2025-10-27。已读 implementation、reasoning、personalization、discussion 和引用表。

学生采样，教师提供 token log-prob，使用 reverse-KL 类反馈更新；Qwen3-8B/32B、DeepMath/AIME 的示范。类型 V/I/C。文章中 Qwen3 的 GPU-hours 对比属于引用的技术报告，不能算本博客的独立复现。本库不使用这些训练倍率证明部署 token 节省，也不把该厂商配方等同于所有 OPD。

## P16 — Search-R1

[Jin 等，Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning，2503.09516v5](https://arxiv.org/html/2503.09516v5)，2025-08。已读 §3–5；由 Agentic-RL 搜索分支进入。

推理和搜索交替，最终答案给奖励，检索结果不计入策略 loss。Qwen2.5-3B/7B，Wikipedia/E5，七个 QA 集；7B 主配置平均 EM .431，对照无搜索 R1 .276。类型 I；后期搜索次数、检索文本和长度可能增加。**不参与梯度的外部文本仍可能参与部署输入计费。**

## P17 — ReTool

[Feng 等，Reinforcement Learning for Strategic Tool Use in LLMs，2504.11536](https://arxiv.org/abs/2504.11536)，2025-04。已读 §2–4.2；由工具推理与冷启动 SFT 分支追读。

先用代码/解释器轨迹冷启动，再以答案正确性训练工具策略；解释器结果被 loss-mask。Qwen2.5-32B，AIME24/25，最大序列 16,384；训练曲线报告平均输出约 10K→6K，更多计算交给代码。类型 T/I；不能遗漏 sandbox 成本，也不能用不同训练 steps 或文献 avg@k 直接宣称同算力优势。

## P18 — Agentic-R1

[Du 等，Distilled Dual-Strategy Reasoning，2507.05707v2](https://arxiv.org/html/2507.05707v2)，2025-08。已读 §3–4、附录 A.3/A.6/A.7。

DualDistill 组合工具教师与文本教师，再自蒸馏；7B 学生、约 2.6K 轨迹，五个数学集、4,096/32,768 两预算。自蒸馏版本平均成绩提高，但某些 MATH500 配置落后纯文本基线。类型 I/T（固定上限性能）；不是在线 Agentic-RL 优化器，也没有完整平均任务 I/O 曲线，放在策略学习交叉分支而不误分类。

## P19 — Demystifying Reinforcement Learning in Agentic Reasoning

[Yu 等，2510.11701v1](https://arxiv.org/html/2510.11701v1)，2025-10。已读 §2–6、附录 A。

对 GRPO 的 loss、clip、超长惩罚和工具调用奖励作消融。Qwen2.5-7B/Qwen3-4B，3K SFT/30K RL、代码解释器；报告训练样本效率和解题性能变化。类型 I；对工具调用施加奖励不等于惩罚所有工具。没有完整部署成本表，不能仅凭长度奖励的存在就宣布省 token。

## P20 — InstructGPT

[Ouyang 等，Training language models to follow instructions with human feedback，2203.02155v1](https://arxiv.org/abs/2203.02155v1)，2022-03。已读 §3–4、§5.3、附录 C。

SFT→人类偏好奖励模型→PPO。GPT-3 1.3B/6B/175B，在 API 类指令与人工偏好上比较；小 InstructGPT 可在该偏好目标上胜过更大 GPT-3，部分通用 NLP 任务会退化。类型 V/I；指令遵循与人类偏好不是固定质量的 token 节省。

## P21 — Constitutional AI

[Bai 等，Harmlessness from AI Feedback，2212.08073v1](https://arxiv.org/abs/2212.08073v1)，2022-12。已读 §1.2、§3–4。

按原则 critique/revise 后 SFT，再用 AI 比较训练偏好模型并作 RLAIF；52B 模型，安全/帮助性人工评测。类型 V/I；省人类标签不等于省部署 token，过度优化会产生固定化或过严输出。

## P22 — DPO

[Rafailov 等，Your Language Model is Secretly a Reward Model，2305.18290v3](https://arxiv.org/html/2305.18290v3)，2024-07。已读 §4–6、附录 D。

在偏好模型与 KL 正则假设下重参数化目标，直接学习 chosen/rejected 的 policy/reference 对数比。GPT-2/GPT-J/Pythia 上做情感、摘要与对话比较，目标为 reward/KL 或偏好胜率。类型 I/C；省去部分 RL 训练流程，不是缩短回答的损失。

## P23 — SimPO

[Meng、Xia、Chen，Simple Preference Optimization with a Reference-Free Reward，2405.14734v3](https://arxiv.org/html/2405.14734v3)，2024-11。已读 §2、§4、附录 C/E/H/J。

平均 token log-prob 加目标间隔，不需 reference；Llama/Mistral/Gemma 的偏好数据和指令评测。类型 I/C；长度归一化修正训练偏差，并不奖励短回答。Gemma 的某些结果平均回答反而更长；length-controlled win rate 是控制长度偏差的质量指标，不能解释为 token 降幅。

## P24 — Training Language Models to Reason Efficiently

[Arora、Zanette，2502.04463v4](https://arxiv.org/html/2502.04463v4)，2025-11。已读 §3–5、附录 F–J。

对正确回答加题内标准化长度的平滑惩罚，效率系数控制取舍；R1-Distill-Qwen-1.5B/7B，约 3.2K Numina 题，数学/常识/逻辑评测。7B MATH 某系数约 4K→2.6K token，同时损失约 2.2 个百分点。类型 T。理论的表格策略、正确解覆盖及总体最优假设不保证有限神经网络训练零性能损失；不同归一化和长训练也可能失败。

## 定向复核补记

Luna 对 P07/P09/P24 再读方法、实验表与限制；主代理据原文定位修正 O1-Pruner 标题，区分平均点估计与等性能检验，补充 Arora–Zanette 的 MATH500/α=0.1 条件，以及 L1-Max 的软约束。三篇均未提供本项目所需的独立污染审计，不能将域外评测名称等同于绝对干净。未重新执行训练。

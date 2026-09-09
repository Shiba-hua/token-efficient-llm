# 数据、预训练与中训练：原文阅读记录

检索/阅读日：2026-09-09。首轮原文阅读：Luna；分类、取舍与正文整合：主代理。主代理另核对 Rho-1 的完整输入机制、DeepSeek-V3 表 9 的模型身份与长度、BLT 的 FLOPs 口径。以下不构成新实验，也不把训练数据量当作部署 token。

## S01 — Efficient Inference for Large Reasoning Models: A Survey

[Liu 等，2503.23077v3](https://arxiv.org/html/2503.23077v3)，2025-08。已读 Introduction、Taxonomy、Empirical Analyses、Limitations。入口作用：区分推理 token、显存和时延；沿 compact reasoning、奖励、潜在推理的参考文献追读。综述采用的模型/输出/输入分类与生命周期分类不一致，跨阶段路线须重新索引。异质结果不提供统一排名。

## S02 — A Survey on Efficient Large Language Model Training: From Data-centric Perspectives

[Luo 等，2510.25817v1](https://arxiv.org/html/2510.25817v1)，2025-10。已读 §1–4、§6、§9。入口作用：发现选择、质量增强、合成数据、蒸馏与压缩分支；沿 §6 追读 LLMLingua-2。这里的 data efficiency 大多指训练/标注效率，不能直接等价于部署省 token。

## E01 — FineWeb / FineWeb-Edu

[Penedo 等，The FineWeb Datasets，2406.17557v2](https://arxiv.org/html/2406.17557v2)，2024-10；由清洗/去重分支发现。已读 §3.1–3.7、§4–6。

机制：清洗、逐 crawl 去重、教育价值分类筛选是不同选择轴。原实验以 1.71B 模型、相同训练预算和双种子比较；教育子集的 350B-token 实验在 MMLU/ARC 上优于原始 FineWeb。全局去重没有统一优于逐 crawl 去重，是“删得更多总更好”的反例。类型 I：支持数据配方改变能力—训练预算关系，未测等性能部署长度；小模型与知识型任务的结果有范围限制。

## E02 — Rho-1: Not All Tokens Are What You Need

[Lin 等，2404.07965v1](https://arxiv.org/html/2404.07965v1)，2024-04；由 token-level data selection 分支发现。已读 §2、§3、Discussion；主代理复核 §1–2。

机制：用参考模型的损失识别目标分布，重点学习当前模型相对参考模型仍有较高 excess loss 的 token。**输入仍是完整序列，只屏蔽部分损失**。TinyLlama/Mistral-7B 在 15B OpenWebMath 上继续训练，对比同设置 CLM；论文报告数学任务明显改善。类型 I，不能按保留 token 比例直接扣除前向 FLOPs。额外参考模型与打分成本、参考分布偏差都需考虑。

## E03 — Better & Faster Large Language Models via Multi-token Prediction

[Gloeckle 等，2404.19737v1](https://arxiv.org/html/2404.19737v1)，2024-04；由综述及 DeepSeek-V3 引用链发现。已读 §2、§3.1–3.7、§4 和相关附录。

机制：共享主干上预测多个未来位置；训练时增加预测信号，部署可去掉辅助头或作自投机。13B 代码模型报告更多 HumanEval/MBPP 题被解出；7B 四 token 模式有约 3× 解码加速。自然语言选择题、不同数据量下的最优预测跨度并不一致。类型 I/C：用户所需生成序列没有因为一次预测多个 token 而自动变短。

## E04 — DeepSeek-V3 Technical Report

[DeepSeek-AI，2412.19437v2](https://arxiv.org/html/2412.19437v2)，2025-02；由 MTP 路线追到大型工程配方。已读 §2、§3.4、§4、§5.1/5.4、§6；主代理复核表 9。

机制：MoE/MLA/MTP 分别影响激活计算、KV 和未来预测；整套配方不能归因于其中单项。MTP 报告约 1.8× TPS，是 C 类。**表 9 实际比较 DeepSeek-V2.5 baseline 与 +R1 Distill**：MATH-500 pass@1 74.6→83.2，长度 769→1,510；这提供“蒸馏能力更强但回答更长”的直接反例，不是 V3 的两个公开 checkpoint 排名。训练报告规模远超小团队。

## E05 — Byte Latent Transformer: Patches Scale Better Than Tokens

[Pagnoni 等，2412.09871v1](https://arxiv.org/html/2412.09871v1)，2024-12；由表示/分词分支发现。已读 §2–3、§5.3、§6、§9–10；主代理复核固定计算与 50% 的定义。

机制：以局部字节模型的熵动态分 patch，困难位置多分配大模型计算。最大实验 8B/4T bytes，FLOPs 控制比较 Llama 类模型；最高约 50% 推理 FLOPs 节省允许小幅质量取舍。类型 C/I。byte、patch、BPE token 不可直接数值相除作部署 token 改进；实际墙钟、熵模型和新架构工程成本尚需另测。

## E06 — Tokenizer Choice For LLM Training: Negligible or Crucial?

[Ali 等，NAACL Findings 2024](https://aclanthology.org/2024.findings-naacl.247/)，[PDF](https://aclanthology.org/2024.findings-naacl.247.pdf)；由多语言 fertility 分支追读。已读 §2–5、Limitations。

机制：比较 BPE/Unigram、词表大小及单语/多语分词；低 fertility 表示相同文本拆成更少 token。24 组配置使用 2.6B decoder、70B words，评测英/德/法/意/西；更低 fertility 未必得到最佳下游任务分数，词表增大会增加部分计算。类型 I/C，同时提供表示长度变化的直接测量；未证 reasoning 能力与整项任务质量保持不变。

## E07 — Data Engineering for Scaling Language Models to 128K Context

[Fu 等，2402.10171v1](https://arxiv.org/html/2402.10171v1)，2024-02；由长上下文继续训练分支发现。已读 §3–7。

机制：在各数据域内部上采样长序列，保持域比例。LLaMA-2 7B/13B，64K–80K full-attention 继续预训练；约 5B token 才使测试检索较好泛化到 128K，少量训练先取得局部检索能力。和 LongLoRA 等比较，并报告短任务/BookQA。类型 I；获得更长窗口与减少送入窗口的内容是两个问题，长序列继续训练本身也很昂贵。

## E08 — s1: Simple Test-time Scaling

[Muennighoff 等，2501.19393](https://arxiv.org/html/2501.19393)，2025；由 S01 的紧凑训练数据与预算控制分支追读。已读 §2–4、§5.1–5.2。此处为阅读时的 HTML 版本，复现实验还应锁定代码与权重 revision。

机制：从约 59K 候选按质量、难度与多样性筛成 1K；SFT 后通过结束思考或继续 “Wait” 调节思考预算。Qwen2.5-32B 的 s1-32B 在 AIME24/MATH500/GPQA 用 budget forcing 时为 56.7/93.0/59.6，未用时 50.0/92.6/56.6。类型 I/T：小数据与预算可控的证据；增加 Wait 的性能收益不能被写成少 token，过长还可能循环。

## E09 — LLMLingua-2

[Pan 等，ACL Findings 2024](https://aclanthology.org/2024.findings-acl.57/)，[PDF](https://aclanthology.org/2024.findings-acl.57.pdf)；由 S02 §6.2 的蒸馏压缩引用进入。已读方法、实验与动态压缩比附录。

机制：GPT-4 生成保留/删除标签，训练双向 encoder 作抽取式压缩，部署时使用小压缩器。MeetingBank、Mistral-7B 的约 3× 压缩实验，输入约 3,003→970，QA F1 66.95→76.22；LongBench 更高压缩比有明显质量损失。类型 T，压缩器有计算开销；只支持指定任务的输入节省，不能把输入压缩比当作总输出或整项 Agent 成本比例。

更多早期数据质量的逐条核查保留在[第一版数据来源记录](../meta/foundations-sources.md)及[数据技术专题](../docs/technical/01-data.md)。那些检查不提供“绝对干净”的数据量保证。

## E10 — DoReMi

[Xie 等，Optimizing Data Mixtures Speeds Up Language Model Pretraining，2305.10429v4](https://arxiv.org/abs/2305.10429v4)，2023-11。已读 §1–4、§6；由 S02 的 domain reweighting 分支追读。

小 reference/proxy 以 Group DRO 学域权重，再迁移到大模型。Pile 22 域，280M proxy→8B；同预算平均 one-shot EM 提升约 6.5 个百分点，达到基线分数由 200K→75K steps。类型 I；代理训练额外成本约为主训练 FLOPs 的 8%，域划分与跨规模迁移有前提。它调混合比例，不是逐文档删除，也不直接减少部署输入。

## E11 — DCLM

[Li 等，DataComp-LM: In Search of the Next Generation of Training Sets，2406.11794v4](https://arxiv.org/html/2406.11794v4)，2025-04。已读 §1–6、去污染与指令训练附录；由数据综述及 FineWeb 关联引用追读。

统一候选语料池和训练/评测设置，比较抽取、去重、过滤与混合；模型覆盖 412M–7B，评测 53 项。7B 约 280B 训练 token 的基线展示较强 MMLU 等能力；不同规模下筛选方案排序有相关性。类型 I；与 FineWeb 构成独立团队的数据配方证据，但两者仍共享网页语料和若干基准。没有推理长度与服务成本比较。

## E12 — Don’t Stop Pretraining

[Gururangan 等，Adapt Language Models to Domains and Tasks，ACL 2020](https://aclanthology.org/2020.acl-main.740/)。已读 §1–5、附录 B/C/E/F；由继续预训练分支追读。

DAPT 用领域文本，TAPT 用任务训练文本，还可选近邻文档。RoBERTa-base、四域八分类任务；DAPT 的 ChemProt F1 81.9→84.2，ACL-ARC 63.0→75.4；错域适配可能退化。类型 I；这是 encoder 分类证据，不能直接当作 decoder 的推理 token 改善。

## E13 — SmolLM3

[Hugging Face 官方技术报告](https://huggingface.co/blog/smollm3)，2025-07-08。已读架构、数据混合、两类 mid-training、SFT/APO、合并与双模式评测；由近期开源模型配方追读。

3B 模型在 11.2T 预训练后进行长上下文与推理阶段，支持 think/no-think。报告推理训练会损伤 RULER 长上下文能力，使用 checkpoint 合并恢复；专用长书籍/代码混合并未自动优于自然长文本。类型 V/I；模式能力不同，没有同质量完整 I/O 曲线，不能仅因 no-think 就宣布效率提升。

## E14 — Chinchilla

[Hoffmann 等，Training Compute-Optimal Large Language Models，2203.15556](https://arxiv.org/abs/2203.15556)，2022。已读 §1、§3–5 和附录；由 scaling 分支进入。

用数百个 dense 模型拟合固定训练计算下的参数/数据比例；70B、1.4T-token 的 Chinchilla 在与 Gopher 相近训练 FLOPs 下更强。类型 I/C；参数较少使每 token 计算降低，但它没有测生成序列更短。基于预训练 loss 的最优不等于包含长期部署需求的最优。

## E15 — Beyond Chinchilla-Optimal

[Sardana 等，Accounting for Inference in Language Model Scaling，2401.00448v3](https://arxiv.org/html/2401.00448v3)，2025-04。已读 §1–6、§8、附录 A–D；沿 Chinchilla 后续工作进入。

将预期推理需求纳入生命周期计算目标，通常使较小、训练更久的模型有吸引力。47 个 MPT 模型，150M–6B；大部署需求下的部分更大模型结论是 scaling law 外推。类型 C；输入/输出 token 被作为给定需求，没有证明减少这些 token。请求长度、硬件利用率与质量 proxy 都影响推断。

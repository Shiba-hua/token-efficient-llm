# 评测与可执行数据：原文记录

阅读日：2026-09-09。主代理阅读 S08、B01、B02；Luna 阅读 B03–B06 并专门复核 SWE-Gym 仓库隔离，主代理写作。四个原有基准的固定数据、源码、样例和论文差异见[评测章](../docs/09-evaluation.md)及[固定样例目录](../data/benchmarks/)；保留的历史核查只支持其注明的版本。

## S08 — A Survey on Evaluation of LLM-based Agents

[Yehudai 等，Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.1330/)，2026-07。已取得并阅读全文 PDF 的分类图、§2–3 和 §7 成本/环境讨论。按能力（规划、工具、记忆等）与应用（网页、代码、科学等）组织评测，提示将模型、harness、成本和动态环境变化分开。由其参考文献追读 B01；综述分类不能代替原始基准的划分、判分器与成本实现。

## B01 — AI Agents That Matter

[Kapoor、Stroebl、Siegel、Nadgir、Narayanan，2407.01502v1](https://arxiv.org/html/2407.01502v1)，2024-07。已读 §2 成本控制、§3 HotpotQA 实验、§4 模型/下游评测及 holdout 讨论。v2 HTML 访问失败，记录的是 v1。

DSPy/ColBERTv2 HotpotQA，比较准确率单目标与准确率/成本联合优化，5 次运行均值；GPT-3.5 配置约少 53% 可变费用，Llama-3-70B 约少 41%，质量相近。这里的成功指标是检索到全部 gold supporting documents，不是回答 EM；成本为该时点美元，另有离线固定优化成本。类型 C/T（调用内容会变化，数字是费用）；不能直接改写为全模型通用 token 降幅。

## B02 — CostBench

[Liu 等，2511.02734v1](https://arxiv.org/html/2511.02734v1)，2025-11-04。已读 §3 环境和划分、§4 设置/指标、§5 分析。由高效 Agent/成本评测检索进入。

六个旅行相关领域，原子与组合工具、动态赋值的工具成本；过滤后表 2 为 1,902 train / 381 test。10 个模型、temperature 0、生成上限 16,384。评测正确执行、相对最优计划的成本差距及轨迹匹配。类型 C：主要成本是环境分配的工具费用，不是语言模型 I/O token。适合研究能否识别便宜动作，不能代替真实工具失败、文本观察和完整任务 token 曲线。模板/规则生成和随机费用不自动保证无污染或现实泛化。

## B03 — OSWorld

[Xie 等，Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments，2404.07972v2](https://arxiv.org/abs/2404.07972v2)，2024-05-30。已读 §2、§3.1–3.4、§4–5、附录 B。

369 个 Ubuntu 任务，另有 43 个 Windows 任务；302 种初始状态、134 个执行判分函数，含 101 个跨应用任务、30 个不可完成任务，84 题复用前作。论文报告约 1,800 人时建设。没有传统独立 train/dev/test，也不提供全部任务的专家最优轨迹。VM 文件/窗口/应用状态检查比动作模仿更接近完成任务，但依赖软件版本与断言覆盖；部分分、拒绝不可行任务须遵循具体 evaluator。类型 I（评测资源），不能将公开评测配置计作干净训练轨迹。

## B04 — WebArena

[Zhou 等，A Realistic Web Environment for Building Autonomous Agents，2307.13854](https://arxiv.org/abs/2307.13854)，2023。已读 §2、§3.1–3.2、§4–5、附录 A。

812 项意图、241 个模板，四类自托管网站及辅助工具。通过答案 exact/must-include/fuzzy 与 URL/数据库状态判断；部分模糊匹配调用 GPT-4-0613。主要是评测集，没有官方独立训练演示集。模板相关性应进入划分/统计，不把网页可运行等同于测试独立，也不将离线自托管结果外推到整个互联网。

## B05 — Mind2Web

[Deng 等，Towards a Generalist Agent for the Web，2306.06070v3](https://arxiv.org/abs/2306.06070v3)，2023-12-09。已读 §2–4、§6、附录 B–D。

2,350 条审核后的任务，137 个网站、31 个域，平均 7.3 个动作，提供页面快照/DOM/HAR 与人工动作。train 1,009，cross-task 252、cross-website 177、cross-domain 912。评估元素准确率、操作 F1、step/task success，但基于记录的页面和动作序列，不是完整在线终态执行。类型 I：可做监督训练；演示路径不是唯一正确路径，旧网页快照不保证今日可重放。

## B06 — SWE-Gym

[Pan 等，Training Software Engineering Agents and Verifiers with SWE-Gym，2412.21139v2](https://arxiv.org/abs/2412.21139v2)，2025-06-06。已读 §3–6、附录 B；额外复核 §3、§3.1–3.2、§4.1–4.2、§5.2 的仓库独立性。

2,438 个可执行 Python issue、11 个仓库；SWE-Gym Lite 是其中 230 个较易原型任务，不是独立测试集。Raw 的 64,689 issues / 358 repos 没有相同的环境/测试保证。491 条成功训练轨迹最多覆盖 294 个去重任务，平均约 19 轮、约 19K token；轨迹数不等于独立题数。

论文明确 11 个可执行仓库与 SWE-bench 仓库分离，外测使用 SWE-bench Lite 300 / Verified 500；不能扩大为 Raw 全部仓库已经完成同等级隔离审计。通过 gold patch 测试与成功轨迹筛选提供可执行证据，但测试不完备、Python/仓库覆盖窄。类型 I：训练成本/数据统计，不是部署 token 节省实验。

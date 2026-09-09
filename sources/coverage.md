# 覆盖边界、发现链与交叉索引

[总论](../docs/00-overview.md) · [文献台账](papers.json) · [阅读协议](README.md)

资料截止：2026-09-09。用户提供的[知乎分类文章](https://www.zhihu.com/question/1958561146647876509/answer/2075524312644195693)用于扩展方向覆盖；效率结论回到原始论文。13 章以研究方向为主目录，数据、预训练、中训练各自独立。这里记录实际选择范围，未声称穷尽全部数据库或统计引用量。

## 本轮综述入口

下表中的“读取”是综述自身的分类/候选提取，具体范围见对应 JSON。原论文首读由主代理去重后另行派发。2026 年补充既包括新综述，也包括旧 arXiv 工作的新版本。

| 综述 | 固定版本 | 提取记录 |
| --- | --- | --- |
| [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) | v19 (last revised 2026-03-18; arXiv page checked 2026-09-09) | [2303.18223](../meta/survey-extractions/2303.18223.json) |
| [Towards Large Reasoning Models: A Survey of Reinforced Reasoning with Large Language Models](https://arxiv.org/abs/2501.09686) | v3, 2025-01-23 | [2501.09686](../meta/survey-extractions/2501.09686.json) |
| [LLM Post-Training: A Deep Dive into Reasoning Large Language Models](https://arxiv.org/abs/2502.21321) | v2, revised 2025-03-24 (arXiv page checked 2026-09-09) | [2502.21321](../meta/survey-extractions/2502.21321.json) |
| [A Survey of Efficient Reasoning for Large Reasoning Models: Language, Multimodality, and Beyond](https://arxiv.org/abs/2503.21614) | v2 (arXiv HTML dated 2025-12-31; checked 2026-09-09) | [2503.21614](../meta/survey-extractions/2503.21614.json) |
| [A Survey on Test-Time Scaling in Large Language Models: What, How, Where, and How Well?](https://arxiv.org/abs/2503.24235) | arXiv v3, 2025-05-04 | [2503.24235](../meta/survey-extractions/2503.24235.json) |
| [Efficient Reasoning Models: A Survey](https://arxiv.org/abs/2504.10903) | v2 (29 Sep 2025) | [2504.10903](../meta/survey-extractions/2504.10903.json) |
| [A Survey of Token Compression for Efficient Multimodal Large Language Models](https://arxiv.org/html/2507.20198v5) | v5 (01 Feb 2026; formerly When Tokens Talk Too Much) | [2507.20198](../meta/survey-extractions/2507.20198.json) |
| [The Landscape of Agentic Reinforcement Learning for LLMs: A Survey](https://arxiv.org/abs/2509.02547) | arXiv v5, 2026-04-17 | [2509.02547](../meta/survey-extractions/2509.02547.json) |
| [A Survey of Reinforcement Learning for Large Reasoning Models](https://arxiv.org/abs/2509.08827) | v3, 2025-10-09 (arXiv page says last revised 9 Oct 2025) | [2509.08827](../meta/survey-extractions/2509.08827.json) |
| [Agentic Reasoning for Large Language Models](https://arxiv.org/abs/2601.12538) | v1 | [2601.12538](../meta/survey-extractions/2601.12538.json) |

原有 Stop Overthinking、Prompt Compression、Efficient Agents、数据中心训练及评测综述卡继续作为发现入口，不假定每篇本轮都重新阅读全文。新引用先进入[候选提及表](../meta/candidate-mentions.json)，再与旧记录和别名匹配；最终形成[41 篇生产队列](../meta/reading-queue.json)及两篇校准论文。

2026 年原始工作补充包括 The Art、OPSD、RecurGuard、LongVU-TTT；Coconut 等采用本轮指定版本。它们分别填补统一效率训练、学生前缀监督、资源攻击防御与视频适配问题，不意味着所有 2026 年方向已穷尽。

## 生命周期索引

| 生命周期 | 主要章节 | 跨阶段问题 |
| --- | --- | --- |
| 数据 | [01](../docs/01-data/index.md)、[05](../docs/05-posttraining/index.md) | 难度、候选覆盖与正确短轨迹；压缩器也可构造训练标签 |
| 预训练 | [02](../docs/02-pretraining/index.md)、[04](../docs/04-architecture/index.md) | 多未来目标、表示粒度与部署预算的间接链条 |
| 中训练 | [03](../docs/03-midtraining/index.md)、[09](../docs/09-multimodal/index.md) | 领域/模式持续适配与多模态更新；按原文辨明阶段 |
| 后训练 | [05](../docs/05-posttraining/index.md)、[04](../docs/04-architecture/index.md)、[08](../docs/08-agents/index.md) | SFT、蒸馏/OPD、偏好、RLVR、Agentic-RL；连续表示和工具策略 |
| 推理部署 | [06](../docs/06-reasoning/index.md)、[07](../docs/07-context/index.md)、[08](../docs/08-agents/index.md)、[09](../docs/09-multimodal/index.md)、[10](../docs/10-systems/index.md) | 搜索与停止、上下文、调用通信、多模态和系统执行 |
| 评测 | [11](../docs/11-evaluation/index.md)、[12](../docs/12-safety/index.md)、[13](../docs/13-domains/index.md) | 全任务分母、失败/攻击成本、领域正确性和迁移 |

## 效率机制索引

| 机制 | 代表路线与主要讲解位置 | 必须检查的代价 |
| --- | --- | --- |
| 少读 | LLMLingua-2、RECOMP、ICAE、LightMem → [07](../docs/07-context/index.md)；FastV/LongVU → [09](../docs/09-multimodal/index.md) | 压缩器、回读、漏证据与非文本单位 |
| 少写 | 最短正确轨迹 → [01](../docs/01-data/index.md)；TokenSkip、DAST、The Art → [05](../docs/05-posttraining/index.md) | 难题覆盖、正确/错误长度和高预算能力 |
| 少搜索／重试 | DeepConf/Snell → [06](../docs/06-reasoning/index.md)；ReTool/证明工具 → [05](../docs/05-posttraining/index.md)、[13](../docs/13-domains/index.md) | 验证成本、相关错误与工具执行 |
| 少交互／通信 | ReWOO、LLMCompiler、AgentPrune、MEM1 → [08](../docs/08-agents/index.md) | 少轮数是否丢失反馈；发送与接收成本是否都改变 |
| 计算内化 | MTP/Rho-1 → [02](../docs/02-pretraining/index.md)、[03](../docs/03-midtraining/index.md)；System 2→1/GKD/OPSD → [05](../docs/05-posttraining/index.md) | 能力到部署长度的证据链是否真正测过 |
| 表示改变 | Coconut、CODI、BLT、LCM → [04](../docs/04-architecture/index.md)；多模态特征 → [09](../docs/09-multimodal/index.md) | latent 步、字节、句子和特征不能伪装成免费计算 |
| 预算分配 | L1 → [05](../docs/05-posttraining/index.md)；搜索/停止 → [06](../docs/06-reasoning/index.md)；路由 → [08](../docs/08-agents/index.md) | 上限与实耗、费用与 token、失配难度与攻击尾部 |

投机解码/EAGLE 在台账中使用“计算执行（相邻成本）”边界标签，不强行归为少 token 的七类机制。

每篇精读在台账中有一个主归属和若干关联章；同一方法只保留一个精读页面。标签表达研究关联，不替代证据等级。例如投机解码列入预算/执行关联，不表示它减少最终文本 token。

## 选择性覆盖与尚缺证据

预训练、中训练章节集中解释 MTP、LCM、Rho-1 及领域/模式路线。大规模数据配比、MoE、SSM 与长上下文架构作为机制边界和历史入口，没有逐一扩写完整原文。原因是本轮选择的代表证据大多尚未测量同质量任务 token 曲线，不能用训练效率来填补问题。

多模态正文覆盖图像、视频、语音；世界模型对真实交互节省与任务 token 的联系仍缺直接可比证据。系统章详解投机解码，缓存、量化、调度和端侧部署保留边界。安全章聚焦推理资源攻击，隐私、治理没有被强行写成效率路线。领域章以数学程序、跨语言推理和形式化证明为代表，未穷尽全部专业模型与科学自动化框架。

这些是可公开检查的停止范围，不能用于“某方向无人研究”的断言。下一轮应先围绕具体缺口扩展候选，复用已有精读，而不是按每篇综述重新派发相同论文。

## 43 篇共享精读索引

| 主归属 | 论文 | 固定读取版本 |
| --- | --- | --- |
| 01-data | [Self-Training Elicits Concise Reasoning in Large Language Models](../docs/01-data/reference/2502.20122.md) | arXiv v3, 2025-06-10 |
| 02-pretraining | [Better & Faster Large Language Models via Multi-token Prediction](../docs/02-pretraining/reference/2404.19737.md) | arXiv v1, 2024-04-30 |
| 02-pretraining | [Large Concept Models](../docs/02-pretraining/reference/2412.08821.md) | 2412.08821v2 |
| 03-midtraining | [Rho-1: Not All Tokens Are What You Need](../docs/03-midtraining/reference/2404.07965.md) | v1 |
| 04-architecture | [Coconut](../docs/04-architecture/reference/2412.06769.md) | v4 (2026-08-23) |
| 04-architecture | [Byte Latent Transformer: Patches Scale Better Than Tokens](../docs/04-architecture/reference/2412.09871.md) | 2412.09871v1 |
| 04-architecture | [CODI](../docs/04-architecture/reference/2502.21074.md) | 2502.21074v3 |
| 05-posttraining | [GKD / On-policy Distillation of Language Models](../docs/05-posttraining/reference/2306.13649.md) | 2306.13649v3 (2024-01-17), ICLR 2024 camera-ready |
| 05-posttraining | [Distilling System 2 into System 1](../docs/05-posttraining/reference/2407.06023.md) | 2407.06023v3 |
| 05-posttraining | [CoT-Valve](../docs/05-posttraining/reference/2502.09601.md) | 2502.09601v1 |
| 05-posttraining | [TokenSkip](../docs/05-posttraining/reference/2502.12067.md) | v3 (2025-09-16; EMNLP 2025 camera-ready) |
| 05-posttraining | [DAST](../docs/05-posttraining/reference/2503.04472.md) | v3 (2026-01-12) |
| 05-posttraining | [L1](../docs/05-posttraining/reference/2503.04697.md) | 2503.04697v2 / COLM 2025 |
| 05-posttraining | [ReTool](../docs/05-posttraining/reference/2504.11536.md) | 2504.11536v2 |
| 05-posttraining | [Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models](../docs/05-posttraining/reference/2601.18734.md) | 2601.18734v3 (2026-03-20) |
| 05-posttraining | [The Art of Efficient Reasoning: Data, Reward, and Optimization](../docs/05-posttraining/reference/2602.20945.md) | v3 |
| 06-reasoning | [Compute-optimal Test-Time Scaling](../docs/06-reasoning/reference/2408.03314.md) | v1 (2024-08-06), fixed for reading on 2026-09-09 |
| 06-reasoning | [Chain of Draft: Thinking faster by writing less](../docs/06-reasoning/reference/2502.18600.md) | 2502.18600v2 |
| 06-reasoning | [DeepConf](../docs/06-reasoning/reference/2508.15260.md) | 2508.15260v1 |
| 07-context | [ICAE](../docs/07-context/reference/2307.06945.md) | 2307.06945v4 (ICLR 2024 camera-ready source; arXiv 2024-05-08) |
| 07-context | [RECOMP](../docs/07-context/reference/2310.04408.md) | v1 (2023-10-06) |
| 07-context | [The Complexity Trap](../docs/07-context/reference/2508.21433.md) | 2508.21433v3 |
| 07-context | [LightMem: Lightweight and Efficient Memory-Augmented Generation](../docs/07-context/reference/2510.18866.md) | 2510.18866v4 |
| 07-context | [LLMLingua-2](../docs/07-context/reference/src-7cc3fedc37b4.md) | {'title': 'LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression', 'authors': ['Zhuoshi Pan', 'Qianhui Wu', 'Huiqiang Jiang', 'Menglin Xia', 'Xufang Luo', 'Jue Zhang', 'Qingwei Lin', 'Victor Rühle', 'Yuqing Yang', 'Chin-Yew Lin', 'H. Vicky Zhao', 'Lili Qiu', 'Dongmei Zhang'], 'year': 2024, 'venue': 'Findings of ACL 2024', 'pages': '963-981', 'doi': '10.18653/v1/2024.findings-acl.57', 'url': 'https://aclanthology.org/2024.findings-acl.57/', 'pdf_url': 'https://aclanthology.org/2024.findings-acl.57.pdf'} |
| 08-agents | [REWOO: Decoupling Reasoning from Observations for Efficient Augmented Language Models](../docs/08-agents/reference/2305.18323.md) | arXiv v1, submitted 2023-05-23 |
| 08-agents | [LLMCompiler](../docs/08-agents/reference/2312.04511.md) | 2312.04511v3 |
| 08-agents | [RouteLLM](../docs/08-agents/reference/2406.18665.md) | 2406.18665v4 |
| 08-agents | [AgentPrune](../docs/08-agents/reference/2410.02506.md) | 2410.02506v1 |
| 08-agents | [MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents](../docs/08-agents/reference/2506.15841.md) | 2506.15841v2 |
| 09-multimodal | [FastV](../docs/09-multimodal/reference/2403.06764.md) | 2403.06764v3 |
| 09-multimodal | [LLaVA-PruMerge](../docs/09-multimodal/reference/2403.15388.md) | 2403.15388v6 |
| 09-multimodal | [LongVU](../docs/09-multimodal/reference/2410.17434.md) | 2410.17434v1 |
| 09-multimodal | [SpeechPrune](../docs/09-multimodal/reference/2412.12009.md) | 2412.12009v2 (arXiv online 2025-03-30) |
| 09-multimodal | [LongVU-TTT](../docs/09-multimodal/reference/2608.25729.md) | 2608.25729v1 |
| 10-systems | [Speculative Decoding](../docs/10-systems/reference/2211.17192.md) | 2211.17192v2 |
| 10-systems | [EAGLE-3](../docs/10-systems/reference/2503.01840.md) | 2503.01840v3 |
| 11-evaluation | [AI Agents That Matter](../docs/11-evaluation/reference/2407.01502.md) | 2407.01502v1 |
| 11-evaluation | [Do Not Think That Much for 2+3=? on the Overthinking of o1-like LLMs](../docs/11-evaluation/reference/2412.21187.md) | 2412.21187v2 |
| 12-safety | [OverThink](../docs/12-safety/reference/2502.02542.md) | 2502.02542v4 |
| 12-safety | [RecurGuard: Runtime Monitoring for Reasoning-Token Consumption Attacks](../docs/12-safety/reference/2606.07968.md) | v1 (2026-06-06) |
| 13-domains | [Language models are multilingual chain-of-thought reasoners](../docs/13-domains/reference/2210.03057.md) | 2210.03057v1 |
| 13-domains | [PAL](../docs/13-domains/reference/2211.10435.md) | 2211.10435v2 (2023-01-27) |
| 13-domains | [DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition](../docs/13-domains/reference/2504.21801.md) | arXiv:2504.21801v2 (2025-07-18) |

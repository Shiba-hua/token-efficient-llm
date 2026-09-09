# 分类对照、发现链与覆盖边界

检索/阅读截止：2026-09-09。本文是主代理整理的选择性地图，不报告未实际统计的引用数，不声称完成全部数据库的系统综述。以下记录分类怎样进入正文，以及补读原文后怎样修正分类。

## 综述之间的一致与不一致

| 入口与阅读记录 | 主要分类视角 | 对本地图的贡献 | 单独使用会遗漏什么 |
| --- | --- | --- | --- |
| [S01 高效推理综述](early-stages.md) | 推理长度、步骤加速与推理策略 | 连接后训练、部署控制、相邻计算 | 训练数据与长程工具输入 |
| [S02 数据中心训练](early-stages.md) | 数据选择、组织和训练信号 | 补文档/域/token三个选择粒度 | 更低训练成本不证明部署少token |
| [S03 Stop Overthinking](posttraining-map.md) | model/output/input-prompt | 发现短链、控制、蒸馏原文 | Agent的通信、动作与状态 |
| [S04 Efficient Reasoning](posttraining-map.md) | inference/SFT/RL、多模态与预算 | 补条件长度、预算与表示分支 | 与本地图的全任务成本口径不同 |
| [S05 Prompt Compression](context-agents-map.md) | hard/soft compression | LLMLingua系列与连续输入表示 | 压短单次输入未必缩短任务 |
| [S06 Toward Efficient Agents](context-agents-map.md) | memory/tool/planning | 状态管理、工具选择、预算规划 | 不按预训练/后训练分章 |
| [S07 多智能体综述](context-agents-map.md) | 角色、通信、协作、应用 | 拓扑和参与机制的发现入口 | 并非每种协作都经济 |
| [S08 Agent评测综述](evaluation-map.md) | 能力×应用、成本与环境 | 补网页/桌面/SWE、成本评测 | 仍需逐个读任务和evaluator |

“软压缩”最能说明分类为什么需要交叉：S05中的连续输入表示若只按训练阶段索引，会混进SFT；若只按可见token，则又容易和Coconut的推理表示混淆。本轮据此补读AutoCompressors/ICAE，并在07与10分别解释。S07主要作为分类入口，不以其综述转述充当节省实验。

## 从综述深入原始工作的实际路径

| 问题链 | 读到的代表原文 | 为解释关系补充的反例或条件 |
| --- | --- | --- |
| 数据如何有效 | FineWeb/DCLM→DoReMi/Rho-1 | 去重粒度、代理迁移、全文仍参与前向 |
| 少量数据能否教短解 | s1→最短自训练→C3oT/TokenSkip | 教师预算、题族选择、难题压缩退化 |
| 能力能否内化 | System2→1→GKD/OPD；Coconut→CODI | GSM8K答案蒸馏失败；V2.5蒸馏长度增长 |
| 预算该给多少 | TALE/DAST↔O1-Pruner/Arora–Zanette→L1 | 难度依赖、软预算、缺污染审计 |
| 文本与工具如何配合 | Search-R1→ReTool→Agentic-R1；Agentic-RL消融 | 检索屏蔽loss不等于无输入成本；蒸馏不等于RL |
| 压缩什么 | LLMLingua→LongLLMLingua→LLMLingua-2；RECOMP | 查询相关性、压缩器成本、遗漏证据 |
| 长程记忆如何维护 | 遮蔽/摘要→Complexity Trap；ReadAgent→AgentFold | 摘要延长轨迹、回读摊销、单轮上下文不是总量 |
| 连续输入如何压缩 | AutoCompressors↔ICAE | 白盒接口、PPL损失、前处理与缓存条件 |
| 何时调用谁 | ReAct→LLMCompiler/PTC；FrugalGPT→RouteLLM | 不同ReAct基线、美元与token分离 |
| 多少搜索与通信 | SC→ToT→Snell；Sparse/Group→S²-MAD/AgentPrune | 单代理强基线、阈值变化、输出不一定下降 |
| 何时结束 | BATS↔DEER↔L1 | 上限≠实耗，试答和验证也有成本 |
| 计算与表示 | SpecDec→EAGLE；Coconut/CODI→Huginn；BLT/Dream；FastV↔PruMerge | 同输出分布、latent步、视觉细节与训练成本 |
| 证据如何成立 | AI Agents That Matter→四基准→CostBench/GUI/SWE原文 | 数据重叠、判分版本、经济指标与任务分母 |

箭头表示沿相关问题追读，不是宣称所有论文之间有直接引用或因果继承；直接借用关系会在正文明确说明。每篇的实际阅读章节、版本、条件与局限见来源卡。

## 跨阶段机制索引

| 机制 | 主要章节 | 同时关联 | 最容易重复的想法 |
| --- | --- | --- | --- |
| 学得更有效 | 01/02/03 | 04/09 | 把训练loss变好当推理更短 |
| 同题短解/省略步骤 | 04 | 01/05/06 | 每题选最短正确轨迹 |
| 按难度分配长度 | 05/06 | 07/08 | 加预算prompt或长度reward |
| 少读证据和历史 | 07 | 01/08/09 | 固定间隔摘要、关键token分类 |
| 少走工具/调用轮次 | 08 | 06/07 | 程序化调用、工具结果过滤 |
| 少通信、少参与者 | 08 | 05/09 | 共识早停、剪通信图 |
| 改变表示或每token计算 | 02/10 | 04/07 | 将视觉/latent/缓存当零token |
| 验证驱动选择/停止 | 08/09 | 04/05/06 | 不计验证成本的“省推理” |

## 检索记录与本轮停止范围

各来源卡保留具体发现路径。本轮收尾交叉检索实际使用了以下查询（2026-09-09）；它们是可复查的补漏查询，**不是全部历史检索的完整日志**：

| 查询 | 用途与处理 |
| --- | --- |
| `efficient reasoning survey large language models 2026 token efficiency survey agents memory tool planning` | 检查reasoning与Agent分类边界，回到原综述和原论文 |
| `prompt compression survey soft compression AutoCompressors ICAE LLMLingua` | 确认硬/软压缩缺口，补两篇原文 |
| `Stanford CS336 Spring 2026 Lecture 10 Inference YouTube` | 视频入口发现，不依据二手摘要写技术结论 |
| `NICE 学术 姜慧强 LLMLingua LongLLMLingua Bilibili` | 作者中文报告入口，核查实际页面元信息 |
| `site.youtube.com/watch "CS336" "Spring 2026" "Lecture 10"` | 核对视频题目/链接，未取得全文字幕 |
| `site.bilibili.com "BV19K41187Ny"` | 页面定位，后续直接打开作者报告 |

收尾检索仍能发现新论文，但主要落入已覆盖机制；软压缩缺口已补。当前每个生命周期章节具有主要路线、竞争/组合关系、限制证据与阅读顺序，故进入整合。**尚未穷尽**：最新多模态Agent训练配方、所有任务专用路由器、量化/KV系统细分、全部科研自动化框架。不能把这份地图用作这些相邻领域的完整综述。

部分HTML版本不可访问时使用实际取得的PDF/其他注明版本；未取得全文的工作不写具体实验结论。视频访问范围见[视频入口](videos.md)。

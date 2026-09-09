# Token-efficient LLM

**围绕“同样的任务性能，更少的 token”，理解各条研究路线、代表方法、证据冲突与可检验机会。**

这是一份面向准备进入该领域的研究者的中文**领域研究地图**。从数据、预训练和中训练，到后训练、上下文、工具与多智能体，正文解释各方为什么选择不同方法、它们怎样继承或竞争，以及现在知道什么、还不知道什么。小团队可验证的曲线向左移动优先，极大预算下的能力上限作为相关分支。

![性能—token 成本前沿的概念图](assets/generated/frontier-concept.png)

保留的概念插画没有实验数据。主横轴是整项任务所有方法侧模型调用的输入+输出，包括摘要、分支、重试；费用、延迟、训练成本和隐藏计算另列。具体口径见[总论](docs/00-overview.md)和[评测章](docs/09-evaluation.md)。

## 建议先读

**[总论](docs/00-overview.md) → 一个方向分论 → [最近邻与选题卡](docs/research-opportunities.md)**。总论提供全景；分论提供机制、实验条件、限制和阅读顺序；选题卡要求说明与已有工作究竟差在哪里。

| 阅读目的 | 路线 |
| --- | --- |
| 先找小团队可做的任务级方法 | 00 → 07上下文 → 08编排 → 09评测 |
| 研究短链、蒸馏与效率训练 | 00 → 01数据 → 04SFT/OPD → 05偏好 → 06RL |
| 理解上游模型与表示选择 | 00 → 02预训练 → 03中训练 → 10相邻方向 |
| 准备提出研究idea | 任一分论 → [跨阶段索引](sources/coverage.md) → [机会与模型角色](docs/research-opportunities.md) |

## 生命周期研究地图

| 章节 | 主要路线与比较 |
| --- | --- |
| [00 总论](docs/00-overview.md) | 全生命周期鸟瞰；OpenAI/Anthropic公开方案；基础范式、条件性证据和争议 |
| [01 数据准备](docs/01-data.md) | FineWeb/DCLM、DoReMi、Rho-1、题目/轨迹选择、答案与过程可信度 |
| [02 预训练](docs/02-pretraining.md) | Chinchilla与部署摊销、tokenizer/BLT、NTP/MTP、计算与token边界 |
| [03 中训练](docs/03-midtraining.md) | DAPT/TAPT、长上下文数据工程、信号选择、分阶段能力/模式混合 |
| [04 SFT、蒸馏与OPD](docs/04-sft-distillation.md) | 答案蒸馏、最短自训练、C3oT/TokenSkip/CoT-Valve、GKD/OPD |
| [05 偏好优化](docs/05-preference-rl.md) | RLHF/RLAIF与DPO/SimPO的不同角色；TALE/DAST预算和难度偏好 |
| [06 RLVR与Agentic-RL](docs/06-rlvr-agentic.md) | O1-Pruner/Arora–Zanette、L1、Search-R1/ReTool/Agentic-R1；GRPO的准确位置 |
| [07 Prompt与上下文](docs/07-prompt-context.md) | LLMLingua系列、RECOMP、软压缩、遮蔽/摘要/回读记忆、DSPy/GEPA |
| [08 Harness与智能体编排](docs/08-harness-agents.md) | ReAct/LLMCompiler、级联/路由、搜索分配、通信剪枝、工具发现、动态停止 |
| [09 评测](docs/09-evaluation.md) | OckBench/OTB/THINK-Bench/AppWorld；真实样例与判分；GUI/SWE数据；CostBench |
| [10 相邻方向](docs/10-other-directions.md) | 投机解码、latent/循环推理、扩散、字节/视觉表示、系统成本 |

## 证据与阅读入口

本轮采用**综述发现 → 原论文方法/实验/局限 → 前作与后续/竞争方案 → 跨来源核对**。不把引用量或热度当作质量证明，不把多个相关实验包装成独立共识，不拼接不同模型/任务/基线的数字作统一排名。

| 入口 | 内容 |
| --- | --- |
| [来源协议与索引](sources/README.md) | 原论文/官方文档、阅读章节、模型/任务/预算、结论与局限 |
| [分类对照和发现链](sources/coverage.md) | 八个综述入口的交叉分类、补漏过程、跨阶段方法和未穷尽范围 |
| [企业公开实践](sources/vendor-practice.md) | OpenAI/Anthropic的已披露方案与未披露边界 |
| [视频入口](sources/videos.md) | 作者Bilibili报告、YouTube课程；明确实际访问程度 |
| [图像来源](assets/README.md) | 保留图片、论文原资产、许可和校验和 |
| [第二版交付核查](meta/research-map-audit.md) | 内容审查、机械检查、视觉与GitHub检查的实际范围 |

资料截止日：**2026-09-09**。这是一份选择性地图，未运行新的模型训练实验，也尚未通过读者对照实验证明“idea重复率显著降低”。每章的最近邻表和否证问题用于支持这项目标，不能冒充已测结果。

## 推导和教学程序

完整公式、手算、练习与旧教程移至[关联技术专题](docs/technical/README.md)；旧章节文件名和重要入口继续可用。另保留[直接控制推理长度专题](docs/efficient-reasoning-recipes.md)和[术语表](docs/glossary.md)。这些程序用真实题目或明确标注的教学模拟检查逻辑，不构成模型效率结果。

在仓库根目录运行，Python 3.10或更新版本，无需GPU或付费接口：

```bash
python3 examples/evaluation_lab.py
python3 examples/data_audit.py
python3 examples/tokenization_lab.py
python3 examples/posttraining_lab.py
python3 examples/context_lab.py
python3 examples/agent_budget_lab.py
python3 examples/efficient_reasoning_lab.py
python3 -m unittest discover -s tests -v
python3 examples/posttraining_lab.py --self-test
python3 scripts/check_content.py
```

数值配图用`requirements-plots.txt`中的绘图库及`scripts/make_figures.py`生成。表格是原生Markdown，公式和路线关系图使用GitHub支持的数学与Mermaid语法。

[贡献方式](CONTRIBUTING.md)。第一版的导航呈现参考了[Datawhale Hello-Agents](https://github.com/datawhalechina/hello-agents)；本项目独立编写，不是Datawhale官方项目。

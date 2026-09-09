# Token-efficient LLM

**围绕同样的任务性能、更少的实际 token，理解研究问题、路线关系、代表方法、证据与行动入口。**

这是一份中文领域研究地图，面向准备进入该领域的研究者。主目录按 13 个研究方向组织，生命周期与效率机制提供交叉索引。优先考虑同等性能下曲线向左移动，极大预算下的能力上限作为相关分支。

## 从这里开始

**[总论](docs/00-overview.md) → 一个方向的章节概要与方法详解 → 同章 `reference/` 论文精读 → [选题与最小实验](docs/research-opportunities.md)。**

| 章节 | 阅读入口 |
| --- | --- |
| [00 总论](docs/00-overview.md) | 全领域关系、成本口径、稳定认识与争议 |
| [01 数据准备与合成](docs/chapters/01-data/index.md) | 本章概要、机制详解、原图、精读与行动入口 |
| [02 预训练与 Scaling](docs/chapters/02-pretraining/index.md) | 本章概要、机制详解、原图、精读与行动入口 |
| [03 中训练与持续适配](docs/chapters/03-midtraining/index.md) | 本章概要、机制详解、原图、精读与行动入口 |
| [04 模型架构、表示与内部计算](docs/chapters/04-architecture/index.md) | 本章概要、机制详解、原图、精读与行动入口 |
| [05 后训练与对齐](docs/chapters/05-posttraining/index.md) | 本章概要、机制详解、原图、精读与行动入口 |
| [06 推理策略、验证与规划](docs/chapters/06-reasoning/index.md) | 本章概要、机制详解、原图、精读与行动入口 |
| [07 Prompt、知识、检索、长上下文与记忆](docs/chapters/07-context/index.md) | 本章概要、机制详解、原图、精读与行动入口 |
| [08 Harness、智能体与工具使用](docs/chapters/08-agents/index.md) | 本章概要、机制详解、原图、精读与行动入口 |
| [09 多模态与世界模型](docs/chapters/09-multimodal/index.md) | 本章概要、机制详解、原图、精读与行动入口 |
| [10 效率系统与部署](docs/chapters/10-systems/index.md) | 本章概要、机制详解、原图、精读与行动入口 |
| [11 评测、诊断与可解释性](docs/chapters/11-evaluation/index.md) | 本章概要、机制详解、原图、精读与行动入口 |
| [12 安全、隐私、可靠性与治理](docs/chapters/12-safety/index.md) | 本章概要、机制详解、原图、精读与行动入口 |
| [13 领域模型、多语言与 AI for Science](docs/chapters/13-domains/index.md) | 本章概要、机制详解、原图、精读与行动入口 |

## 根据问题选择路线

| 研究问题 | 建议连读 |
| --- | --- |
| 现成模型怎样少读、少调用 | 07 上下文 → 08 智能体 → 06 停止/验证 → 11 评测 |
| 短推理怎样训练，难题是否受损 | 01 数据 → 05 后训练（含 The Art）→ 06 预算策略 → 11 评测 |
| 上游训练能否减少部署 token | 02 预训练 → 03 中训练 → 04 架构 → 11 成本与证据链 |
| 多模态、系统或领域方法改变了什么单位 | 09 多模态 → 10 系统 → 13 领域 → 12 可靠性约束 |

主成本是整项任务所有方法侧模型调用的**实际输入＋输出 token**，包含辅助调用、重试与失败。训练成本、费用、延迟和内部非文本计算另列。只压短单轮输出、提高吞吐或换 tokenizer，不能自动证明任务前沿改善。

## 文献、图示与质量记录

本版整合 **43 篇独立论文精读**；较早的局部来源卡与候选记录保留其原有阅读状态。由主代理先汇总综述候选、统一去重，再按唯一论文分派 Luna。两篇校准样稿达标后批量生产，批量精读完成 4 次抽查；未对其余精读逐篇进行正文覆盖普查。

- [生命周期、机制索引与覆盖边界](sources/coverage.md)
- [来源与阅读协议](sources/README.md) · [统一文献台账](sources/papers.json)
- [培训与样稿校准](meta/research-map-v3/calibration.md) · [实际抽查记录](meta/research-map-v3/qc-samples.json)
- [论文原图来源](assets/papers-v3/README.md) · [路线图设计与来源](assets/maps-v3/README.md)
- [第三版交付检查](meta/research-map-v3/delivery.md)
- [企业公开实践](sources/vendor-practice.md) · [视频入口及实际访问程度](sources/videos.md)

资料截止日：**2026-09-09**。这是选择性研究地图，包含 2026 年新工作及更新版综述。未运行新的模型训练实验；研究机会是待检验判断，不是已验证贡献。

## 关联推导与教学程序

[技术专题](docs/technical/README.md)、[直接控制推理长度专题](docs/efficient-reasoning-recipes.md)和[术语表](docs/glossary.md)保留。旧章节 URL 提供新目录与历史版本入口。教学程序不构成模型效率实验。

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
python3 scripts/research_registry.py check
python3 scripts/check_content.py
python3 scripts/check_research_map.py
```

环境为 Python 3.10 或更新版本，以上检查无需 GPU 或付费模型接口。[贡献方式](CONTRIBUTING.md)。本项目独立编写，不是 Datawhale 官方项目；第一版导航曾参考 [Hello-Agents](https://github.com/datawhalechina/hello-agents)。

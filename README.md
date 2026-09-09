# Token-efficient LLM：推理时 token 效率研究地图

**同样的任务性能，更少的实际推理 token。** 本地图按 13 个研究方向组织，生命周期与效率机制作为交叉索引。资料截止日为 2026-09-09。

从[总论](docs/00-overview.md)开始，再选择一个章节。每章的 `index.md` 包含本章概要和方法详解；需要读完整论文内容时，点击正文的精读链接，进入同章 `reference/`。精读文件直接采用方法名，例如 `TokenSkip.md`；论文编号保留在文献台账。当前版本共有 13 章、43 篇共享精读。

| 章节 | 阅读内容 |
| --- | --- |
| [00 总论](docs/00-overview.md) | 目标概念图、研究全景、证据边界和阅读路径 |
| [01 数据准备与合成](docs/01-data/index.md) | 本章概要、方法详解、论文精读和行动入口 |
| [02 预训练与 Scaling](docs/02-pretraining/index.md) | 本章概要、方法详解、论文精读和行动入口 |
| [03 中训练与持续适配](docs/03-midtraining/index.md) | 本章概要、方法详解、论文精读和行动入口 |
| [04 模型架构、表示与内部计算](docs/04-architecture/index.md) | 本章概要、方法详解、论文精读和行动入口 |
| [05 后训练与对齐](docs/05-posttraining/index.md) | 本章概要、方法详解、论文精读和行动入口 |
| [06 推理策略、验证与规划](docs/06-reasoning/index.md) | 本章概要、方法详解、论文精读和行动入口 |
| [07 Prompt、知识、检索、长上下文与记忆](docs/07-context/index.md) | 本章概要、方法详解、论文精读和行动入口 |
| [08 Harness、智能体与工具使用](docs/08-agents/index.md) | 本章概要、方法详解、论文精读和行动入口 |
| [09 多模态与世界模型](docs/09-multimodal/index.md) | 本章概要、方法详解、论文精读和行动入口 |
| [10 效率系统与部署](docs/10-systems/index.md) | 本章概要、方法详解、论文精读和行动入口 |
| [11 评测、诊断与可解释性](docs/11-evaluation/index.md) | 本章概要、方法详解、论文精读和行动入口 |
| [12 安全、隐私、可靠性与治理](docs/12-safety/index.md) | 本章概要、方法详解、论文精读和行动入口 |
| [13 领域模型、多语言与 AI for Science](docs/13-domains/index.md) | 本章概要、方法详解、论文精读和行动入口 |
| [研究机会](docs/research-opportunities.md) | 最近邻、最小区分实验、否证条件和资源需求 |

[附录：基础推导、算例、术语与基准](docs/appendix/README.md)仅供按需查阅；它按用途组织，没有另一套章节编号。必要的方法机制保留在章节正文。

[生命周期与机制索引](sources/coverage.md) · [文献台账与证据](sources/README.md) · [历史版本](VERSIONS.md)

## 仓库怎么用

`docs/` 是唯一的当前阅读目录；`assets/` 保存正文图像，`sources/` 保存文献证据，`meta/` 保存维护与检查记录。`examples/`、`data/`、`scripts/`、`tests/` 分别提供教学程序、数据、维护工具与测试。历史整套正文仅在历史分支保存。

[本次整理与检查记录](meta/delivery.md) · [偏好依据](meta/preferences.md) · [贡献方式](CONTRIBUTING.md)

## 教学程序与维护检查

以下程序在仓库根目录运行，无需 GPU。教学输出不构成新的模型效率实验。

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
python3 scripts/check_math.py
python3 scripts/build_navigation.py --check
```

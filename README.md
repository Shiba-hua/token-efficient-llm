# Token-efficient LLM

**用更少的 token 完成同样好的任务，用同样的 token 完成更好的任务。**

一本面向初学者的中文研究教程，从数据、预训练和后训练，一直讲到上下文、工具智能体与评测。阅读目标是理解方法改变了什么、能够手算关键公式，并运行小实验检查结论。

![性能—token 成本前沿的概念图](assets/generated/frontier-concept.png)

图为模型生成的概念插画，没有实验数值。横轴统计方法运行时所有模型调用的输入与输出 token，纵轴是同一任务分布上的性能。数值曲线、代码和统计口径见[总论](docs/00-overview.md)与[评测章](docs/09-evaluation.md)。

## 从哪里开始

| 阅读路线 | 建议顺序 | 读完能做什么 |
| --- | --- | --- |
| 快速建立地图 | 00 → 07 → 08 → 09 | 分清 token、费用、延迟；选一个强基线和评测协议 |
| 理解训练方法 | 00 → 01 → 04 → 05 → 06 | 理解 SFT、偏好优化、GRPO、OPD 如何改变策略 |
| 系统学习 | 按 00–10 顺序 | 把数据、模型、部署系统和评测连接起来 |

只需要基础 Python、概率和导数知识。每章先讲直觉，再定义公式；不熟悉的术语可查[术语表](docs/glossary.md)。代码实验主要使用 Python 标准库，不需要 GPU 或付费模型接口。它们验证公式和计量逻辑；真实大模型性能需另行运行模型实验。

## 章节导航

| 章节 | 主要问题 | 动手实验 |
| --- | --- | --- |
| [00 总论：问题与研究地图](docs/00-overview.md) | 什么才是曲线向左上移动？哪些是共同基础，哪些仍在探索？ | [前沿与完整成本账](examples/evaluation_lab.py) |
| [01 各阶段数据准备](docs/01-data.md) | 数据质量、过程监督和去污染分别保证什么？ | [数据审计](examples/data_audit.py) |
| [02 预训练](docs/02-pretraining.md) | NTP、MTP、分词和数据选择如何影响部署能力？ | [分词与训练目标](examples/tokenization_lab.py) |
| [03 中训练](docs/03-midtraining.md) | 领域、长上下文、推理能力怎样继续学习？ | [小语言模型实验](examples/tokenization_lab.py) |
| [04 SFT 与蒸馏](docs/04-sft-distillation.md) | 教师轨迹、学生轨迹和 OPD 有什么不同？ | [后训练算例](examples/posttraining_lab.py) |
| [05 偏好优化](docs/05-preference-rl.md) | RLHF、RLAIF、PPO、DPO 在优化什么？ | [后训练算例](examples/posttraining_lab.py) |
| [06 RLVR 与 Agentic-RL](docs/06-rlvr-agentic.md) | 奖励、组内优势、长度成本和跨步信用如何结合？ | [后训练算例](examples/posttraining_lab.py) |
| [07 Prompt 与上下文](docs/07-prompt-context.md) | 怎样减少读入内容、避免压掉关键证据？ | [上下文实验](examples/context_lab.py) |
| [08 Harness 与多智能体](docs/08-harness-agents.md) | 怎样分配工具、推理、验证和通信预算？ | [智能体预算实验](examples/agent_budget_lab.py) |
| [09 评测与典型题目](docs/09-evaluation.md) | OckBench、OTB、THINK-Bench、AppWorld 如何判分？ | [评分与统计实验](examples/evaluation_lab.py) |
| [10 相邻方向](docs/10-other-directions.md) | 推测解码、潜在推理、扩散、量化和 MoE 改变哪种成本？ | 对照不同计量单位 |

专题阅读：[直接控制推理长度的三条路线](docs/efficient-reasoning-recipes.md)，连接 TokenSkip、长度奖励 RL 和 s1 budget forcing，附[压缩选择与奖励算例](examples/efficient_reasoning_lab.py)。

## 运行教学实验

在仓库根目录运行，建议 Python 3.10 或更新版本：

```bash
python3 examples/evaluation_lab.py
python3 examples/data_audit.py
python3 examples/tokenization_lab.py
python3 examples/posttraining_lab.py
python3 examples/context_lab.py
python3 examples/agent_budget_lab.py
python3 examples/efficient_reasoning_lab.py
python3 -m unittest discover -s tests -v
```

需要重新生成数值配图时，安装 `requirements-plots.txt` 中的绘图库并运行 `python3 scripts/make_figures.py`。数学公式由 GitHub Markdown 原生渲染，表格保留可复制文本。

## 如何阅读证据

- **基础范式**表示多个独立配方广泛采用，不表示它对 token 效率全局最优。
- **典型方法**表示可研究、可复用的具体方案，不自动构成行业共识。
- **教学算例**只说明公式；**论文结果**限定在原论文的模型、数据和预算；本教程不把两者混成新的实验结果。
- 基准的论文与代码可能不一致。评测章写明版本、统计口径和已核实的差异，不能拿一个版本的脚本声称复现另一个版本的数字。

资料核查基准日：**2026-09-09**。这是一份有选择的研究地图，不声称穷尽全部论文。[来源与图像归属](assets/README.md)、[贡献方式](CONTRIBUTING.md)、[交付核查](meta/completion-audit.md)。

章节导航和面向初学者的呈现方式参考 [Datawhale Hello-Agents](https://github.com/datawhalechina/hello-agents)。本教程独立编写，不是 Datawhale 官方项目，也未直接复制其正文或截图表格。

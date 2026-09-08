# 教程交付核查

核查日期：2026-09-09。正文、教学程序、图像复审、GitHub 发布及页面检查已完成。此清单保留原始目标，不以文件数量替代内容验收；[发布记录](github-render-check.md)明确所核对的正文版本与检查范围。

| 原始要求 | 应有的直接证据 | 当前状态 |
| --- | --- | --- |
| GitHub 新建 public 仓库 | [公开仓库](https://github.com/Shiba-hua/token-efficient-llm)；公共 API、远端提交与 CI 的[实查记录](github-render-check.md) | 完成 |
| 总—分结构，覆盖八类方向 | [README 导航](../README.md)、00–10 正文、[逐章审读](content-review.md) | 完成 |
| 总论定义性能—token 前沿 | [总论](../docs/00-overview.md)：完整成本账、实际消耗与预算参数、Pareto 条件 | 完成 |
| 区分行业共识与典型方法 | 总论跨阶段证据表；各章一手来源；广泛采用与最优性分别表述 | 完成 |
| 技术推导与可运行代码 | [七个程序实际输出](example-runs.json)；六项评分单元测试、后训练算术自检通过 | 本地及远端 Python 3.11 / 3.13 检查通过 |
| GRPO 技术细节准确 | [第 06 章](../docs/06-rlvr-agentic.md)分开解释分母 epsilon 与 clipping epsilon；全对、全错及成本奖励算例 | 完成 |
| 内置图像模型绘制概念图 | [生成资产](../assets/generated/frontier-concept.png)、[提示词](../assets/generated/frontier-prompt.txt)；图注明确非实验曲线 | 完成 |
| 代码生成数值图 | [绘图脚本](../scripts/make_figures.py)、固定教学数据、五张数值/题目配图 | 完成 |
| 论文原图来自 LaTeX 源工程 | [资产溯源](../assets/papers/provenance.json)：两个原始图资产与源包成员逐字节一致；AppWorld PNG 为原图 PDF 的显示副本 | 完成 |
| 全部图位视觉验收 | [视觉记录](visual-audit.md)：19 / 19 图位通过，共 22 条独立审查记录；两轮单图修复保留历史 | 完成，范围为完整资产与 Markdown 上下文 |
| 原生文本表格 | README 与全部章节实际显示 21 张 HTML 文本表格 | GitHub 渲染检查通过 |
| 每个详细 benchmark 有 3–10 个真实典型点 | [第 09 章](../docs/09-evaluation.md)：四个基准各三个可定位例子；[数据索引](../data/benchmarks/) | 完成，AppWorld 使用论文公开场景，非完整测试包记录 |
| 评测方法、标准、公式、代码和配图 | OckScore、AUCOAA、THINK 过程效率、AppWorld TGC/SGC；完整成本、成对 bootstrap 和失败分母 | 完成 |
| 补充调研，修正前期误读 | [数据与预训练来源](foundations-sources.md)、[后训练来源](posttraining-sources.md)、[上下文来源](context-sources.md)；新增长度控制专题 | 完成 |
| 初学者可快速学习 | 三条阅读路线、[术语表](../docs/glossary.md)、逐章练习、标准库 CPU 算例与完整图位检查 | 完成；14 页加载与解析检查通过，未声称完成读者学习效果实验 |

## 教学内容边界

正文共 11 章，另有[直接控制推理长度的专题](../docs/efficient-reasoning-recipes.md)。七个程序用于运行公式、计量和微型合成实验。所有数值图均说明数据是教学合成值，还是来自基准题面的几何或网格。两张论文原图保留作者、固定源包版本及许可。

OpenMathReasoning、OpenCodeReasoning 和 TextbookReasoning 的质量说法已回到官方卡、论文及原始讨论核查。第 01 章明确：判分重复不一致率不等于答案错误率；结构清理不证明功能正确；教材来源和两个一致的答案字段不证明逐题正确。文中两个 TBR 错例用于否定无错误保证，不用于估计整体错误率。

## 证据范围

教学程序通过只证明其所测试的公式与流程，不能证明方法提升了真实 LLM 的性能。实际训练、大模型推理实验不属于本教程已经取得的结果；需要按照评测章协议另行采集。

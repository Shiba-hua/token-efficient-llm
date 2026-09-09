# 第三版交付与检查范围

资料截止：2026-09-09。执行偏好见 [research-map-preferences 对照](preferences.md)，任务契约见 [workflow.md](workflow.md)。

## 实际交付

- 13 个方向章，每章 `index.md` 同时包含本章概要、方法机制、原图、证据边界与行动入口；独立总论在各章完成后收束。
- 43 篇共享精读：2 篇培训校准稿＋41 篇批量原文稿。每篇一个主归属；跨章复用同页。
- 10 篇综述分类/候选提取；183 条统一工作/来源记录，区分精读、综述提取、历史局部和未读候选。
- 生命周期与机制索引、最近邻与最小实验、旧章节迁移入口。旧正文由固定 Git 版本提供，旧片段入口保留。
- 13 张章节地图＋最后生成的 1 张全景图。主代理设计关系并落盘；论文原图与生成概念图分别记录来源。

图像目录有 335 条素材映射，包含别名及组成面板，不能当成独立图数。主页面实际有 61 处逻辑图位：章节 60 处、总论 1 处。[素材来源](../../assets/papers-v3/README.md) · [路线图来源](../../assets/maps-v3/README.md)

## 检查完成到哪一层

| 范围 | 实际动作与结果 | 不代表什么 |
| --- | --- | --- |
| 培训 | The Art、LLMLingua-2 首稿反馈后完成 v2 校准 | 不把首稿问题隐去，也不重复首读 |
| 第四级生产 | 4 次正文抽查，均及格；局部数学/表述问题定点修正 | 没有逐篇正文覆盖普查或逐篇作者图质量普查 |
| 主代理综合 | 13 章机制、关键成本边界与[重要原文综合复核](main-synthesis-checks.json) | 没有将全部原文重新精读一遍 |
| 图示 | 61 个主页面逻辑图位由独立 Luna medium 逐图检查；修复后通过 | 不含所有 reference 页图位的全量视觉检查 |
| 机械检查 | 唯一身份、活动任务、精读产物、章节/链接、图像校验和 | 不以字数、标题数或代理退出码验收科研内容 |
| 教学程序 | 13 个单元测试、7 个 lab、后训练算术自测通过 | 没有运行新模型训练或得到新的效率实验结论 |

章节图检查发现并修复 01/03/07 的连线，以及 TokenSkip/OPSD 原图预览的大幅白边；原 PDF 不变。原始失败与复查结果见 [visual-findings.json](visual-findings.json)，逐章范围见 [chapter-visual-audit.json](chapter-visual-audit.json)。总论图见 [overview-visual-audit.json](overview-visual-audit.json)。

机械命令和实际退出结果见 [mechanical-checks.json](mechanical-checks.json)。

综述候选元数据的独立问题及重新培训见 [事故记录](incidents.json)；它不用于扩大第四级正文抽查范围。抽查实际范围见 [qc-samples.json](qc-samples.json)。

## Git 与渲染交付

修订分支为 `codex/research-map-v3`。前序调度契约提交为 `63464eec844711ed0d51e3e4305e204384189c79`，已推送并核对远端。正文、精读和图像提交为 `2f14841cbeb875c4fbac28b4ddf5cbd0a2eac243`，已推送并通过 `git ls-remote` 核对；[该提交的 GitHub Actions](https://github.com/Shiba-hua/token-efficient-llm/actions/runs/34352905980) 已成功。实际 GitHub 检查发现并修正行内公式的定界符问题，记录见 [github-math-formatting.json](github-math-formatting.json)。

用户原有未跟踪 `drafts/` 保留，不纳入本次提交。没有合并到 `main`，也没有运行或发布任何新的模型实验。

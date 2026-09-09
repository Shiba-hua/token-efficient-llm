# OpenAI 与 Anthropic：公开方案证据

阅读日期：2026-09-09。阅读者：主代理。发现路径：官方文档主题检索 → 官方研究/工程页面 → 相关产品报告。以下为原文转述；产品存在、厂商报告的性能与独立科研结论分别处理。动态文档未锁定历史 revision，复现实验应另行固定实际 API/model 版本。

## V01 — OpenAI reasoning effort

- 来源：[Reasoning models](https://developers.openai.com/api/docs/guides/reasoning)，动态 API 文档。
- 已读：Reasoning effort、token 计量相关说明。
- 机制：由调用者设置推理强度，模型也会按任务复杂度调整思考。不同模型支持值不同。
- 证据类型：V/T；这是可调用的预算控制接口，没有给出统一任务集上的等性能节省比例。降低 effort 只改变工作点，不能单独证明前沿改善。
- 归属：推理预算控制、后训练与部署接口之间的交叉。不能反推私有训练损失或认定为 GRPO。

## V02 — OpenAI GPT-5.1 adaptive reasoning

- 来源：[Introducing GPT-5.1 for developers](https://openai.com/index/gpt-5-1-for-developers/)，2025-11-13。
- 已读：Adaptive reasoning、no reasoning mode、Coding 与评测条件。
- 机制：针对简单任务减少思考，复杂任务继续探索与检查。
- 实验：发布页的 npm 查询单例，GPT-5/5.1 均为 medium，约 250→50 token、10→2 秒。这是示范点，不能作为总体均值。SWE-bench 的其他结果不能与这个单例拼成曲线。
- 证据类型：V/T；厂商报告，训练细节不公开。客户引语未在本库作为独立控制实验使用。

## V03 — OpenAI GPT-5.1-Codex-Max

- 来源：[Building more with GPT-5.1-Codex-Max](https://openai.com/index/gpt-5-1-codex-max/)，2025-11-19。
- 已读：Introduction、Speed and cost、Long-running tasks、评测附录。
- 机制：公开声称原生训练支持跨 context window 的 compaction，以及更有效的推理。
- 实验：SWE-bench Verified 上，medium 对 medium，报告优于 GPT-5.1-Codex 且少 **30% thinking tokens**。页面附录的 73.7%→77.9% 对应 high→xhigh，**不能配到这项 30% 比较上**。
- 证据类型：V/T；只支持厂商所述 thinking token 改善，不能推广成完整输入+输出 token 的相同降幅，也不能分离训练与 harness 的因果贡献。

## V04 — OpenAI compaction

- 来源：[Compaction](https://developers.openai.com/api/docs/guides/compaction)，动态 API 文档。
- 已读：Overview、Server-side compaction、Standalone compaction 与输出处理。
- 机制：阈值触发或显式调用压缩，将此前关键状态和推理保存在更短的、不透明加密条目及保留项目中，供后续窗口使用。
- 证据类型：V/T；文档确认接口和减少上下文的意图，未给统一成功率—整项任务 token 曲线。不透明条目不是可据此断言的 Coconut 式连续隐状态架构。
- 限制：压缩调用本身、后续重读与遗漏导致的重试都需要计入实验。页面版本会更新。

## V05 — OpenAI tool search

- 来源：[Tool search](https://developers.openai.com/api/docs/guides/tools-tool-search)，动态 API 文档。
- 已读：工具发现机制、defer_loading、namespace/MCP 使用说明。
- 机制：不把全部工具 schema 预先塞进上下文；发现需要的工具后再加载定义。文档披露模型受过搜索 namespaces/MCP 的训练。
- 证据类型：V/T；直接减少不必要的定义输入，新增发现步骤需要一起计量。没有由此推断私有检索器、损失函数或统一节省比例。

## V06 — OpenAI prompt caching

- 来源：[Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)，动态 API 文档。
- 已读：Why prompt caching matters、usage 与缓存费用的计算说明。
- 机制：复用相同前缀的计算，缓存输入在 usage 中单列并以不同费率计价。
- 证据类型：V/C；避免重复计算和降低费用不等于逻辑输入 token 消失。正文不使用随产品变化的折扣数字作研究结果。

## V07 — OpenAI o1：RL 与推理时计算

- 来源：[Learning to reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/)，2024-09-12。
- 已读：训练/推理扩展说明、Evals、Chain of Thought、附录评测表。
- 机制：通过 RL 训练推理行为，再增加推理时计算。具体优化器未披露。
- 实验：AIME 2024 的正文报告单次、64 样本共识、1,000 样本重排逐步提高成绩；它们的样本预算显著不同，不能合并为 pass@1。
- 证据类型：V/I；支持推理能力与计算预算关系，不提供等性能少 token 的统一结论。

## V08 — OpenAI deliberative alignment

- 来源：[Deliberative alignment: reasoning enables safer language models](https://openai.com/index/deliberative-alignment/)，2024-12-20。
- 已读：Method、Results 和训练流程图说明。
- 机制：用包含安全规范推理的合成轨迹做 SFT，再由能访问规范的奖励模型指导 RL。
- 证据类型：V/I；原目标是安全与恰当拒绝，不能冒充 token 效率方法。它说明效率优化仍需要保留任务约束，不能把必要推理统一当冗余。

## V09 — Anthropic advanced tool use

- 来源：[Introducing advanced tool use on the Claude Developer Platform](https://www.anthropic.com/engineering/advanced-tool-use)，2025-11-24，Bin Wu 等。
- 已读：Tool Search Tool、Programmatic Tool Calling、Tool Use Examples 的机制、内部测试、适用与不适用条件。
- 机制：按需工具发现；让程序调用、聚合和过滤工具结果；用示例减少参数错误。
- 实验：工具发现示范的初始上下文约 77K→8.7K；复杂研究任务中程序化调用平均 43,588→27,297 tokens（约 37%）。后者的准确率证据来自其他内部测试，不能把不同任务集拼成一个等性能点。
- 证据类型：V/T。仅厂商内部结果；工具少、输出小的任务收益可能抵不过新增发现/执行开销。工具示例会增加输入，需判断能否减少失败与重试。

## 使用边界

上述材料足以说明头部企业已经提供预算调节、压缩、动态工具加载、程序化编排等公开方案。它们**不能证明企业已经采用本仓库讨论的某个未披露训练算法**。跨厂商出现相近接口可支持工程问题和方案具有实际需求，仍不足以证明任何实现对所有模型与任务最优。

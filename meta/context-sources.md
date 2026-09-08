# 第 7、8、10 章来源与教学示例核查记录

核查日期：2026-09-09（Asia/Shanghai）。用途：为教程提供可追踪的一手来源。章节中的会计恒等式、合成场景和实验设计建议属于本教程的讲解；不将其标为论文复现或真实模型结果。

原始资料线索来自已有公开研究梳理，正文的事实已重新核对下面的论文、作者代码或官方技术文档。核查不代表执行了论文中的模型实验，也不代表阅读了每篇论文所有附录。

| ID | 一手来源 | 本次核查范围 | 支持的章节内容 | 使用边界 |
|---|---|---|---|---|
| CTX-01 | [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning, v2](https://arxiv.org/abs/2507.19457v2) | arXiv 元数据和摘要 | 7.2：读取轨迹、自然语言反思、提出并评估提示更新 | 未把 rollout 搜索效率改写成部署 token 节省；未引述具体胜率 |
| CTX-02 | [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, v4](https://arxiv.org/abs/2005.11401v4) | 元数据和摘要 | 7.3：生成模型结合外部可检索记忆 | 词项重合代码是教学例子，不是原始 RAG 复现 |
| CTX-03 | [LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression, v2](https://arxiv.org/abs/2403.12968v2) | 元数据、摘要；[官方仓库](https://github.com/microsoft/LLMLingua)说明 | 7.4：蒸馏数据、双向编码器、token 分类式压缩 | 抽取保真不等于任意任务的逻辑等价；教程公式仅为机制示意 |
| CTX-04 | [The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management, v3](https://arxiv.org/html/2508.21433v3) | 摘要，正文策略定义与结果讨论；作者链接的代码与数据位置 | 7.5：旧观察遮蔽与摘要需要正面比较；特定 SWE-agent 环境下遮蔽是强基线 | 正文明确论文费用口径、环境限制，不将其推广为所有 Agent 的 token 结论 |
| CTX-05 | [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) | Introduction、Example workloads、Limits；在线页面核查 | 7.8：共享前缀 KV 复用与预填充/解码边界 | 在线文档会变动；不引用具体默认参数或供应商价格 |
| HAR-01 | [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) | 元数据和摘要 | 8.1：推理与动作交替组织 | 本章状态与预算结构是本教程设计，不声称为原论文规范 |
| HAR-02 | [Code execution with MCP: Building more efficient agents](https://www.anthropic.com/engineering/code-execution-with-mcp) | 原作者工程文章：工具定义、代码接口、按需加载、结果筛选 | 8.3：在运行环境中筛选中间数据，按需加载工具定义 | 作为工程机制证据；没有引用文章中的示例节省比例作为通用效果 |
| HAR-03 | [RouteLLM: Learning to Route LLMs with Preference Data, v4](https://arxiv.org/abs/2406.18665v4) | 元数据和摘要 | 8.5：用偏好数据学习在强弱模型间路由，权衡质量与费用 | 合成阈值实验不是 RouteLLM 复现，模型价格低不等于 token 少 |
| OTH-01 | [Fast Inference from Transformers via Speculative Decoding, v2](https://arxiv.org/html/2211.17192v2) | 元数据、摘要、算法正文 | 10.2：草稿、目标验证、接受与残差校正、目标分布保持 | 简化时间模型列明假设；单位置 Python 算术不是完整解码实现 |
| OTH-02 | [Training Large Language Models to Reason in a Continuous Latent Space, v4](https://arxiv.org/abs/2412.06769v4) | 元数据和摘要；[官方仓库](https://github.com/facebookresearch/coconut)说明 | 10.3：Coconut 将最后隐藏状态作为后续输入 embedding | 未推广论文的逻辑任务效果；内部更新步骤仍需独立计量 |
| OTH-03 | [Large Language Diffusion Models, v3](https://arxiv.org/html/2502.09992v3) | 元数据、摘要和生成方法正文；[官方仓库](https://github.com/ML-GSAI/LLaDA)说明 | 10.4：前向掩码与反向生成，Transformer 预测掩码 token | 位置访问数是本教程全序列假设下的教学计数，不是实测 FLOPs |
| OTH-04 | [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) | 元数据和摘要 | 10.5：利用激活信息辅助低比特权重量化 | 均匀量化公式仅提供入门直觉，不称 AWQ 完整算法；不引用通用加速比 |
| OTH-05 | [Mixtral of Experts](https://arxiv.org/abs/2401.04088) | 元数据和摘要 | 10.6：每 token 激活部分前馈专家，总参数与激活参数区分 | 参数公式是简化结构计算，不直接作为 FLOPs 或显存测量 |

## 可执行示例与验证

这两个文件只依赖 Python 标准库，不下载权重、不需要账号或 API key，不运行真实 LLM。默认命令已在本次写作中执行且进程退出码为 0：

```bash
python3 examples/context_lab.py
python3 examples/agent_budget_lab.py
```

`context_lab.py` 默认输出：

- raw：输入 6960，输出 160，总量 7120。
- mask：输入 3990，输出 160，总量 4150。
- summary：主输入 3580，主输出 160，压缩输入输出 945，总量 4685。
- 最后上下文中植入事实的字符串检查：raw=True、mask=False、summary=True。
- 词项重合检索：26 到 6 个空白分隔教学单位。
- 缓存：完整输入 2000，缓存部分 900，未缓存部分 1100，输出 100，总量 2100；人为价格权重下为 1490。

`agent_budget_lab.py` 默认输出：

- 预算 1000，前两次调用累计 670，第三次拒绝，最终调用后累计 770。
- 合成路由阈值 0.6：人为成功标签均值 0.9，平均教学总量 340。阈值 0.0 为 0.9 和 520；这里对照也统一计入 20 单位路由开销，正文解释了强基线可移除此项。
- 四工作 Agent、三轮、消息长 100：全互联通信为生成 1200、读取 3600，共 4800；设定的星形通信同为 4800。
- 精确工具过滤：100 条合成记录，失败 ID 为 7、81；空白分隔单位从 4600 到 5。
- 推测采样单位置校正：目标与恢复分布均为 [0.1, 0.6, 0.3]，接受概率 0.6。
- 最终输出长度 128；全序列扩散 16 轮对应 2048 次位置访问，明确不是 FLOPs 或输出 token。
- 十亿权重理想存储：16 bit 为 2,000,000,000 字节，4 bit 为 500,000,000 字节。

脚本内的断言验证默认会计恒等式和样例预期。它们不验证真实模型成功率、压缩保真性、论文复现或硬件加速。

## 图形与范围

三个资产均为本教程自行编写的 SVG，无论文截图或来源图重绘声明：

- `assets/plots/context-pipeline.svg` → 第 7 章图 7-1。
- `assets/plots/harness-loop.svg` → 第 8 章图 8-1。
- `assets/plots/harness-resource-metrics.svg` → 第 10 章图 10-1。

XML 解析和 Markdown 引用目标存在性已检查。编写阶段调用系统 Quick Look 渲染时出现 sandbox initialization error，未取得视觉验收结果；完整视觉审计由集成流程另行记录。本记录不把结构检查称为视觉通过。

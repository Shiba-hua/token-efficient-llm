# 术语速查

| 术语 | 本教程中的含义 |
| --- | --- |
| Token | tokenizer 切分后的离散单位，可能是字、词片段或字节片段；不等同于汉字数或单词数 |
| Pareto 前沿 | 在所比较的可达点中，没有其他点同时成本更低、性能更高的集合 |
| Budget | 运行前指定的资源约束；与实际消耗分开记录 |
| NTP / MTP | Next-token / Multi-token Prediction，下一或多个未来 token 预测 |
| 中训练 | 预训练与最终后训练之间、用于领域/上下文/能力适配的继续训练阶段；各论文边界不完全一致 |
| SFT | Supervised Fine-tuning，监督微调 |
| RLHF / RLAIF | 基于人类反馈 / AI反馈的强化学习；先描述反馈来源，具体算法另定 |
| RLVR | Reinforcement Learning from Verifiable Rewards，用可验证结果提供奖励 |
| PPO / GRPO | 近端策略优化 / 组相对策略优化；策略梯度方法，具体实现与奖励仍需说明 |
| DPO | Direct Preference Optimization，直接偏好优化 |
| OPD | On-policy Distillation，在学生当前策略访问的状态或轨迹上取得教师监督 |
| KL | Kullback–Leibler divergence，分布间的不对称差异；方向和采样分布必须写清 |
| Harness | 包裹模型的运行系统：输入、工具、状态、预算、重试、终止与日志 |
| Trajectory | 一次完整任务运行的观测、思考、动作与反馈序列 |
| Verifier | 判断结果的规则、测试或模型；可靠程度由覆盖范围与误差决定 |
| LLM-as-a-judge | 用语言模型判分；重复一致性和真实正确率是不同指标 |
| Grounding | 将指令对应到界面元素或可执行动作；正确定位不保证整项任务完成 |
| Decontamination | 对训练与目标评测材料做去重/相似性排查；不能证明未知基础训练中完全未见 |
| SFT utility | 数据带来的下游训练收益；不等于数据标签正确率 |
| Cached input | 已读入前缀的计算复用；仍可计入原始输入token，费用可能另计 |
| Speculative decoding | 草稿生成后由目标模型验证，主要改变执行时间与计算分配 |

[返回目录](../README.md)

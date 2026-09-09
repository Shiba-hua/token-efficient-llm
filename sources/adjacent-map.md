# 相邻方向：原文记录

阅读日：2026-09-09。Luna 原文阅读，主代理整合。由高效推理综述的投机、latent、diffusion、多模态分支及其原文引用追读；字节表示见[早期 E05](early-stages.md)，Coconut/CODI 见[后训练 P11/P12](posttraining-map.md)。

## A01 — Speculative Decoding

[Leviathan、Kalman、Matias，Fast Inference from Transformers via Speculative Decoding，2211.17192v2](https://arxiv.org/abs/2211.17192v2)，ICML 2023。已读 §1–6、附录 A.1–A.5。

小模型草拟多个 token，大模型并行验证，并通过拒绝采样保持目标分布。T5-XXL/小 T5，WMT/CNN-DM，单 TPU-v4、batch1；翻译示例加速约 2.6× sampling/3.4× greedy。类型 C；最终 token 没减少，额外算术可能增加，硬件和 batch 决定净收益。

## A02 — EAGLE-3

[Li 等，Scaling up Inference Acceleration of Large Language Models via Training-Time Test，2503.01840](https://arxiv.org/abs/2503.01840)，2025。已读 §1–5、训练/吞吐附录；沿 EAGLE/EAGLE-2 前作链进入。

融合目标模型多层特征，直接预测草稿 token，并模拟草稿自反馈训练，减轻多步误差。Vicuna/Llama/R1-Distill 等；SGLang/H100、8B、batch1 的 throughput 158.34→373.25 token/s，大 batch 增益较小。类型 C；提高生成速度，不减少最终任务 token，draft 训练成本另计。

## A03 — Dream 7B

[Ye 等，Diffusion Large Language Models，2508.15487v1](https://arxiv.org/html/2508.15487v1)，2025-08 提交；文内工作日期更早。已读训练、§5.1–5.5、quality-speed 分析。

从 Qwen2.5-7B 自回归权重适配离散去噪，调噪声计划和扩散步数；约 580B 继续训练 token，与 LLaDA/Qwen 等比较知识、数学、代码与规划。类型 I/C；继承原基座的训练成本不能忽略。减少去噪轮数不是减少最终 token，任务间质量/速度优势不同。

## A04 — Recurrent Depth / Huginn

[Geiping 等，Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach，2502.05171v2](https://arxiv.org/html/2502.05171v2)，2025-02。已读 §1–8、附录 A–C；由 latent reasoning 引文追读。

在中间 Transformer core 反复更新隐状态，训练随机展开，部署可调 recurrence。3.5B、约 0.8T 训练 token，比较不同深度和固定深度基线。类型 C/I：增加内部计算可提高某些任务表现，但没有完整等性能 I/O 节省表；新架构和大规模训练对小团队成本高。

## A05 — FastV

[Chen 等，An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Acceleration for VLLM Inference，2403.06764v3](https://arxiv.org/abs/2403.06764v3)，2024-09。已读 §2–5、任务附录。

在浅层用注意力选择视觉 token，再在后层剪枝。LLaVA/Qwen-VL/Video-LLaVA 等；LLaVA-13B、K=2、剪50%时理论 FLOPs 154.6B→84.6B，单A40 A-OKVQA延迟0.539→0.341秒。类型 C/T（内部视觉表示）；不是文本 token 或 API image billing 必然下降，高压缩损伤 OCR/细粒度信息。

## A06 — LLaVA-PruMerge

[Shang、Cai、Xu、Lee、Yan，Adaptive Token Reduction for Efficient Large Multimodal Models，2403.15388v6](https://arxiv.org/abs/2403.15388v6)，2026-01-31 版本，ICCV 2025。已读 §1–4.4、§5 与算法附录，沿 FastV/视觉压缩引用链进入。

在 CLIP 编码器末端按 CLS–patch attention 的离群值选 token，再按 key 相似度将被删位置加权合并；PruMerge+ 增加空间均匀采样。与 FastV 在 LLM 内剪枝的位置不同。LLaVA-1.5 Vicuna-7B/13B，336×336 原始 576 视觉 token；主表 LoRA 微调一轮，不是全部 training-free。7B 的约 5.5% 剩余 token 配置，VQAv2 78.5→72.0、POPE 85.9→76.3；25% 的 PruMerge+ 恢复到 76.8/84.0，仍有损失。固定约 40 token 的 V100 prefill 时间是 roofline 估计，不是端到端实测。类型 C/T（内部视觉），不能与最终生成长度或 API 图像账单混为一谈。

# 02 预训练：每个 token 承载什么，每次计算学会什么

[返回目录](../README.md) · [上一章：数据](01-data.md) · [下一章：中训练](03-midtraining.md)

预训练可以改变表示、知识与计算分配，但本章多数代表工作优化的是**训练或每 token 的计算效率**。它们是领域地图不可遗漏的上游，不能据此宣布部署任务用了更少 token。对小团队，更现实的作用通常是选择基座、复用配方，或在小规模上检验机制。

## 1. 四条路线及其关系

| 路线 | 代表工作 | 直观机制 | 与部署 token 的距离 |
| --- | --- | --- | --- |
| 数据与模型规模怎样配比？ | Chinchilla → Beyond Chinchilla-Optimal | 训练最优先看训练预算；生命周期最优还看未来要服务多少请求 | 常减少每 token FLOPs，未必缩短回答 |
| 同样文本怎么编码？ | BPE/Unigram 对照、Tokenizer Choice → BLT | 改变文本被拆成多少符号；BLT 按难度把字节合成 patch | 表示长度变了，必须避免换计量尺造成假改善 |
| 一个位置应提供多少训练信号？ | NTP → Multi-token Prediction → DeepSeek-V3 MTP | 除下一个词外，同时学习后续位置 | 可改善样本效率和投机解码，不必减少最终 token |
| 哪些参数/历史需要参与计算？ | MoE、MLA、白盒 token/KV 选择 | 每个 token 少激活参数或少保存历史状态 | 主要属于[计算效率相邻方向](10-other-directions.md) |

[原文记录 E03–E06、E14–E15](../sources/early-stages.md)分别标出模型、任务和证据口径。NTP 是基础训练范式，不能把介绍交叉熵本身当成 token 效率研究成果。

## 2. 从训练最优到部署最优：两种 scaling 问题

Chinchilla 比较固定训练 FLOPs 下的参数量和训练数据，说明当时一些大模型训练不足；更小但读更多数据的模型可以更强。Beyond Chinchilla-Optimal 把未来推理计算加入目标：如果服务请求很多，可能值得提前多训练一个较小模型，换取未来每次更低计算。[Chinchilla](https://arxiv.org/abs/2203.15556)、[Beyond Chinchilla-Optimal](https://arxiv.org/html/2401.00448v3)

两者不是“多训练一定省 token”的证据。后者甚至把部署输入/输出长度当作外部需求，其优化对象是计算成本。要与本项目连接，应再问：更充分训练的模型能否以更少推理步骤完成相同任务？这需要新增测量，不能由 scaling law 自动推出。

## 3. 分词、词表和字节 patch：先分清单位

Tokenizer Choice 控制 BPE/Unigram、词表大小和多语言设计，发现相同文本可能被拆成很不同的长度，但低 fertility 不保证下游质量最好。扩词表会增加嵌入/输出头等成本；数字、代码和低资源语言的最佳切分也未必一致。[Tokenizer Choice](https://aclanthology.org/2024.findings-naacl.247/)

BLT 则不再依赖固定词表：局部字节模型按熵划 patch，大 Transformer 在 patch 级工作。它把“容易内容少算、难内容多算”放进表示结构。这里减少的是昂贵计算位置；一个 patch 和一个 BPE token 没有统一信息量。[BLT](https://arxiv.org/html/2412.09871v1)

所以跨 tokenizer 比较至少同时报告实际各模型 token、原文 bytes/字符、任务性能与计算/费用。如果仅把整段话换成一个大词表符号，原始 token 轴会左移，但不能说明模型掌握了更经济的解法。地图将这种改进标成“表示效率”，与“同 tokenizer 下推理缩短”并列。

## 4. MTP 的两条效果链不能混在一起

MTP 给共享主干多个未来位置的预测目标，使它更早捕捉后续结构。训练后可以只保留普通下一 token 头，也可让辅助头提出多个候选、由模型验证。前者研究学得更好，后者研究生成得更快。[MTP 原文](https://arxiv.org/html/2404.19737v1)

![MTP 原论文结构图](../assets/papers/mtp-main.png)

保留作者原图用于说明多位置预测结构，来源与许可见[图像归属](../assets/README.md)。多头数不是最终回答压缩率。

原 MTP 在代码上表现出样本效率和解码加速，但自然语言选择题与不同训练量下的最佳预测跨度不同。DeepSeek-V3 的技术报告把 MTP 纳入大规模训练并报告 TPS 提升，是工程采用证据；MoE、MLA、数据与后训练同时改变，不能把整个模型能力归因于 MTP。[DeepSeek-V3 §2/§3](https://arxiv.org/html/2412.19437v2)

## 5. 关键实验与目前的认识

| 工作 | 原实验条件/结果 | 能支持什么，不能支持什么 |
| --- | --- | --- |
| Chinchilla | 70B、1.4T 训练 token，与 Gopher 相近训练 FLOPs | 小模型充分训练可更有竞争力；未测回答更短 |
| Beyond Chinchilla | 47 个 MPT 模型、150M–6B，结合部署需求拟合 | 大量服务时训练/推理计算应联合考虑；更大模型结论含外推 |
| Tokenizer Choice | 2.6B、70B words、24 个分词配置、多语言任务 | 编码显著影响长度/能力；fertility 不是统一质量指标 |
| MTP | 13B 代码模型；7B 四 token 模式约 3× 解码加速 | 样本效率与加速；不等于少生成三分之二 token |
| BLT | 最大 8B/4T bytes，FLOPs 控制；最高约 50% 推理 FLOPs 取舍 | 动态表示能改变计算前沿；不是同单位 token 曲线 |

“训练、表示和服务预算应一起考虑”有多条研究路线支持；具体配比、tokenizer 和未来预测跨度仍依赖任务、规模与实现。OpenAI/Anthropic 并未在本次读取的公开材料中披露足够预训练细节，不能把上述任一机制写成它们已采用；已公开内容见[企业证据](../sources/vendor-practice.md)。

## 6. 阅读顺序与研究边界

从 Chinchilla→Beyond Chinchilla 理解优化目标，再从 Tokenizer Choice→BLT 理解单位，最后读 MTP→DeepSeek-V3 区分原理与整套配方。

| 初步 idea | 最近邻 | 要建立新贡献还缺什么 |
| --- | --- | --- |
| 预训练更久的小模型更便宜 | Beyond Chinchilla-Optimal | 必须实测是否也减少任务推理 token，而不只每 token 成本 |
| 一次预测多个 token 就更高效 | MTP、投机解码 | 分别测样本效率、最终序列长度和服务吞吐 |
| 把常见片段合成更大的 token | Tokenizer Choice、BLT | 固定信息内容与任务质量，排除仅仅换计数单位 |

小团队可以比较公开基座在同一推理预算曲线上的变化，或做受控 tokenizer/辅助目标消融；从头训练新大架构的成本与证据要求明显更高。前者是可行性判断，尚不是被验证的新方法。

公式与代码移至[预训练技术专题](technical/02-pretraining.md)与[tokenization_lab.py](../examples/tokenization_lab.py)。

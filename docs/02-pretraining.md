# 章节入口已迁移

本地图第三版按研究方向组织。请从以下入口阅读当前正文：

- [02 预训练与 Scaling](chapters/02-pretraining/index.md)
- [04 模型架构、表示与内部计算](chapters/04-architecture/index.md)

[总论](00-overview.md) · [旧版教学专题](technical/02-pretraining.md) · [第二版历史正文](https://github.com/Shiba-hua/token-efficient-llm/blob/d3e4e4bc52ec9b13b12b08ea7de4c55d1fdd6fdd/docs/02-pretraining.md)

<!-- 保留旧版小节片段，避免旧链接失效。 -->
<a id="02-预训练从预测下一个-token-到理解样本效率"></a>
<a id="学习目标"></a>
<a id="1-直觉一句文本提供许多有条件的预测题"></a>
<a id="2-从序列概率推导训练损失"></a>
<a id="3-数据选择已有证据与可以迁移的实验方法"></a>
<a id="4-分词改变计量单位也改变学习难度"></a>
<a id="5-多-token-预测增加监督不必然减少输出"></a>
<a id="6-cpu-实验亲手看见-token-和条件概率"></a>
<a id="7-局限与练习"></a>
<a id="主源参考"></a>
<a id="02-预训练每个-token-承载什么每次计算学会什么"></a>
<a id="1-四条路线及其关系"></a>
<a id="2-从训练最优到部署最优两种-scaling-问题"></a>
<a id="3-分词词表和字节-patch先分清单位"></a>
<a id="4-mtp-的两条效果链不能混在一起"></a>
<a id="5-关键实验与目前的认识"></a>
<a id="6-阅读顺序与研究边界"></a>

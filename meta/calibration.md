# Luna 校准记录

两篇样稿（The Art of Efficient Reasoning v3、LLMLingua-2 ACL2024）均先试写，再由主代理针对机制/单位/数学格式给出具体反馈。第一轮需要修改，第二轮达到用户指定的中等偏上参照，可批量使用 luna-reading-v2。

第一轮问题包括样本与分块计数、训练与推理显存、mask与正信号混淆、全零GRPO组边界、数学定界符及语义保真过度外推。第二轮保留正文与主实验的实质覆盖，修正上述问题，增加可跟踪算例并取得作者原图映射。这是培训阶段修订，不是对批量精读的普查。

批量规则：每10篇抽1篇；及格继续；明确缺漏定点修正；严重虚构或整体偏离则记录事故并重新培训。主代理不会把未抽查稿件标为逐篇复核。

[培训说明v2](luna-reading-prompt-v2.md) · [样稿检查记录](qc-samples.json) · [事故记录](incidents.json)

## 综述候选元数据的后续校准

收尾发现作者姓名扩写虚构，记录为 survey-metadata-01。按同份提取补查两项，完成3份已有PDF首页和1个缩写教学案例的有限校准；经两次具体反馈后达标。使用 [survey-extraction-v2](survey-extraction-prompt-v2.md)，字段结果与范围见 [survey-metadata-calibration.json](survey-metadata-calibration.json)。原始论文阅读 v2 prompt 不受此配置变更影响。

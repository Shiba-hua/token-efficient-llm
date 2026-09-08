# 图片与数据归属

## 图像来源

| 资产 | 来源与用途 | 修改/许可 |
| --- | --- | --- |
| `generated/frontier-concept.png` | Codex 内置图像模型生成的性能—token前沿概念插画 | [最终提示词](generated/frontier-prompt.txt)；2026-09-09生成；非实验结果 |
| `plots/evaluation-*.png` | `scripts/make_figures.py` 生成 | 除官方迷宫网格及AMO题面几何外，性能数值均为合成教学数据 |
| `plots/*.svg` | 对应章节教学程序或教程作者自绘 | 图注区分教学单位与实际模型token |
| `papers/mtp-main.png` | Fabian Gloeckle 等，Better & Faster Large Language Models via Multi-token Prediction，arXiv:2404.19737v1，源包 `img/main_fig_col.png` | 原字节提取，CC BY 4.0 |
| `papers/appworld-main.pdf` | Harsh Trivedi 等，AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents，arXiv:2407.18901v1，源包 `images/main.pdf` | 原字节提取，CC BY 4.0 |
| `papers/appworld-main.png` | 上述单独图像PDF的PNG显示副本 | `pdftoppm -png -r 144 -singlefile` 转换；未截取论文页面，内容未编辑 |

原始论文图的作者、源包网址、成员路径、SHA256和许可链接见 [provenance.json](papers/provenance.json)。作者没有为本教程背书。图中模型名、应用图标及商标归各自权利人。

## 典型数据点

`data/benchmarks/` 保留少量教学例子的官方定位及中文转述。OTB迷宫来自固定版本的真实网格，图中橙色路径是本教程计算后添加的解释。它们不是新发布的独立基准，不能拿这些已讲解的题作盲测。

基准内容及上游代码保留各自许可，不因本仓库许可发生变更。AppWorld这里只转述作者在论文中公开的场景，未重新分发解密后的受保护测试任务包。

`plots/research-map.svg` 为本教程自行绘制的研究流程图；SVG 本身为可编辑原始源码，无外部图像或实验数据。

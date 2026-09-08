# GitHub 发布与显示核查

核查日期：2026-09-09。正文与图像版本：[171932705547bf74f8b80b5f7cce31989c5e9c6d](https://github.com/Shiba-hua/token-efficient-llm/tree/171932705547bf74f8b80b5f7cce31989c5e9c6d)。本记录随后作为验收文档提交；逐页原始计数见 [github-render-check.json](github-render-check.json)。

## 实际页面检查

使用浏览器打开 GitHub 上的 README、11 个章节、长度控制专题和术语表，共 14 个页面。浏览器视口为 1280 × 720；检查渲染后的文章 DOM，并目视抽查首页插画、成本公式、文本表格和章节入口。

| 项目 | 实测结果 |
| --- | --- |
| 数学表达式 | 428 / 428 个完成原生 MathML 渲染，与源码审读数量一致 |
| 数学类型 | 358 个行内公式、70 个块公式 |
| 解析错误 | 0 个公式错误提示；正文中 0 处遗留未解析的美元定界符 |
| 原生表格 | 21 张 HTML 文本表格，不是图片 |
| 图片加载 | 19 / 19 处完成加载，包含首页重复使用的概念图 |
| 页面宽度 | 所查文章的 scrollWidth 未超过 clientWidth |

初次显示检查发现行内公式紧贴中文标点时漏识别、普通 Markdown 转义影响数学字符串，以及渲染器拒绝 `operatorname` 宏。修复采用受保护的行内数学、`math` 围栏、比较符号命令和直立函数名。独立审读确认 428 个公式顺序、类型、数学条件及含义不变，19 个非数学代码围栏逐字一致；所有图像与图注语义保持一致。

该检查证明指定版本、指定视口下的加载与解析结果，并结合了有限位置的目视检查。它不是所有滚动位置或所有设备宽度的像素级验收。完整图片与 Markdown 上下文的逐图审查另见 [视觉验收记录](visual-audit.md)。

## 公开性与自动检查

GitHub 公共 API 返回 `private=false`、`visibility=public`、默认分支 `main`。上述正文版本已推送，远端分支与本地提交核对一致。

[Tutorial checks 运行 34256596408](https://github.com/Shiba-hua/token-efficient-llm/actions/runs/34256596408)对应上述正文提交，结果为 `completed / success`。Python 3.11 与 3.13 两个矩阵 job 均实际完成：

- Markdown 文件、围栏与本地链接检查；
- 六项评分单元测试；
- 七个标准库教学程序；
- 后训练梯度、裁剪、优势与奖励的算术自检。

后续提交的运行记录由 [GitHub Actions](https://github.com/Shiba-hua/token-efficient-llm/actions/workflows/ci.yml) 持续保存。自动检查不证明论文训练结果已复现，也不替代引用与图像审读。

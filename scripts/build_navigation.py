"""Build the two reader entrypoints from one chapter manifest."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def render():
    chapters = json.loads((ROOT / 'meta/chapters.json').read_text())
    outputs = {}
    for target in ['README.md', 'docs/README.md']:
        def link(path):
            return os.path.relpath(path, str(Path(target).parent))
        table = '| 章节 | 阅读内容 |\n| --- | --- |\n'
        table += f'| [00 总论]({link("docs/00-overview.md")}) | 目标概念图、研究全景、证据边界和阅读路径 |\n'
        for c in chapters:
            table += f'| [{c["slug"][:2]} {c["title"]}]({link(c["page"])}) | 本章概要、方法详解、论文精读和行动入口 |\n'
        table += f'| [研究机会]({link("docs/research-opportunities.md")}) | 最近邻、最小区分实验、否证条件和资源需求 |\n'
        text = '# Token-efficient LLM：推理时 token 效率研究地图\n\n' if target=='README.md' else '# 研究地图阅读目录\n\n'
        text += '**同样的任务性能，更少的实际推理 token。** 本地图按 13 个研究方向组织，生命周期与效率机制作为交叉索引。资料截止日为 2026-09-09。\n\n'
        text += f'从[总论]({link("docs/00-overview.md")})开始，再选择一个章节。每章的 `index.md` 包含本章概要和方法详解；需要读完整论文内容时，点击正文的精读链接，进入同章 `reference/`。精读文件直接采用方法名，例如 `TokenSkip.md`；论文编号保留在文献台账。当前版本共有 13 章、43 篇共享精读。\n\n'
        text += table
        text += f'\n[附录：基础推导、算例、术语与基准]({link("docs/appendix/README.md")})仅供按需查阅；它按用途组织，没有另一套章节编号。必要的方法机制保留在章节正文。\n\n'
        text += f'[生命周期与机制索引]({link("sources/coverage.md")}) · [文献台账与证据]({link("sources/README.md")}) · [历史版本]({link("VERSIONS.md")})\n\n'
        if target=='README.md':
            text += '## 仓库怎么用\n\n`docs/` 是唯一的当前阅读目录；`assets/` 保存正文图像，`sources/` 保存文献证据，`meta/` 保存维护与检查记录。`examples/`、`data/`、`scripts/`、`tests/` 分别提供教学程序、数据、维护工具与测试。历史整套正文仅在历史分支保存。\n\n'
            text += '[本次整理与检查记录](meta/delivery.md) · [偏好依据](meta/preferences.md) · [贡献方式](CONTRIBUTING.md)\n\n'
            text += '## 教学程序与维护检查\n\n以下程序在仓库根目录运行，无需 GPU。教学输出不构成新的模型效率实验。\n\n```bash\npython3 examples/evaluation_lab.py\npython3 examples/data_audit.py\npython3 examples/tokenization_lab.py\npython3 examples/posttraining_lab.py\npython3 examples/context_lab.py\npython3 examples/agent_budget_lab.py\npython3 examples/efficient_reasoning_lab.py\npython3 -m unittest discover -s tests -v\npython3 examples/posttraining_lab.py --self-test\npython3 scripts/research_registry.py check\npython3 scripts/check_content.py\npython3 scripts/check_research_map.py\npython3 scripts/check_math.py\npython3 scripts/build_navigation.py --check\n```\n'
        else:
            text += '[返回项目首页](../README.md)。本目录中的总论和 13 个章目录组成完整正文；不需要再寻找 `chapters/` 或 `technical/`。\n'
        outputs[target]=text
    return outputs

def main():
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');p.add_argument('--check',action='store_true');args=p.parse_args()
    mismatches=[]
    for path,text in render().items():
        target=ROOT/path
        if args.write:target.write_text(text)
        elif not target.exists() or target.read_text()!=text:mismatches.append(path)
    if mismatches:raise SystemExit('Navigation differs from chapter manifest: '+', '.join(mismatches))
    print('Navigation: 13 chapter links and overview share one manifest.')
if __name__=='__main__':main()

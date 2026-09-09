"""Generate tutorial plots. All performance curves are synthetic illustrations.

The OTB maze is a rendering of a sourced task grid, not a paper screenshot.
"""
from collections import deque
from pathlib import Path
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "teaching"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 160})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def frontier():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.array([500, 1000, 2000, 4000, 8000])
    ax.plot(x, [.35, .50, .63, .72, .76], "o-", label="Baseline", color="#2369ae")
    ax.plot(x, [.42, .58, .68, .74, .73], "s-", label="Candidate", color="#bf3838")
    ax.set(xlabel="Actual total inference tokens / task", ylabel="Task success fraction",
           title="Observed budget points can cross", ylim=(.25, .85))
    ax.annotate("High-budget regression", xy=(8000,.73), xytext=(4200,.52),
                arrowprops={"arrowstyle":"->"})
    ax.legend(loc="lower right")
    fig.text(.5, -.01, "Synthetic teaching data — no LLM experiment", ha="center", fontsize=10)
    save(fig, "evaluation-frontier.png")


def aucoaa():
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    t=np.arange(1001)
    lengths=np.array([0,200,500,1200]); correct=np.array([1,1,0,1])
    oaa=(correct[None,:]*(lengths[None,:]<t[:,None])).mean(axis=1)
    axes[0].step(t,oaa,where="post",color="#2369ae")
    axes[0].fill_between(t,oaa,step="post",alpha=.15,color="#2369ae")
    axes[0].set(xlabel="Thinking-token threshold t",ylabel="OAA(t)",title="Thresholding existing responses",ylim=(0,1))
    axes[1].bar(["Paired metric","Product of means"],[.5,.25],color=["#2369ae","#bf3838"])
    axes[1].set(ylabel="AUCOAA-like score",ylim=(0,.65),title="Same two responses; different formulas")
    axes[1].text(.5,.58,"correct = [1, 0]; lengths = [0, 1000]",ha="center",fontsize=10)
    fig.tight_layout()
    fig.text(.5,-.03,"Synthetic responses — illustrates the effect of pairing",ha="center",fontsize=10)
    save(fig,"evaluation-aucoaa.png")


def maze():
    rows=json.loads((ROOT/"data/benchmarks/otb-examples.json").read_text())
    item=next(x for x in rows if "grid" in x)
    grid=item["grid"]
    assert len(grid)==21 and all(len(row)==21 for row in grid)
    start=next((r,c) for r,row in enumerate(grid) for c,s in enumerate(row) if s=="]")
    goal=next((r,c) for r,row in enumerate(grid) for c,s in enumerate(row) if s=="\\")
    queue=deque([start]); parents={start:None}
    while queue:
        r,c=queue.popleft()
        if (r,c)==goal:break
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nxt=(r+dr,c+dc)
            if (0<=nxt[0]<21 and 0<=nxt[1]<21 and grid[nxt[0]][nxt[1]]!="."
                    and nxt not in parents):
                parents[nxt]=(r,c);queue.append(nxt)
    if goal not in parents:raise ValueError("Sourced maze cannot be solved")
    path=[];node=goal
    while node is not None:path.append(node);node=parents[node]
    assert len(path)-1==int(item["answer"])
    fig,ax=plt.subplots(figsize=(6,6))
    ax.imshow([[int(s==".") for s in row] for row in grid],cmap="Greys",vmin=0,vmax=1)
    ax.plot([c for r,c in path],[r for r,c in path],color="#dd7b22",linewidth=3)
    for label,(r,c),color in [("S",start,"#126bb1"),("G",goal,"#bf3838")]:
        ax.scatter([c],[r],color=color,s=200,zorder=3)
        ax.text(c,r,label,color="white",ha="center",va="center",weight="bold",zorder=4)
    ax.set_xticks([]);ax.set_yticks([])
    ax.set_title("OTB maze / source_index=0\nShortest path: 15 steps")
    fig.text(.5,.025,"Actual released grid; BFS path added by this tutorial",ha="center",fontsize=9)
    save(fig,"evaluation-otb-maze.png")


def triangle():
    fig,ax=plt.subplots(figsize=(6,5))
    for i in range(5):
        for j in range(5-i):
            x=i+j/2;y=j*math.sqrt(3)/2
            ax.scatter(x,y,s=65,color="#2369ae")
            for di,dj in ((1,0),(0,1),(-1,1)):
                ni,nj=i+di,j+dj
                if ni>=0 and nj>=0 and ni+nj<=4:
                    ax.plot([x,ni+nj/2],[y,nj*math.sqrt(3)/2],color="#bbb",lw=.8,zorder=0)
    ax.set_aspect("equal");ax.axis("off")
    ax.set_title("OckBench / AMO-0\n15 points; each side divided into 4")
    fig.text(.5,.04,"Select n points: when is an isosceles triangle unavoidable?",ha="center",fontsize=10)
    save(fig,"evaluation-ock-triangle.png")


def think_ratio():
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    axes[0].barh(["Early correct","Late correct"],[100,900],label="Prefix to first correct",color="#2369ae")
    axes[0].barh(["Early correct","Late correct"],[900,100],left=[100,900],label="Remaining",color="#bcc9d5")
    axes[0].set(xlabel="Thinking tokens",title="Identical total length: 1000")
    axes[0].legend(loc="upper center",bbox_to_anchor=(.5,-.18),fontsize=9)
    axes[1].bar(["Early correct","Late correct"],[.1,.9],color=["#2369ae","#bf3838"])
    axes[1].set(ylabel="Prefix / total",title="A higher ratio need not save tokens",ylim=(0,1))
    fig.tight_layout()
    fig.text(.5,-.13,"Synthetic counterexample to treating this ratio as a Pareto improvement",ha="center",fontsize=9)
    save(fig,"evaluation-think-ratio.png")


if __name__=="__main__":
    frontier();aucoaa();maze();triangle();think_ratio()
    print("Wrote 5 plots; actual OTB maze independently checked by BFS (15 steps).")

# -*- coding: utf-8 -*-
"""
================================================================================
 树状图 v3：横向扇形布局，无重叠
 Дерево v3: горизонтальная «веерная» раскладка без перекрытий.
================================================================================

 布局 / Раскладка:
   Корень (всего транзакций)
     └─ L1: самые частые товары (в ряд)
          └─ от каждого товара веером вниз 3 частые пары (y = пары)
               └─ от пары — её тройка (y = тройки)

 列间距远大于框宽，左右不重叠；层级在不同 y，上下不重叠。
 Шаг между колонками больше ширины рамки; уровни на разной высоте.

 用法 / Использование:
   python make_tree.py
================================================================================
"""

import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apriori import load_transactions, apriori

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "baskets.csv")
FIG = os.path.join(ROOT, "figures", "fig4_tree.png")

MIN_SUP = 0.01
TOP_L1 = 5
TOP_PAIRS = 3
TOP_TRIPLES = 1

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# ---- 几何参数 / Геометрия ----
Y_ROOT, Y_L1, Y_PAIR, Y_TRI = 5.6, 4.3, 2.7, 1.3
DX_L1 = 7.0             # 相邻高频单品水平间距 / шаг между L1
DX_PAIR = 2.3           # 同一分支内二元组水平间距 / шаг пар внутри ветви
W_L1, W_PAIR, W_TRI = 2.3, 2.05, 2.2
H_BOX = 0.62


def short(name: str, n: int = 13) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


def node(ax, x, y, lines, color, w):
    p = FancyBboxPatch((x - w / 2, y - H_BOX / 2), w, H_BOX,
                       boxstyle="round,pad=0.06", linewidth=1.6,
                       edgecolor=color, facecolor="#ffffff")
    ax.add_patch(p)
    ax.text(x, y, "\n".join(lines), ha="center", va="center",
            fontsize=8.2, color="#222222")


def arrow(ax, x1, y1, x2, y2):
    a = FancyArrowPatch((x1, y1 - H_BOX / 2), (x2, y2 + H_BOX / 2),
                        arrowstyle="-|>", mutation_scale=9,
                        linewidth=0.9, color="#aaaaaa")
    ax.add_patch(a)


def main():
    transactions = load_transactions(DATA)
    n = len(transactions)
    freq = apriori(transactions, max(1, int(round(MIN_SUP * n))))

    by_len = defaultdict(list)
    for s, c in freq.items():
        by_len[len(s)].append((s, c))
    for k in by_len:
        by_len[k].sort(key=lambda x: -x[1])

    # 选分支：top-L1，每个 top-pairs / top-triples
    branches = []
    for itemset, cnt1 in by_len[1][:TOP_L1]:
        it = next(iter(itemset))
        pairs = sorted([(s, c) for s, c in by_len[2] if itemset <= s],
                       key=lambda x: -x[1])[:TOP_PAIRS]
        chosen = []
        for pair, cnt2 in pairs:
            triples = sorted([(s, c) for s, c in by_len.get(3, []) if pair <= s],
                             key=lambda x: -x[1])[:TOP_TRIPLES]
            chosen.append((pair, cnt2, triples))
        branches.append((it, cnt1, chosen))

    # 坐标 / Координаты
    l1_xs = [i * DX_L1 for i in range(len(branches))]
    root_x = (l1_xs[0] + l1_xs[-1]) / 2
    x_min, x_max = l1_xs[0] - 3.2, l1_xs[-1] + 3.2

    fig, ax = plt.subplots(figsize=(24, 7.2))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.4, 6.3)
    ax.axis("off")

    C_ROOT, C_L1, C_L2, C_L3 = "#3b5b92", "#4C72B0", "#55A868", "#C44E52"

    node(ax, root_x, Y_ROOT, ["Корень", f"всего {n}"], C_ROOT, 2.6)

    for i, (it, cnt1, pairs) in enumerate(branches):
        x1 = l1_xs[i]
        # L1 单品 | L1-товар
        node(ax, x1, Y_L1,
             [short(it, 17), f"{cnt1} ({100*cnt1/n:.1f}%)"], C_L1, W_L1)
        arrow(ax, root_x, Y_ROOT, x1, Y_L1)

        # 二元组在 x1 左右展开 | Пары веером вокруг x1
        pair_xs = [x1 + (k - (len(pairs) - 1) / 2) * DX_PAIR for k in range(len(pairs))]
        for (pair, cnt2, triples), px in zip(pairs, pair_xs):
            others = [e for e in pair if e != it]
            node(ax, px, Y_PAIR,
                 [f"{short(it,10)}+{short(others[0],11)}",
                  f"{cnt2} ({100*cnt2/n:.1f}%)"], C_L2, W_PAIR)
            arrow(ax, x1, Y_L1, px, Y_PAIR)
            # 三元组 | Тройка
            for tri, cnt3 in triples:
                toth = [e for e in tri if e not in pair]
                node(ax, px, Y_TRI,
                     [f"{short(it,8)}+{short(others[0],8)}+{short(toth[0],9)}",
                      f"{cnt3} ({100*cnt3/n:.1f}%)"], C_L3, W_TRI)
                arrow(ax, px, Y_PAIR, px, Y_TRI)

    ax.set_title("Дерево наиболее вероятных (частых) наборов — порог поддержки 1%\n"
                 "на узлах: количество транзакций и относительная поддержка",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    plt.close(fig)
    print("OK ->", FIG)


if __name__ == "__main__":
    main()

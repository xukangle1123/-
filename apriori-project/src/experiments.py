# -*- coding: utf-8 -*-
"""
================================================================================
 实验脚本：在固定数据集上改变支持度阈值，评估 Apriori 性能与结果规模
 Экспериментальный скрипт: фиксированный набор данных, изменяемый порог
 поддержки; оценка быстродействия Apriori и размера результата.
================================================================================

 实验内容 / Что делаем:
   1. 在阈值 1%, 3%, 5%, 10%, 15% 上分别运行 Apriori，记录运行时间
      Запускаем Apriori при порогах 1%, 3%, 5%, 10%, 15% и замеряем время.
   2. 统计各长度（1,2,3,...）频繁项集的数量
      Считаем число частых наборов разной длины.
   3. 保存结果到 results/*.csv
      Сохраняем результаты в results/*.csv.
   4. 绘制两张要求的图 + 一张辅助图
      Строим два требуемых графика + один вспомогательный.

 用法 / Использование:
   python experiments.py
================================================================================
"""

import os
import sys
import time
import csv

import matplotlib
matplotlib.use("Agg")  # 无界面后端（保存 PNG）| Безоконный бэкенд (сохранение в PNG)
import matplotlib.pyplot as plt
from collections import defaultdict

# 保证可以 import 同目录的 apriori 模块 | Чтобы импортировать модуль apriori из той же папки
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apriori import apriori, load_transactions, count_support, gen_candidates

# 项目根目录 | Корень проекта
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "baskets.csv")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# 实验用的支持度阈值（相对值）| Пороги поддержки для эксперимента (относительные)
THRESHOLDS = [0.01, 0.03, 0.05, 0.10, 0.15]

# 绘图字体（DejaVu Sans 支持西里尔字母）| Шрифт (DejaVu Sans поддерживает кириллицу)
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def run_experiment(transactions, min_sup_abs, n_txn):
    """
    在给定阈值下运行一次完整实验。
    Выполняет один полный эксперимент при заданном пороге.

    返回 / Возвращает:
      freq:      频繁项集字典 | словарь частых наборов
      runtime:   Apriori 耗时（秒）| время выполнения Apriori (сек)
      by_len:    长度 -> 数量 | длина -> количество
    """
    t1 = time.perf_counter()
    freq = apriori(transactions, min_sup_abs)
    runtime = time.perf_counter() - t1
    by_len = defaultdict(int)
    for s in freq:
        by_len[len(s)] += 1
    return freq, runtime, by_len


def main():
    print("Загрузка данных... / 正在加载数据...")
    transactions = load_transactions(DATA)
    n_txn = len(transactions)
    print(f"Транзакций: {n_txn} / 交易数: {n_txn}")

    # ---------- 1. 运行全部阈值实验 | 1. Запуск экспериментов ----------
    summary = []            # 每个阈值的汇总 | Сводка по каждому порогу
    length_data = {}        # 阈值 -> {长度: 数量} | порог -> {длина: кол-во}
    all_max_len = 0

    for thr in THRESHOLDS:
        min_abs = max(1, int(round(thr * n_txn)))
        freq, runtime, by_len = run_experiment(transactions, min_abs, n_txn)
        total = len(freq)
        summary.append({
            "threshold": thr,
            "min_abs": min_abs,
            "runtime_s": round(runtime, 4),
            "n_itemsets": total,
            "max_len": max(by_len) if by_len else 0,
        })
        length_data[thr] = dict(by_len)
        all_max_len = max(all_max_len, max(by_len) if by_len else 0)
        print(f"  порог {thr:.0%}: наборов={total}, время={runtime:.3f} с "
              f"/ 阈值 {thr:.0%}: 项集数={total}, 耗时={runtime:.3f} 秒")

    # ---------- 2. 保存汇总 CSV | 2. Сохранение сводки в CSV ----------
    with open(os.path.join(RESULTS, "summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["threshold", "min_abs", "runtime_s", "n_itemsets", "max_len"])
        w.writeheader()
        w.writerows(summary)

    # ---------- 3. 保存长度分布 CSV | 3. Сохранение распределения по длинам ----------
    with open(os.path.join(RESULTS, "lengths.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["threshold"] + [f"len_{i}" for i in range(1, all_max_len + 1)])
        for thr in THRESHOLDS:
            row = [f"{thr:.0%}"] + [length_data[thr].get(i, 0) for i in range(1, all_max_len + 1)]
            w.writerow(row)

    # ---------- 4. 图 1：性能对比（耗时 vs 阈值） ----------
    #         Рис. 1: сравнение быстродействия при разных порогах
    x = [f"{t:.0%}" for t in THRESHOLDS]
    times = [s["runtime_s"] for s in summary]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, times, color="#4C72B0", width=0.55)
    for b, v in zip(bars, times):
        ax.text(b.get_x() + b.get_width() / 2, v + max(times) * 0.02,
                f"{v:.3f} с", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Порог поддержки", fontsize=12)
    ax.set_ylabel("Время выполнения, с", fontsize=12)
    ax.set_title("Быстродействие Apriori при разных порогах поддержки", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig1_runtime.png"), dpi=150)
    plt.close(fig)

    # ---------- 5. 图 2：不同长度频繁项集数量（分组柱状图） ----------
    #         Рис. 2: число частых наборов разной длины при разных порогах
    lengths = list(range(1, all_max_len + 1))
    n_thr = len(THRESHOLDS)
    width = 0.8 / n_thr
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, thr in enumerate(THRESHOLDS):
        vals = [length_data[thr].get(l, 0) for l in lengths]
        ax.bar([l - 0.4 + width * (i + 0.5) for l in lengths], vals,
               width=width, label=f"{thr:.0%}", color=colors[i % len(colors)])
    ax.set_xticks(lengths)
    ax.set_xlabel("Длина набора", fontsize=12)
    ax.set_ylabel("Число частых наборов", fontsize=12)
    ax.set_title("Число частых наборов разной длины при разных порогах поддержки", fontsize=12)
    ax.legend(title="Порог", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig2_lengths.png"), dpi=150)
    plt.close(fig)

    # ---------- 6. 辅助图 3：频繁项集总数 vs 阈值 ----------
    #         Вспом. Рис. 3: общее число частых наборов от порога
    totals = [s["n_itemsets"] for s in summary]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, totals, marker="o", color="#C44E52", linewidth=2)
    for xi, v in zip(x, totals):
        ax.text(xi, v * 1.05, str(v), ha="center", fontsize=10)
    ax.set_xlabel("Порог поддержки", fontsize=12)
    ax.set_ylabel("Всего частых наборов", fontsize=12)
    ax.set_title("Общее число частых наборов от порога поддержки", fontsize=12)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig3_total.png"), dpi=150)
    plt.close(fig)

    print("\nГотово! Рисунки сохранены в figures/ / 完成！图表已保存到 figures/")
    print("Файлы: fig1_runtime.png, fig2_lengths.png, fig3_total.png / 文件: ...")


if __name__ == "__main__":
    main()

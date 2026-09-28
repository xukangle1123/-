# -*- coding: utf-8 -*-
"""
================================================================================
 关联规则挖掘：用置信度和提升度检验频繁"链条"
 Генерация ассоциативных правил: проверка частых цепочек через
 достоверность (confidence) и лифт (lift).
================================================================================

 说明 / Описание:
   从频繁项集出发，枚举所有非空真子集生成规则 A -> B，计算
     支持度   поддержка     = sup(A∪B)
     置信度   достоверность = sup(A∪B) / sup(A)
     提升度   лифт          = sup(A∪B) / (sup(A)·sup(B))
   并按提升度/置信度排序输出，保存到 results/。

 用法 / Использование:
   python rules_exp.py
================================================================================
"""

import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apriori import apriori, load_transactions, generate_rules

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "baskets.csv")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

# 规则实验：(阈值, 最小置信度) | Пороги и минимальная достоверность для правил
RULE_EXPS = [(0.01, 0.3), (0.03, 0.5)]
MIN_CONF_DEFAULT = 0.3


def main():
    transactions = load_transactions(DATA)
    n_txn = len(transactions)
    print(f"Транзакций: {n_txn} / 交易数: {n_txn}")

    for thr, min_conf in RULE_EXPS:
        min_abs = max(1, int(round(thr * n_txn)))
        freq = apriori(transactions, min_abs)
        rules = generate_rules(freq, min_conf, n_txn)
        # 按提升度降序，其次置信度降序 | Сортировка: лифт по убыванию, затем достоверность
        rules.sort(key=lambda r: (-r["lift"], -r["confidence"]))

        out = os.path.join(RESULTS, f"rules_{int(thr * 100)}%.csv")
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["antecedent", "consequent", "support", "confidence", "lift"])
            for r in rules:
                w.writerow([
                    ", ".join(r["antecedent"]),
                    ", ".join(r["consequent"]),
                    round(r["support"], 4),
                    round(r["confidence"], 4),
                    round(r["lift"], 4),
                ])
        print(f"порог {thr:.0%}: правил={len(rules)} -> {os.path.basename(out)} "
              f"/ 阈值 {thr:.0%}: 规则数={len(rules)}")
        # 打印前 10 条（按提升度）| Печать топ-10 по лифту
        for i, r in enumerate(rules[:10]):
            a = "{" + ", ".join(r["antecedent"]) + "}"
            c = "{" + ", ".join(r["consequent"]) + "}"
            print(f"  {i+1:>2}. {a} -> {c} | sup={r['support']:.3f} conf={r['confidence']:.2f} lift={r['lift']:.2f}")
    print("Готово! Правила сохранены в results/. / 完成！规则已保存到 results/.")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
================================================================================
 Apriori 频繁项集挖掘与关联规则生成程序
 Программа поиска частых наборов объектов и генерации ассоциативных правил
 (алгоритм Apriori и его применение к данным о покупках в супермаркете)
================================================================================

【功能概述 | Обзор возможностей】
  1. 从 CSV 文件读取交易数据（每行一笔交易，商品用逗号分隔）
     Загрузка транзакций из CSV-файла (одна строка = одна корзина, товары через запятую).
  2. 使用经典 Apriori 算法挖掘频繁项集，并输出每个项集的支持度
     Поиск частых наборов классическим алгоритмом Apriori с выводом поддержки каждого набора.
  3. 支持两种排序方式：按支持度降序 / 按字典序（лексикографически）
     Два способа упорядочивания результата: по убыванию поддержки / лексикографически.
  4. 从频繁项集生成关联规则 A -> B，并计算支持度、置信度和提升度
     Генерация ассоциативных правил A -> B с расчётом поддержки, достоверности (confidence) и лифта (lift).
  5. 命令行参数控制：数据集路径、支持度阈值、置信度阈值、排序方式
     Параметры командной строки: путь к данным, порог поддержки, порог достоверности, способ сортировки.

【用法 | Использование】
  python apriori.py --data ../data/baskets.csv --min-sup 0.05 --min-conf 0.5 --sort support
  python apriori.py --data ../data/baskets.csv --min-sup 0.05 --sort lex

【作者 | Автор】: 徐康乐 (студент, 23-й курс, ЮУрГУ-ЮАУ)
【语言 | Язык】: Python 3
================================================================================
"""

import argparse          # 命令行参数解析 | Разбор аргументов командной строки
import sys               # 系统标准输入输出 | Стандартный ввод/вывод
import time              # 计时（性能实验用）| Замер времени (для экспериментов)
import itertools         # 组合生成（子集枚举）| Генерация комбинаций (перебор подмножеств)
from collections import defaultdict  # 带默认值的字典（支持度计数）| Словарь с значением по умолчанию


# ------------------------------------------------------------------------------
# 工具函数：双语控制台输出 | Вспомогательная функция: двуязычный вывод в консоль
# ------------------------------------------------------------------------------
def log(ru: str, zh: str):
    """同时输出俄语和中文两行信息 | Выводит строку на русском и на китайском."""
    print(f"[RU] {ru}")
    print(f"[CN] {zh}")


# ------------------------------------------------------------------------------
# 1. 读取交易数据 | Загрузка транзакций
# ------------------------------------------------------------------------------
def load_transactions(path: str):
    """
    从 CSV 文件加载交易数据。
    Загружает транзакции из CSV-файла.

    说明 / Примечание:
      - 文件编码为 cp1251（俄语 Windows 常用编码）
        Кодировка файла — cp1251 (распространённая кодировка для русского Windows).
      - 第一行为表头（商品名列名），不参与挖掘
        Первая строка — заголовок (имена колонок), не участвует в анализе.
      - 每行一笔交易；同一行内以逗号分隔商品
        Каждая строка — одна корзина; товары разделены запятыми.
      - 空单元格忽略；重复商品合并（交易是集合）
        Пустые ячейки игнорируются; дубликаты схлопываются (корзина = множество).

    参数 / Параметры:
      path: CSV 文件路径 | путь к CSV-файлу

    返回 / Возвращает:
      transactions: list[frozenset[str]] — 交易列表（每笔交易是一个商品集合）
                    список транзакций (каждая транзакция — множество товаров)
    """
    transactions = []
    with open(path, "r", encoding="cp1251", newline="") as f:
        first = True
        for line in f:
            line = line.strip()
            if not line:
                continue
            if first:                     # 跳过表头 | Пропускаем заголовок
                first = False
                continue
            # 按逗号切分，去除空白，过滤空串，去重
            # Разбиваем по запятой, убираем пробелы, отбрасываем пустые, убираем дубли
            items = frozenset(x.strip() for x in line.split(",") if x.strip())
            transactions.append(items)
    return transactions


# ------------------------------------------------------------------------------
# 2. 候选集生成（Apriori 连接步 + 剪枝步）
#    Генерация кандидатов (шаг соединения + шаг отсечения в Apriori)
# ------------------------------------------------------------------------------
def gen_candidates(prev: list, k: int):
    """
    由 k-1 项频繁集生成 k 项候选集。
    Порождение k-кандидатов из (k-1)-частых наборов.

    Apriori 性质 / Свойство Apriori:
      任何频繁项集的子集必为频繁项集，因此非频繁子集的超集可以安全剪枝。
      Любое подмножество частого набора тоже часто, поэтому надмножества
      нечастых подмножеств можно смело отсекать.

    参数 / Параметры:
      prev: 上一层的频繁项集列表（每个元素是 frozenset）| список частых наборов уровня k-1
      k:    目标项集长度 | требуемая длина набора

    返回 / Возвращает:
      candidates: 候选集列表（已按字典序排序）| список кандидатов (отсортирован лексикографически)
    """
    n = len(prev)
    candidates = set()
    # 连接步 / Шаг соединения: 两两合并 k-1 项集，若它们前 k-2 项相同则连接成 k 项集
    # Объединяем пары (k-1)-наборов, у которых совпадают первые k-2 элементов.
    for i in range(n):
        for j in range(i + 1, n):
            a, b = prev[i], prev[j]
            # 将两个集合视为有序列表比较前 k-2 个元素（Apriori 标准连接条件）
            # Сравниваем первые k-2 элемента как упорядоченные списки (стандартное условие соединения)
            la, lb = sorted(a), sorted(b)
            if la[: k - 2] == lb[: k - 2]:
                candidates.add(frozenset(a | b))
    # 剪枝步 / Шаг отсечения: 删除包含非频繁 (k-1)-子集的候选
    # Удаляем кандидатов, содержащих нечастые (k-1)-подмножества.
    prev_set = set(prev)
    pruned = set()
    for cand in candidates:
        ok = True
        # 枚举候选的所有 (k-1)-子集 | Перебираем все (k-1)-подмножества кандидата
        for subset in itertools.combinations(sorted(cand), k - 1):
            if frozenset(subset) not in prev_set:
                ok = False
                break
        if ok:
            pruned.add(cand)
    # 返回字典序排序的列表（保证结果可复现）| Возвращаем отсортированный список (воспроизводимость)
    return sorted(pruned, key=lambda s: sorted(s))


# ------------------------------------------------------------------------------
# 3. 支持度计数（单遍扫描数据库）| Подсчёт поддержки (один проход по базе)
# ------------------------------------------------------------------------------
def count_support(transactions, candidates):
    """
    统计每个候选集在数据库中的出现次数（支持度计数）。
    Считает частоту встречаемости каждого кандидата в базе транзакций.

    参数 / Параметры:
      transactions: 交易列表 | список транзакций
      candidates:   候选集列表 | список кандидатов

    返回 / Возвращает:
      counts: dict[frozenset -> int]  支持度计数 | счётчики поддержки
    """
    counts = defaultdict(int)
    cand_list = list(candidates)
    # 逐笔交易检查：交易包含候选集则计数 +1
    # Для каждой транзакции: если кандидат содержится в корзине — инкремент.
    for txn in transactions:
        for cand in cand_list:
            if cand.issubset(txn):
                counts[cand] += 1
    return counts


# ------------------------------------------------------------------------------
# 4. 核心 Apriori 算法 | Ядро алгоритма Apriori
# ------------------------------------------------------------------------------
def apriori(transactions, min_sup_abs: int):
    """
    运行完整 Apriori 算法，返回所有频繁项集及其支持度。
    Выполняет полный алгоритм Apriori и возвращает все частые наборы с поддержкой.

    参数 / Параметры:
      transactions:  交易列表 | список транзакций
      min_sup_abs:   最小支持度（绝对次数）| минимальная поддержка (в абсолютных значениях)

    返回 / Возвращает:
      frequent: dict[frozenset -> int]  所有频繁项集 -> 支持度计数
               все частые наборы -> счётчик поддержки
    """
    frequent = {}                       # 结果字典 | Результирующий словарь
    n_txn = len(transactions)

    # 第一步：扫描数据库，统计所有单个商品的频率
    # Шаг 1: сканируем базу, считаем частоту всех отдельных товаров.
    item_counts = defaultdict(int)
    for txn in transactions:
        for item in txn:
            item_counts[item] += 1

    # 筛选频繁 1-项集 | Отбираем частые 1-наборы
    L1 = [frozenset([it]) for it, c in item_counts.items() if c >= min_sup_abs]
    L1.sort(key=lambda s: sorted(s))    # 字典序排序 | Лексикографическая сортировка
    for s in L1:
        frequent[s] = item_counts[next(iter(s))]

    # 迭代：由 L_{k-1} 生成 C_k，计数后筛出 L_k
    # Итерации: из L_{k-1} строим C_k, считаем поддержку, отбираем L_k.
    prev = L1
    k = 2
    while prev:
        candidates = gen_candidates(prev, k)
        if not candidates:
            break
        counts = count_support(transactions, candidates)
        # 只保留达到阈值的候选 | Оставляем кандидатов, достигших порога
        Lk = [c for c in candidates if counts[c] >= min_sup_abs]
        if not Lk:
            break
        for s in Lk:
            frequent[s] = counts[s]
        prev = Lk
        k += 1

    return frequent


# ------------------------------------------------------------------------------
# 5. 关联规则生成（含置信度与提升度）| Генерация ассоциативных правил (confidence, lift)
# ------------------------------------------------------------------------------
def generate_rules(frequent: dict, min_conf: float, n_txn: int):
    """
    从频繁项集生成关联规则 A -> B，并计算三个指标：
    Из частых наборов строит правила A -> B и вычисляет три метрики:

      支持度   поддержка     = sup(A∪B)
      置信度   достоверность = sup(A∪B) / sup(A)
      提升度   лифт          = confidence / sup(B) = sup(A∪B) / (sup(A)·sup(B))

    参数 / Параметры:
      frequent: 频繁项集 -> 支持度计数 | частые наборы -> счётчик поддержки
      min_conf: 最小置信度（0..1）| минимальная достоверность
      n_txn:    交易总数（把计数转成相对值）| общее число транзакций (для относительных значений)

    返回 / Возвращает:
      rules: list[dict] 每条规则含 antecedent, consequent, support, confidence, lift
             список правил: antecedent, consequent, support, confidence, lift
    """
    rules = []
    # 只考虑长度 >= 2 的频繁项集 | Рассматриваем только частые наборы длины >= 2
    for itemset, sup_ab in frequent.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        # 枚举真子集 A（非空、非全集）| Перебор непустых собственных подмножеств A
        for r in range(1, len(items)):
            for A in itertools.combinations(items, r):
                A = frozenset(A)
                B = itemset - A          # 规则结论 | Консеквент правила
                sup_a = frequent[A]
                confidence = sup_ab / sup_a                 # 置信度 | Достоверность
                if confidence < min_conf:                   # 过滤低置信度规则 | Отсев слабых правил
                    continue
                sup_b = frequent[B]
                lift = confidence / (sup_b / n_txn)         # 提升度 | Лифт
                rules.append({
                    "antecedent": tuple(sorted(A)),
                    "consequent": tuple(sorted(B)),
                    "support": sup_ab / n_txn,              # 相对支持度 | Относительная поддержка
                    "confidence": confidence,               # 置信度 | Достоверность
                    "lift": lift,                           # 提升度 | Лифт
                })
    return rules


# ------------------------------------------------------------------------------
# 6. 结果格式化 | Форматирование результатов
# ------------------------------------------------------------------------------
def format_itemset(s):
    """把项集转成可读字符串（商品名用 { } 包裹）| Набор -> читаемая строка."""
    return "{" + ", ".join(sorted(s)) + "}"


def print_frequent(frequent: dict, sort_mode: str, n_txn: int):
    """
    输出频繁项集列表。| Печать списка частых наборов.
    排序方式 / Способ сортировки:
      'support' — 按支持度降序 | по убыванию поддержки
      'lex'     — 按字典序（先按长度，再按商品名）| лексикографически
    """
    items = list(frequent.items())
    if sort_mode == "support":
        # 支持度降序，同支持度再按字典序（保证稳定）| По убыванию поддержки, при равенстве — лексикографически
        items.sort(key=lambda kv: (-kv[1], sorted(kv[0])))
    else:
        # 字典序：按商品名字典序 | Лексикографически: по именам товаров
        items.sort(key=lambda kv: sorted(kv[0]))
    log(f"Найдено частых наборов: {len(items)} | Порог поддержки: {n_txn and ''}", "")
    print("=" * 100)
    print(f"{'Набор (частый набор)':<55} {'Поддержка (раз)':>15} {'Поддержка (%)':>15}")
    print("-" * 100)
    for s, cnt in items:
        print(f"{format_itemset(s):<55} {cnt:>15} {100.0 * cnt / n_txn:>14.2f}%")
    print("=" * 100)


def print_rules(rules: list, top: int = None):
    """输出关联规则表（支持度/置信度/提升度）| Печать таблицы ассоциативных правил."""
    if not rules:
        log("Не найдено правил, удовлетворяющих порогу достоверности.",
            "没有找到满足置信度阈值的关联规则。")
        return
    # 默认按提升度降序 | По умолчанию сортируем по убыванию лифта
    rules_sorted = sorted(rules, key=lambda r: (-r["lift"], -r["confidence"]))
    if top:
        rules_sorted = rules_sorted[:top]
    print("=" * 100)
    print(f"{'Правило (правило)':<42} {'Поддержка':>9} {'Достоверн.':>11} {'Лифт':>8}")
    print("-" * 100)
    for r in rules_sorted:
        rule_str = f"{format_itemset(r['antecedent'])} -> {format_itemset(r['consequent'])}"
        print(f"{rule_str:<42} {r['support']:>8.3f} {r['confidence']:>10.2f} {r['lift']:>8.2f}")
    print("=" * 100)


# ------------------------------------------------------------------------------
# 7. 命令行入口 | Точка входа командной строки
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Apriori: поиск частых наборов и ассоциативных правил / 频繁项集与关联规则挖掘")
    parser.add_argument("--data", required=True,
                        help="путь к CSV-файлу с транзакциями / 交易数据 CSV 路径")
    parser.add_argument("--min-sup", type=float, default=0.05,
                        help="порог поддержки, доля от 0 до 1 (по умолчанию 0.05) / 支持度阈值 0..1")
    parser.add_argument("--min-conf", type=float, default=0.5,
                        help="порог достоверности 0..1 (по умолчанию 0.5) / 置信度阈值")
    parser.add_argument("--sort", choices=["support", "lex"], default="support",
                        help="способ упорядочивания списка наборов / 结果排序方式")
    parser.add_argument("--rules", action="store_true",
                        help="генерировать ассоциативные правила / 额外生成关联规则")
    parser.add_argument("--top", type=int, default=20,
                        help="сколько правил вывести / 输出多少条规则")
    args = parser.parse_args()

    t0 = time.perf_counter()
    transactions = load_transactions(args.data)
    n_txn = len(transactions)
    t_load = time.perf_counter() - t0

    log(f"Загружено транзакций: {n_txn}", f"已加载交易数量: {n_txn}")
    log(f"Порог поддержки: {args.min_sup:.1%}  (абсолютно: {max(1, int(args.min_sup * n_txn))} транзакций)",
        f"支持度阈值: {args.min_sup:.1%}  (绝对值: {max(1, int(args.min_sup * n_txn))} 笔交易)")

    min_sup_abs = max(1, int(round(args.min_sup * n_txn)))

    t1 = time.perf_counter()
    frequent = apriori(transactions, min_sup_abs)
    t_apriori = time.perf_counter() - t1

    log(f"Время загрузки данных: {t_load:.3f} с | Время Apriori: {t_apriori:.3f} с",
        f"数据加载耗时: {t_load:.3f} 秒 | Apriori 耗时: {t_apriori:.3f} 秒")

    print_frequent(frequent, args.sort, n_txn)

    # 关联规则：用置信度和提升度检验"链条" | Правила: проверка цепочек через confidence и lift
    if args.rules:
        log("Генерация ассоциативных правил (поддержка, достоверность, лифт)...",
            "正在生成关联规则（支持度、置信度、提升度）...")
        t2 = time.perf_counter()
        rules = generate_rules(frequent, args.min_conf, n_txn)
        t_rules = time.perf_counter() - t2
        log(f"Правил найдено: {len(rules)} | Время генерации: {t_rules:.3f} с",
            f"共发现规则: {len(rules)} 条 | 规则生成耗时: {t_rules:.3f} 秒")
        print_rules(rules, args.top)

    return frequent


if __name__ == "__main__":
    main()

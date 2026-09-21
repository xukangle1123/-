# Поиск частых наборов алгоритмом Apriori (Анализ больших данных)

Лабораторная работа: поиск частых наборов объектов в данных о покупках
(`baskets.csv`) алгоритмом Apriori, эксперименты при порогах поддержки
1%–15%, визуализация и проверка ассоциативных правил.

## Структура репозитория

```
apriori-project/
├── data/baskets.csv          # исходный набор данных (7500 транзакций, cp1251)
├── src/
│   ├── apriori.py            # основная программа: Apriori + ассоциативные правила
│   ├── experiments.py        # эксперименты (пороги 1%–15%), замер времени, рисунки
│   ├── rules_exp.py          # генерация и проверка правил (поддержка/достоверность/лифт)
│   └── build_report.py       # генерация PDF-отчёта
├── results/                  # результаты в CSV (summary, lengths, rules_*)
├── figures/                  # fig1_runtime.png, fig2_lengths.png, fig3_total.png
└── report/report.pdf         # отчёт (PDF)
```

## Как запустить

```bash
# 1) Поиск частых наборов (порог 5%, сортировка по убыванию поддержки)
python src/apriori.py --data data/baskets.csv --min-sup 0.05 --sort support

# 2) Сортировка лексикографически
python src/apriori.py --data data/baskets.csv --min-sup 0.05 --sort lex

# 3) Плюс ассоциативные правила (поддержка, достоверность, лифт)
python src/apriori.py --data data/baskets.csv --min-sup 0.01 --min-conf 0.3 --rules

# 4) Эксперименты 1%/3%/5%/10%/15% + рисунки
python src/experiments.py

# 5) Правила в CSV
python src/rules_exp.py

# 6) Отчёт PDF
python src/build_report.py
```

Требуется Python 3 с пакетами: `pandas`, `matplotlib`, `numpy`, `fpdf2`.
Для визуальной проверки отчёта: `pymupdf`.

# -*- coding: utf-8 -*-
"""
Retail A/B case - воспроизводимый анализ.
Положите файл рядом с CSV и запустите:  python analysis.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

DATA = Path(__file__).parent          # папка с CSV; поменяйте при необходимости
pd.set_option("display.width", 160)

PRE  = ("2025-06-16", "2025-07-13")   # 28 дней до редизайна
POST = ("2025-07-14", "2025-08-10")   # 28 дней после
HEAT = ("2025-07-21", "2025-07-27")
PROMO = ("2025-07-21", "2025-08-03")
PROMO_PRODUCT = "P0001"

rd = lambda n: pd.read_csv(DATA / n, encoding="utf-8-sig")

# ---------------------------------------------------------------- 0. ЗАГРУЗКА
tx    = rd("transactions.csv")
items = rd("transaction_items.csv")
prod  = rd("products.csv")
stores = rd("stores.csv")
weather = rd("weather_daily.csv")

print("=" * 70, "\n0. СХЕМА transactions.csv\n", "=" * 70)
print(tx.dtypes, "\n")
print(tx.head(3).to_string(), "\n")

# автоопределение имён колонок - при расхождении задайте вручную
def pick(df, *cands):
    for c in cands:
        if c in df.columns:
            return c
    raise KeyError(f"не нашёл ни одну из {cands} среди {list(df.columns)}")

TX_ID, TX_STORE, TX_STAT, TX_AMT = "transaction_id", "store_id", "status", "amount_total"
TX_TS = "closed_at"

tx["date"] = pd.to_datetime(tx[TX_TS]).dt.normalize()


# ---------------------------------------------------------- 1. КОНТРОЛЬ КАЧЕСТВА
print("=" * 70, "\n1. КАЧЕСТВО ДАННЫХ\n", "=" * 70)
print("Статусы чеков:", tx[TX_STAT].value_counts().to_dict())
print("Дублей transaction_id:", tx[TX_ID].duplicated().sum())
print("Пропуски:", tx.isna().sum()[lambda s: s > 0].to_dict() or "нет")
print("Диапазон дат:", tx.date.min().date(), "-", tx.date.max().date())
print("Отрицательных/нулевых сумм:", (tx[TX_AMT] <= 0).sum())
print("store_id вне справочника:", set(tx[TX_STORE]) - set(stores.store_id) or "нет")

# ВАЖНО: работаем только с завершёнными чеками. Проверьте, что метка названа так же.
COMPLETED = [s for s in tx[TX_STAT].unique() if str(s).lower() in
             ("completed", "complete", "ok", "success", "done", "closed", "завершен", "завершён")]
print("Считаю завершёнными:", COMPLETED)
tx_ok = tx[tx[TX_STAT].isin(COMPLETED)].copy()
print(f"Отброшено {len(tx) - len(tx_ok)} чеков из {len(tx)}\n")

# сходимость шапки и позиций
it_sum = items.groupby("transaction_id", as_index=False).line_amount.sum()
chk = tx_ok[[TX_ID, TX_AMT]].merge(it_sum, left_on=TX_ID, right_on="transaction_id", how="left")
chk["delta"] = chk[TX_AMT] - chk.line_amount
print("Чеков без позиций:", chk.line_amount.isna().sum())
print("Расхождение шапки и позиций > 0.01:", (chk.delta.abs() > 0.01).sum())
print("Осиротевших позиций:", (~items.transaction_id.isin(tx[TX_ID])).sum())
print("Отрицательное количество:", (items.quantity <= 0).sum())

# покрытие: каждый магазин должен иметь все 56 дней
cover = tx_ok.groupby(TX_STORE).date.nunique()
print("Магазинов с неполным покрытием дат:", (cover < 56).sum(), "\n")

# ------------------------------------------------------ 2. ПАНЕЛЬ МАГАЗИН × ДЕНЬ
daily = (tx_ok.groupby([TX_STORE, "date"])
         .agg(revenue=(TX_AMT, "sum"), checks=(TX_ID, "nunique"))
         .reset_index().rename(columns={TX_STORE: "store_id"}))
daily["avg_check"] = daily.revenue / daily.checks
daily = daily.merge(stores[["store_id", "pair_id", "experiment_group", "store_format"]], on="store_id")
daily["period"] = np.where(daily.date <= PRE[1], "pre", "post")
daily["is_heat"]  = daily.date.between(*pd.to_datetime(HEAT))
daily["is_promo"] = daily.date.between(*pd.to_datetime(PROMO))

# ------------------------------------------- 3. ПАРНЫЙ DiD (главный результат)
def paired_did(panel, metric, log=True):
    """Возвращает средний DiD по 30 парам, 95% ДИ и p-value парного t-теста."""
    p = panel.copy()
    p["y"] = np.log(p[metric]) if log else p[metric]
    m = p.groupby(["pair_id", "experiment_group", "period"]).y.mean().unstack("period")
    m["diff"] = m["post"] - m["pre"]                       # изменение внутри магазина
    d = m["diff"].unstack("experiment_group")
    did = d["test"] - d["control"]                          # 30 значений, по одному на пару
    t, pval = stats.ttest_1samp(did, 0)
    se = did.std(ddof=1) / np.sqrt(len(did))
    crit = stats.t.ppf(0.975, len(did) - 1)
    lo, hi = did.mean() - crit * se, did.mean() + crit * se
    if log:   # переводим лог-разницу в проценты
        return dict(metric=metric, effect_pct=100 * (np.exp(did.mean()) - 1),
                    ci_low=100 * (np.exp(lo) - 1), ci_high=100 * (np.exp(hi) - 1),
                    p=pval, n_pairs=len(did), raw_log=did.mean())
    return dict(metric=metric, effect=did.mean(), ci_low=lo, ci_high=hi, p=pval, n_pairs=len(did))

print("=" * 70, "\n2. ЭФФЕКТ РЕДИЗАЙНА - парный DiD, единица анализа = пара магазинов\n", "=" * 70)
main = pd.DataFrame([paired_did(daily, m) for m in ("revenue", "checks", "avg_check")])
print(main.round(2).to_string(index=False))

print("\nВ рублях (без логарифма), выручка на магазин в день:")
print(pd.DataFrame([paired_did(daily, "revenue", log=False)]).round(1).to_string(index=False))

# --------------------------------------------------------- 4. ДЕКОМПОЗИЦИЯ
print("\n" + "=" * 70, "\n3. ЗА СЧЁТ ЧЕГО: log(выручка) = log(чеки) + log(средний чек)\n", "=" * 70)
r, c, a = (main.set_index("metric").raw_log[k] for k in ("revenue", "checks", "avg_check"))
print(f"  выручка      {100*(np.exp(r)-1):+6.2f}%")
print(f"  = чеки       {100*(np.exp(c)-1):+6.2f}%   ({100*c/r:5.1f}% вклада)")
print(f"  + ср. чек    {100*(np.exp(a)-1):+6.2f}%   ({100*a/r:5.1f}% вклада)")
print(f"  сходимость (должно быть ~0): {r - c - a:.2e}")

# ---------------------------------------------------------- 5. РОБАСТНОСТЬ
print("\n" + "=" * 70, "\n4. УСТОЙЧИВОСТЬ К ЖАРЕ И ПРОМО\n", "=" * 70)
scenarios = {
    "все дни":                daily,
    "без жары 21-27.07":      daily[~daily.is_heat],
    "без промо 21.07-03.08":  daily[~daily.is_promo],
    "без жары и промо":       daily[~daily.is_heat & ~daily.is_promo],
    "только convenience":     daily[daily.store_format == "convenience"],
    "только supermarket":     daily[daily.store_format == "supermarket"],
}
rob = pd.DataFrame([{"сценарий": k, **paired_did(v, "revenue")} for k, v in scenarios.items()])
print(rob[["сценарий", "effect_pct", "ci_low", "ci_high", "p"]].round(2).to_string(index=False))

# плацебо: сдвигаем "запуск" на 4 недели назад внутри пре-периода
plac = daily[daily.period == "pre"].copy()
plac["period"] = np.where(plac.date <= "2025-06-29", "pre", "post")
print("\nПлацебо-тест (ложный запуск 30.06, только пре-период) - должно быть ~0 и незначимо:")
print(pd.DataFrame([paired_did(plac, "revenue")])[["effect_pct", "ci_low", "ci_high", "p"]].round(2).to_string(index=False))

# понедельная динамика - проверка параллельности трендов до запуска
print("\nПонедельная разница test-control по выручке (недели до запуска должны быть плоскими):")
wk = daily.copy()
wk["week"] = ((wk.date - pd.Timestamp(PRE[0])).dt.days // 7) + 1
wtab = (wk.groupby(["week", "experiment_group"]).revenue.mean().unstack()
        .assign(разница_pct=lambda d: 100 * (d.test / d.control - 1)))
wtab["запуск"] = np.where(wtab.index <= 4, "до", "после")
print(wtab.round(2).to_string())

# ------------------------------------------------------------ 6. ЭНЕРГЕТИК
print("\n" + "=" * 70, "\n5. ЭНЕРГЕТИК P0001 ВО ВРЕМЯ ПРОМО\n", "=" * 70)
it = items.merge(tx_ok[[TX_ID, TX_STORE, "date"]], left_on="transaction_id", right_on=TX_ID)
it = it.merge(prod[["product_id", "category_name", "subcategory_name"]], on="product_id")
soft = it[it.category_name == "Прохладительные напитки"].copy()

def window(d):
    if d < pd.Timestamp(PROMO[0]):  return "1_до промо"
    if d <= pd.Timestamp(PROMO[1]): return "2_промо"
    return "3_после промо"
soft["win"] = soft.date.map(window)

# абсолютные продажи энергетика в день на магазин
p1 = soft[soft.product_id == PROMO_PRODUCT]
n_stores, days = stores.store_id.nunique(), soft.groupby("win").date.nunique()
abs_tab = pd.DataFrame({
    "штук_в_день_на_магазин": p1.groupby("win").quantity.sum() / days / n_stores,
    "выручка_в_день_на_магазин": p1.groupby("win").line_amount.sum() / days / n_stores,
})
print("Абсолютные продажи P0001:")
print(abs_tab.round(2).to_string())

# доля в категории
share = (soft.groupby("win").apply(
    lambda g: pd.Series({
        "доля_по_выручке_%": 100 * g.loc[g.product_id == PROMO_PRODUCT, "line_amount"].sum() / g.line_amount.sum(),
        "доля_по_штукам_%":  100 * g.loc[g.product_id == PROMO_PRODUCT, "quantity"].sum() / g.quantity.sum(),
    }), include_groups=False))
print("\nДоля P0001 в категории 'Прохладительные напитки':")
print(share.round(2).to_string())

# каннибализация: что стало с остальными энергетиками
other = soft[(soft.subcategory_name == "Энергетики") & (soft.product_id != PROMO_PRODUCT)]
print("\nОстальные энергетики, выручка в день на магазин (проверка каннибализации):")
print((other.groupby("win").line_amount.sum() / days / n_stores).round(2).to_string())

print("""
ВНИМАНИЕ ПРИ ИНТЕРПРЕТАЦИИ:
  - промо шло во ВСЕХ 60 магазинах одновременно, контрольной группы нет;
  - жара 21-27.07 целиком лежит внутри окна промо 21.07-03.08;
  => эффект промо и эффект жары на напитки статистически неразделимы.
     Корректный вывод - описательный ("во время промо продажи выросли на X%"),
     а не причинный ("промо дало +X%").
""")

# ------------------------------------------------------------- 7. ВЫГРУЗКА
out = Path("outputs"); out.mkdir(exist_ok=True)
main.to_csv(out / "did_main.csv", index=False, encoding="utf-8-sig")
rob.to_csv(out / "did_robustness.csv", index=False, encoding="utf-8-sig")
daily.to_csv(out / "daily_panel.csv", index=False, encoding="utf-8-sig")
print(f"Результаты сохранены в {out.resolve()}")
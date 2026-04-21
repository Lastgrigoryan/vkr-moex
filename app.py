import streamlit as st
import pandas as pd
import numpy as np

from src.moex_iss import fetch_candles
from src.factors import compute_factors_one
from src.selection import score_and_select_top3
from src.export import to_excel_bytes
from src.markowitz import build_returns_matrix, markowitz_min_variance

st.set_page_config(page_title="MOEX Portfolio Builder", layout="wide")
st.title("Формирование диверсифицированного портфеля акций на основе многофакторной модели")

# --- Пресеты стратегий (веса факторов)
PRESETS = {
    "Высокий (High Momentum)": {"w_mom": 0.60, "w_stab": 0.15, "w_dd": 0.15, "w_liq": 0.10},
    "Средний (Balanced)":      {"w_mom": 0.35, "w_stab": 0.30, "w_dd": 0.20, "w_liq": 0.15},
    "Низкий (Low Risk)":       {"w_mom": 0.15, "w_stab": 0.45, "w_dd": 0.30, "w_liq": 0.10},
}

# --- Инициализация весов в session_state (сохраняются при перезапуске страницы)
if "w_mom" not in st.session_state:
    p = PRESETS["Средний (Balanced)"]
    st.session_state["w_mom"] = p["w_mom"]
    st.session_state["w_stab"] = p["w_stab"]
    st.session_state["w_dd"] = p["w_dd"]
    st.session_state["w_liq"] = p["w_liq"]
    st.session_state["preset_name"] = "Средний (Balanced)"

# --- Inputs
colA, colB, colC = st.columns(3)
with colA:
    date_from = st.text_input("Дата начала (YYYY-MM-DD)", "2022-01-01")
with colB:
    date_till = st.text_input("Дата конца (YYYY-MM-DD)", "2026-02-26")
with colC:
    interval = st.selectbox("Интервал свечей (ISS interval)", [24, 60], index=0)

st.subheader("Справочник отраслей")
sectors_file = st.file_uploader("Загрузите файл", type=["csv"])

# --- Кнопки стратегий (пресеты)
st.subheader("Риск")

strategy = st.segmented_control(
    "Выберите степень риска",
    ["Высокий", "Средний", "Низкий"],
    default="Низкий"
)

if strategy == "Высокий":
    p = PRESETS["Высокий (High Momentum)"]
elif strategy == "Средний":
    p = PRESETS["Среднесрочный (Balanced)"]
else:
    p = PRESETS["Низкий (Low Risk)"]

st.session_state["w_mom"] = p["w_mom"]
st.session_state["w_stab"] = p["w_stab"]
st.session_state["w_dd"] = p["w_dd"]
st.session_state["w_liq"] = p["w_liq"]
st.session_state["preset_name"] = strategy

# --- Показ текущих весов (и возможность вручную подправить)
st.subheader("Текущие веса факторов (можно подкорректировать вручную)")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.session_state["w_mom"] = st.number_input(
        "Momentum (динамика роста)",
        min_value=0.0, max_value=1.0, value=float(st.session_state["w_mom"]), step=0.05
    )
with c2:
    st.session_state["w_stab"] = st.number_input(
        "Stability (отрицательная волатильность)",
        min_value=0.0, max_value=1.0, value=float(st.session_state["w_stab"]), step=0.05
    )
with c3:
    st.session_state["w_dd"] = st.number_input(
        "Drawdown (максимальная историческая просадка)",
        min_value=0.0, max_value=1.0, value=float(st.session_state["w_dd"]), step=0.05
    )
with c4:
    st.session_state["w_liq"] = st.number_input(
        "Liquidity (ликвидность)",
        min_value=0.0, max_value=1.0, value=float(st.session_state["w_liq"]), step=0.05
    )

w_mom = float(st.session_state["w_mom"])
w_stab = float(st.session_state["w_stab"])
w_dd = float(st.session_state["w_dd"])
w_liq = float(st.session_state["w_liq"])

sumw = w_mom + w_stab + w_dd + w_liq
st.caption(f"Выбранная стратегия: **{st.session_state.get('preset_name','—')}** | Сумма весов: **{sumw:.2f}**")

# Если сумма не равна 1 — нормализуем (чтобы score был сопоставим)
normalize = st.checkbox("Автоматически нормализовать веса до суммы 1", value=True)
if sumw == 0:
    st.error("Сумма весов = 0. Задайте ненулевые веса.")
    st.stop()

if abs(sumw - 1.0) > 1e-6:
    if normalize:
        w_mom, w_stab, w_dd, w_liq = w_mom / sumw, w_stab / sumw, w_dd / sumw, w_liq / sumw
        st.info(f"Веса нормализованы: Momentum={w_mom:.2f}, Stability={w_stab:.2f}, Drawdown={w_dd:.2f}, Liquidity={w_liq:.2f}")
    else:
        st.warning(f"Сумма весов ≠ 1 (сейчас {sumw:.2f}). Можно включить нормализацию.")

run = st.button("Рассчитать")

if run:
    if sectors_file is None:
        st.error("Нужно загрузить файл с колонками secid")
        st.stop()

    sectors = pd.read_csv(sectors_file, encoding="utf-8-sig", sep=None, engine="python")
    sectors.columns = [c.strip().lower().replace("\ufeff", "") for c in sectors.columns]
    if not {"secid", "sector"}.issubset(sectors.columns):
        st.error(f"Колонки в файле: {sectors.columns.tolist()}. Нужно: secid, sector")
        st.stop()

    sectors["secid"] = sectors["secid"].astype(str).str.upper().str.strip()
    sectors["sector"] = sectors["sector"].astype(str).str.strip()
    sectors = sectors.dropna(subset=["secid", "sector"]).drop_duplicates("secid")

    tickers = list(sectors["secid"])
    st.info(f"Тикеров к обработке: {len(tickers)}")

    all_candles = []
    factors_list = []
    failed = []

    with st.spinner("Анализ свечей и расчет факторов..."):
        for i, secid in enumerate(tickers, 1):
            try:
                dfc = fetch_candles(secid, date_from, date_till, interval=interval)
                if dfc.empty or dfc["close"].nunique() < 20:
                    failed.append((secid, "no/too few candles"))
                    continue
                all_candles.append(dfc)
                factors_list.append(compute_factors_one(dfc))
            except Exception as e:
                failed.append((secid, str(e)))

            if i % 25 == 0:
                st.write(f"Обработано: {i}/{len(tickers)}")

    if not factors_list:
        st.error("Не удалось получить факторы ни по одной акции. Проверьте даты/интернет/тикеры.")
        st.stop()

    candles_df = pd.concat(all_candles, ignore_index=True)
    factors_df = pd.concat(factors_list, ignore_index=True)

    # --- Портфель Марковица
    markowitz_df = pd.DataFrame()
    markowitz_summary = pd.DataFrame()

    try:
        returns_df = build_returns_matrix(candles_df)
        markowitz_df, markowitz_summary = markowitz_min_variance(returns_df)
    except Exception as e:
        st.warning(f"Не удалось рассчитать портфель Марковица: {e}")

    scored, top3 = score_and_select_top3(
        factors=factors_df,
        sectors=sectors,
        w_mom=w_mom, w_stab=w_stab, w_dd=w_dd, w_liq=w_liq
    )

    st.subheader("Портфель по методу Марковица")

    if not markowitz_df.empty:
        st.markdown("### Состав портфеля Марковица")
        st.dataframe(markowitz_df, use_container_width=True)

        st.markdown("### Сводные показатели портфеля Марковица")
        st.dataframe(markowitz_summary, use_container_width=True)
    else:
        st.info("Портфель Марковица не был рассчитан.")

    st.subheader("Результаты")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Scored (все акции)")
        # если у тебя глобальный скоринг — сортируй просто по score
        # если отраслевой — можно по sector/score
        st.dataframe(scored.sort_values(["score"], ascending=False), use_container_width=True)

    with c2:
        st.markdown("### Top-3 по каждой отрасли")
        st.dataframe(top3, use_container_width=True)

    if failed:
        st.markdown("### Не удалось обработать")
        st.dataframe(pd.DataFrame(failed, columns=["secid", "reason"]), use_container_width=True)

    # Добавим лист с весами/стратегией
    weights_df = pd.DataFrame([{
        "preset_name": st.session_state.get("preset_name", ""),
        "w_mom": w_mom,
        "w_stab": w_stab,
        "w_dd": w_dd,
        "w_liq": w_liq,
        "sum": w_mom + w_stab + w_dd + w_liq,
        "date_from": date_from,
        "date_till": date_till,
        "interval": interval,
    }])

    excel_bytes = to_excel_bytes(
        weights=weights_df,
        sectors=sectors,
        candles=candles_df,
        factors=factors_df,
        scored=scored,
        top3=top3,
        markowitz=markowitz_df,
        markowitz_summary=markowitz_summary,
        failed=pd.DataFrame(failed, columns=["secid", "reason"]) if failed else pd.DataFrame(columns=["secid", "reason"]),
    )

    st.download_button(
        "Скачать Excel с расчётами",
        data=excel_bytes,
        file_name="portfolio_selection_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
st.subheader("Проверка цены акции по SECID")

col1, col2, col3 = st.columns(3)

with col1:
    check_secid = st.text_input("SECID акции", value="SBER").upper().strip()

with col2:
    buy_date = st.text_input("Дата покупки (YYYY-MM-DD)", value="2024-01-10")

with col3:
    sell_date = st.text_input("Дата продажи (YYYY-MM-DD)", value="2024-02-10")

check_run = st.button("Показать цену покупки и продажи")

if check_run:
    try:
        df_check = fetch_candles(check_secid, buy_date, sell_date, interval=24)

        if df_check.empty:
            st.error("По данной акции не найдено данных за указанный период.")
        else:
            df_check["date"] = pd.to_datetime(df_check["date"])

            buy_dt = pd.to_datetime(buy_date)
            sell_dt = pd.to_datetime(sell_date)

            # Берем первую доступную цену не раньше даты покупки
            buy_row = df_check[df_check["date"] >= buy_dt].sort_values("date").head(1)

            # Берем последнюю доступную цену не позже даты продажи
            sell_row = df_check[df_check["date"] <= sell_dt].sort_values("date").tail(1)

            if buy_row.empty or sell_row.empty:
                st.error("Не удалось найти цену покупки или продажи на выбранные даты.")
            else:
                buy_price = float(buy_row["close"].iloc[0])
                buy_real_date = buy_row["date"].dt.strftime("%Y-%m-%d").iloc[0]

                sell_price = float(sell_row["close"].iloc[0])
                sell_real_date = sell_row["date"].dt.strftime("%Y-%m-%d").iloc[0]

                ret = (sell_price / buy_price - 1.0) * 100

                result_df = pd.DataFrame([{
                    "secid": check_secid,
                    "buy_date_requested": buy_date,
                    "buy_date_actual": buy_real_date,
                    "buy_price": buy_price,
                    "sell_date_requested": sell_date,
                    "sell_date_actual": sell_real_date,
                    "sell_price": sell_price,
                    "return_%": ret
                }])

                st.dataframe(result_df, use_container_width=True)

    except Exception as e:
        st.error(f"Ошибка при получении данных: {e}")
# =========================
# Проверка доходности портфеля из файла
# =========================

st.subheader("Проверка доходности портфеля из файла")

test_file = st.file_uploader(
    "Загрузите CSV/XLSX с рекомендациями (top3, markowitz и т.д.)",
    type=["csv", "xlsx"],
    key="portfolio_test_file"
)

buy_col, sell_col = st.columns(2)
with buy_col:
    test_buy_date = st.text_input("Дата покупки (YYYY-MM-DD)", value="2025-03-23", key="test_buy_date")
with sell_col:
    test_sell_date = st.text_input("Дата продажи (YYYY-MM-DD)", value="2026-03-22", key="test_sell_date")

sheet_name = None

if test_file is not None and test_file.name.lower().endswith(".xlsx"):
    try:
        xls = pd.ExcelFile(test_file)
        sheet_name = st.selectbox("Выберите лист Excel", xls.sheet_names, key="test_sheet_name")
    except Exception as e:
        st.error(f"Не удалось открыть Excel-файл: {e}")

run_test_file = st.button("Рассчитать доходность по всем акциям из файла")

if run_test_file:
    if test_file is None:
        st.error("Сначала загрузите CSV или XLSX файл.")
        st.stop()

    # ---------- 1. Читаем файл ----------
    try:
        if test_file.name.lower().endswith(".csv"):
            portfolio_df = pd.read_csv(test_file, encoding="utf-8-sig", sep=None, engine="python")
        else:
            portfolio_df = pd.read_excel(test_file, sheet_name=sheet_name)
    except Exception as e:
        st.error(f"Ошибка чтения файла: {e}")
        st.stop()

    portfolio_df.columns = [str(c).strip().lower().replace("\ufeff", "") for c in portfolio_df.columns]

    if "secid" not in portfolio_df.columns:
        st.error(f"В файле нет колонки 'secid'. Найденные колонки: {portfolio_df.columns.tolist()}")
        st.stop()

    portfolio_df["secid"] = portfolio_df["secid"].astype(str).str.upper().str.strip()
    portfolio_df = portfolio_df.dropna(subset=["secid"]).drop_duplicates(subset=["secid"]).copy()

    # ищем колонку веса
    weight_col = None
    for c in ["weight_markowitz", "weight"]:
        if c in portfolio_df.columns:
            weight_col = c
            break

    tickers_to_test = portfolio_df["secid"].tolist()

    st.info(f"К тестированию загружено акций: {len(tickers_to_test)}")

    # ---------- 2. Для каждой акции получаем цены покупки/продажи ----------
    results = []
    failed_test = []

    with st.spinner("Получаю исторические цены для всех акций из файла..."):
        for i, secid in enumerate(tickers_to_test, 1):
            try:
                df_test = fetch_candles(secid, test_buy_date, test_sell_date, interval=24)

                if df_test.empty:
                    failed_test.append((secid, "Нет свечей за выбранный период"))
                    continue

                df_test["date"] = pd.to_datetime(df_test["date"])

                buy_dt = pd.to_datetime(test_buy_date)
                sell_dt = pd.to_datetime(test_sell_date)

                # первая доступная свеча на или после даты покупки
                buy_row = df_test[df_test["date"] >= buy_dt].sort_values("date").head(1)

                # последняя доступная свеча на или до даты продажи
                sell_row = df_test[df_test["date"] <= sell_dt].sort_values("date").tail(1)

                if buy_row.empty or sell_row.empty:
                    failed_test.append((secid, "Не найдена цена покупки или продажи"))
                    continue

                buy_price = float(buy_row["close"].iloc[0])
                buy_real_date = buy_row["date"].dt.strftime("%Y-%m-%d").iloc[0]

                sell_price = float(sell_row["close"].iloc[0])
                sell_real_date = sell_row["date"].dt.strftime("%Y-%m-%d").iloc[0]

                ret = (sell_price / buy_price - 1.0) * 100.0

                row = {
                    "secid": secid,
                    "buy_date_requested": test_buy_date,
                    "buy_date_actual": buy_real_date,
                    "buy_price": buy_price,
                    "sell_date_requested": test_sell_date,
                    "sell_date_actual": sell_real_date,
                    "sell_price": sell_price,
                    "return_%": ret
                }

                # переносим все полезные поля из исходного файла
                src_row = portfolio_df[portfolio_df["secid"] == secid].iloc[0].to_dict()
                row.update(src_row)

                results.append(row)

            except Exception as e:
                failed_test.append((secid, str(e)))

            if i % 20 == 0:
                st.write(f"Проверено {i}/{len(tickers_to_test)}")

    if not results:
        st.error("Не удалось рассчитать доходность ни по одной акции.")
        if failed_test:
            st.dataframe(pd.DataFrame(failed_test, columns=["secid", "reason"]), use_container_width=True)
        st.stop()

    result_df = pd.DataFrame(results)

    # ---------- 3. Считаем агрегированные показатели ----------
    mean_return = result_df["return_%"].mean()
    median_return = result_df["return_%"].median()
    hit_rate = (result_df["return_%"] > 0).mean() * 100.0
    max_ret = result_df["return_%"].max()
    min_ret = result_df["return_%"].min()

    summary_rows = [{
        "n_assets": len(result_df),
        "mean_return_%": mean_return,
        "median_return_%": median_return,
        "hit_rate_%": hit_rate,
        "max_return_%": max_ret,
        "min_return_%": min_ret
    }]

    # Если есть веса — считаем доходность портфеля
    if weight_col is not None:
        result_df[weight_col] = pd.to_numeric(result_df[weight_col], errors="coerce").fillna(0.0)

        # нормализуем веса
        wsum = result_df[weight_col].sum()
        if wsum > 0:
            result_df["weight_normalized"] = result_df[weight_col] / wsum
            portfolio_return = (result_df["weight_normalized"] * result_df["return_%"]).sum()
        else:
            result_df["weight_normalized"] = 0.0
            portfolio_return = np.nan

        summary_rows[0]["weighted_portfolio_return_%"] = portfolio_return

    summary_df = pd.DataFrame(summary_rows)

    # ---------- 4. Вывод ----------
    st.markdown("### Результаты тестирования по всем акциям")
    st.dataframe(result_df.sort_values("return_%", ascending=False).reset_index(drop=True), use_container_width=True)

    st.markdown("### Сводные показатели")
    st.dataframe(summary_df, use_container_width=True)

    if failed_test:
        st.markdown("### Не удалось обработать")
        st.dataframe(pd.DataFrame(failed_test, columns=["secid", "reason"]), use_container_width=True)

    # ---------- 5. Скачивание результата ----------
    test_excel = to_excel_bytes(
        portfolio_input=portfolio_df,
        portfolio_test_result=result_df,
        portfolio_test_summary=summary_df,
        portfolio_test_failed=pd.DataFrame(failed_test, columns=["secid", "reason"]) if failed_test else pd.DataFrame(columns=["secid", "reason"])
    )

    st.download_button(
        "Скачать результаты тестирования Excel",
        data=test_excel,
        file_name="portfolio_test_result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
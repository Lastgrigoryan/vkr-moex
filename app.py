import streamlit as st
import pandas as pd
import numpy as np

from src.moex_iss import fetch_candles
from src.factors import compute_factors_one
from src.selection import score_and_select_top3
from src.export import to_excel_bytes

st.set_page_config(page_title="MOEX Portfolio Builder", layout="wide")
st.title("Система формирования диверсифицированного портфеля (MOEX акции)")

# --- Пресеты стратегий (веса факторов)
PRESETS = {
    "Краткосрочный (High Momentum)": {"w_mom": 0.60, "w_stab": 0.15, "w_dd": 0.15, "w_liq": 0.10},
    "Среднесрочный (Balanced)":      {"w_mom": 0.35, "w_stab": 0.30, "w_dd": 0.20, "w_liq": 0.15},
    "Долгосрочный (Low Risk)":       {"w_mom": 0.15, "w_stab": 0.45, "w_dd": 0.30, "w_liq": 0.10},
}

# --- Инициализация весов в session_state (сохраняются при перезапуске страницы)
if "w_mom" not in st.session_state:
    p = PRESETS["Среднесрочный (Balanced)"]
    st.session_state["w_mom"] = p["w_mom"]
    st.session_state["w_stab"] = p["w_stab"]
    st.session_state["w_dd"] = p["w_dd"]
    st.session_state["w_liq"] = p["w_liq"]
    st.session_state["preset_name"] = "Среднесрочный (Balanced)"

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
st.subheader("Стратегия")

strategy = st.segmented_control(
    "Выберите стратегию",
    ["Краткосрочный", "Среднесрочный", "Долгосрочный"],
    default="Среднесрочный"
)

if strategy == "Краткосрочный":
    p = PRESETS["Краткосрочный (High Momentum)"]
elif strategy == "Среднесрочный":
    p = PRESETS["Среднесрочный (Balanced)"]
else:
    p = PRESETS["Долгосрочный (Low Risk)"]

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
        "Stability (−vol) (обратная волатильность)",
        min_value=0.0, max_value=1.0, value=float(st.session_state["w_stab"]), step=0.05
    )
with c3:
    st.session_state["w_dd"] = st.number_input(
        "Drawdown (−MDD) (максимальная историческая просадка)",
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
        st.error("Нужно загрузить файл с колонками secid, sector (например sectors_final.csv)")
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

    with st.spinner("Скачиваю candles (MOEX ISS) и считаю факторы..."):
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

    scored, top3 = score_and_select_top3(
        factors=factors_df,
        sectors=sectors,
        w_mom=w_mom, w_stab=w_stab, w_dd=w_dd, w_liq=w_liq
    )

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
        failed=pd.DataFrame(failed, columns=["secid", "reason"]) if failed else pd.DataFrame(columns=["secid", "reason"]),
    )

    st.download_button(
        "Скачать Excel с расчётами",
        data=excel_bytes,
        file_name="portfolio_selection_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
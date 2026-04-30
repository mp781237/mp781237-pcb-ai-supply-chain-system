import math
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="PCB 供應鏈 AI 系統", layout="wide")

st.title("📊 PCB 供應鏈 AI 學習與分析系統 v2.4")
st.caption("學習產業鏈位置、可轉債資訊、族群相對強弱、動能象限與輪動策略回測。僅供研究與學習，非投資建議。")

STOCKS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "載板": {
        "欣興": {"code": "3037", "cb": True, "role": "ABF / BT 載板，AI GPU / ASIC 需求相關", "power": "Tier 1：AI 需求核心"},
        "南電": {"code": "8046", "cb": True, "role": "ABF 載板，半導體封裝材料鏈", "power": "Tier 1：AI 需求核心"},
        "景碩": {"code": "3189", "cb": True, "role": "載板，封裝基板族群", "power": "Tier 1：AI 需求核心"},
    },
    "CCL": {
        "台光電": {"code": "2383", "cb": False, "role": "高階 CCL，高頻高速材料", "power": "Tier 1：材料定價權"},
        "台燿": {"code": "6274", "cb": True, "role": "高階 CCL，AI 伺服器材料", "power": "Tier 1：材料定價權"},
        "聯茂": {"code": "6213", "cb": True, "role": "CCL，銅箔基板材料", "power": "Tier 2：材料循環"},
    },
    "PCB": {
        "金像電": {"code": "2368", "cb": True, "role": "伺服器 / AI PCB 硬板", "power": "Tier 3：接單與補漲"},
        "華通": {"code": "2313", "cb": True, "role": "PCB 硬板，消費電子與通訊板", "power": "Tier 3：接單與補漲"},
        "健鼎": {"code": "3044", "cb": False, "role": "PCB 硬板，規模型製造", "power": "Tier 3：接單與補漲"},
    },
    "上游材料": {
        "金居": {"code": "8358", "cb": True, "role": "銅箔，導電材料", "power": "Tier 2：漲價循環"},
        "富喬": {"code": "1815", "cb": True, "role": "玻纖布，PCB 結構材料", "power": "Tier 2：供需循環"},
        "台玻": {"code": "1802", "cb": False, "role": "玻纖與玻璃材料", "power": "Tier 2：材料循環"},
        "達邁": {"code": "3645", "cb": True, "role": "PI 材料，軟板上游", "power": "Tier 2：軟板材料"},
    },
    "設備": {
        "志聖": {"code": "2467", "cb": True, "role": "PCB / 載板製程設備，吃 Capex 循環", "power": "Capex 前置訊號"},
        "由田": {"code": "3455", "cb": True, "role": "檢測設備，PCB / 半導體相關", "power": "Capex 前置訊號"},
        "群翊": {"code": "6664", "cb": True, "role": "製程設備，PCB / 載板相關", "power": "Capex 前置訊號"},
    },
}

CORE_GROUPS = ["載板", "CCL", "PCB"]
QUESTION_BANK = [
    {"q": "台光電屬於哪一層？", "options": ["上游材料", "CCL", "PCB", "載板"], "answer": "CCL", "explain": "台光電主要是 CCL 銅箔基板，高頻高速材料是 AI 伺服器題材中的重要位置。"},
    {"q": "志聖（2467）在 PCB 供應鏈中比較像哪一種角色？", "options": ["銅箔", "CCL", "設備", "載板"], "answer": "設備", "explain": "志聖屬於設備廠，主要吃擴產、製程升級與 Capex 循環，不是材料本身。"},
    {"q": "如果載板 > CCL > PCB，通常代表什麼？", "options": ["標準輪動", "假輪動", "空頭", "資料錯誤"], "answer": "標準輪動", "explain": "這代表權力端或需求端先反應，再傳導到 CCL 與 PCB，屬於比較健康的輪動結構。"},
    {"q": "CB 是什麼？", "options": ["公司債", "可轉債", "普通股", "ETF"], "answer": "可轉債", "explain": "CB 是 Convertible Bond，可轉換公司債。對高波動股票而言，可能形成轉債套利或偏防守的參與方式。"},
]


def ticker_candidates(code: str) -> List[str]:
    return [f"{code}.TW", f"{code}.TWO"]


@st.cache_data(ttl=60 * 30, show_spinner=False)
def download_price_by_ticker(ticker: str, period: str = "1y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0)
        price_key = "Adj Close" if "Adj Close" in level0 else "Close"
        if price_key not in level0:
            return pd.DataFrame()
        close = df.xs(price_key, axis=1, level=0).iloc[:, 0]
    else:
        price_key = "Adj Close" if "Adj Close" in df.columns else "Close"
        if price_key not in df.columns:
            return pd.DataFrame()
        close = df[price_key]
    return pd.DataFrame({"Close": pd.to_numeric(close, errors="coerce")}).dropna()


@st.cache_data(ttl=60 * 30, show_spinner=False)
def download_price(code: str, period: str = "1y") -> Tuple[pd.DataFrame, str]:
    for ticker in ticker_candidates(code):
        df = download_price_by_ticker(ticker, period=period)
        if not df.empty:
            return df, ticker
    return pd.DataFrame(), "抓取失敗"


def safe_return(code: str, lookback: int) -> Tuple[float, str]:
    df, used_ticker = download_price(code, period="1y")
    if len(df) <= lookback:
        return float("nan"), used_ticker
    latest = float(df["Close"].iloc[-1])
    past = float(df["Close"].iloc[-lookback])
    if past == 0:
        return float("nan"), used_ticker
    return float((latest / past - 1) * 100), used_ticker


def classify_momentum(row: pd.Series) -> str:
    r20 = row["20日漲跌幅%"]
    r60 = row["60日漲跌幅%"]
    if pd.isna(r20) or pd.isna(r60):
        return "資料不足"
    if r20 >= 0 and r60 < 0:
        return "🔥 初動/轉強"
    if r20 >= 0 and r60 >= 0:
        return "🚀 主升/強勢"
    if r20 < 0 and r60 >= 0:
        return "🟡 強勢回檔"
    return "❄️ 弱勢"


def build_stock_table() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for category, group in STOCKS.items():
        for name, info in group.items():
            code = info["code"]
            ret20, used_ticker = safe_return(code, 20)
            ret60, used_ticker_60 = safe_return(code, 60)
            rows.append({
                "類別": category, "公司": name, "股號": code,
                "實際Ticker": used_ticker if used_ticker != "抓取失敗" else used_ticker_60,
                "20日漲跌幅%": ret20, "60日漲跌幅%": ret60,
                "是否有CB": "有" if info["cb"] else "無",
                "產業定位": info["role"], "權力分級": info["power"],
            })
    table = pd.DataFrame(rows)
    table["動能象限"] = table.apply(classify_momentum, axis=1)
    return table


def group_return(table: pd.DataFrame, category: str, column: str = "20日漲跌幅%") -> float:
    values = pd.to_numeric(table.loc[table["類別"] == category, column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else float("nan")


def fmt_pct(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "資料不足"
    return f"{value:.2f}%"


def determine_state(leader: float, ccl: float, pcb: float) -> Tuple[str, str, str]:
    if any(math.isnan(x) for x in [leader, ccl, pcb]):
        return "⚪ 資料不足", "部分股價資料抓取失敗，先不要判斷。", "info"
    if leader > ccl > pcb:
        return "🔥 AI 主升段 / 標準輪動", "載板 > CCL > PCB，代表權力端先動、PCB 仍可能補漲。", "success"
    if pcb > leader:
        return "⚠️ 假輪動或末端補漲", "PCB 強於載板，可能是短線補漲或題材末端，不宜只看表面強勢。", "warning"
    if leader > 0 and ccl > 0 and pcb > 0:
        return "🟡 同步上漲", "三者都偏強，但輪動順序不明顯，策略訊號較弱。", "warning"
    return "🟡 混沌 / 觀望", "目前沒有清楚的載板 → CCL → PCB 輪動結構。", "warning"


def build_group_index(category: str, period: str = "1y") -> pd.DataFrame:
    series_list = []
    for name, info in STOCKS[category].items():
        df, _ = download_price(info["code"], period=period)
        if df.empty:
            continue
        s = df["Close"].copy()
        s.name = name
        series_list.append(s)
    if not series_list:
        return pd.DataFrame()
    prices = pd.concat(series_list, axis=1).dropna(how="all").ffill().dropna()
    indexed = prices / prices.iloc[0] * 100
    return pd.DataFrame({category: indexed.mean(axis=1)})


def build_all_group_indices(period: str = "1y") -> pd.DataFrame:
    frames = [build_group_index(group, period) for group in CORE_GROUPS]
    frames = [x for x in frames if not x.empty]
    return pd.concat(frames, axis=1).dropna() if frames else pd.DataFrame()


def add_quadrant_shapes(fig: go.Figure, df: pd.DataFrame) -> go.Figure:
    x_values = pd.to_numeric(df["60日漲跌幅%"], errors="coerce").dropna()
    y_values = pd.to_numeric(df["20日漲跌幅%"], errors="coerce").dropna()
    if x_values.empty or y_values.empty:
        return fig
    x_min, x_max = min(float(x_values.min()), -1.0), max(float(x_values.max()), 1.0)
    y_min, y_max = min(float(y_values.min()), -1.0), max(float(y_values.max()), 1.0)
    pad_x, pad_y = max((x_max - x_min) * 0.12, 2.0), max((y_max - y_min) * 0.12, 2.0)
    x_min, x_max, y_min, y_max = x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_xaxes(range=[x_min, x_max], zeroline=False)
    fig.update_yaxes(range=[y_min, y_max], zeroline=False)
    fig.add_annotation(x=x_max, y=y_max, text="🚀 主升/強勢", showarrow=False, xanchor="right", yanchor="top")
    fig.add_annotation(x=x_min, y=y_max, text="🔥 初動/轉強", showarrow=False, xanchor="left", yanchor="top")
    fig.add_annotation(x=x_max, y=y_min, text="🟡 強勢回檔", showarrow=False, xanchor="right", yanchor="bottom")
    fig.add_annotation(x=x_min, y=y_min, text="❄️ 弱勢", showarrow=False, xanchor="left", yanchor="bottom")
    return fig


def init_learning_state() -> None:
    if "answered" not in st.session_state:
        st.session_state.answered = {}


def record_answer(question: str, correct: bool) -> None:
    st.session_state.answered[question] = correct


def learning_score() -> Tuple[int, int]:
    total = len(st.session_state.answered)
    correct = sum(1 for v in st.session_state.answered.values() if v)
    return correct, total


def build_backtest_dataset(period: str) -> pd.DataFrame:
    frames = []
    for group in CORE_GROUPS:
        idx = build_group_index(group, period=period)
        if idx.empty:
            return pd.DataFrame()
        frames.append(idx)
    data = pd.concat(frames, axis=1).dropna()
    if data.empty:
        return data
    data["Leader20"] = data["載板"] / data["載板"].shift(20) - 1
    data["Core20"] = data["CCL"] / data["CCL"].shift(20) - 1
    data["PCB20"] = data["PCB"] / data["PCB"].shift(20) - 1
    data["PCB60"] = data["PCB"] / data["PCB"].shift(60) - 1
    data["PCB_ret"] = data["PCB"].pct_change()
    return data.dropna()


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    dd = equity / equity.cummax() - 1
    return float(dd.min())


def cagr(equity: pd.Series) -> float:
    if equity.empty or len(equity) < 2:
        return float("nan")
    days = max((equity.index[-1] - equity.index[0]).days, 1)
    return float((equity.iloc[-1] / equity.iloc[0]) ** (365 / days) - 1)


def run_rotation_backtest(period: str, max_hold: int, exposure: float) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    data = build_backtest_dataset(period)
    if data.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    equity = 1.0
    benchmark = 1.0
    in_pos = False
    entry_date = None
    entry_equity = None
    holding_days = 0
    trades = []
    records = []

    for i in range(1, len(data)):
        date = data.index[i]
        prev = data.iloc[i - 1]
        today_ret = float(data["PCB_ret"].iloc[i])
        bench_ret = today_ret
        signal = (prev["Leader20"] > prev["Core20"] > prev["PCB20"]) and (prev["PCB20"] > 0) and (prev["PCB60"] < 0)

        if in_pos:
            equity *= 1 + exposure * today_ret
            holding_days += 1
            exit_reason = None
            if prev["PCB20"] < 0:
                exit_reason = "PCB20轉弱"
            elif holding_days >= max_hold:
                exit_reason = "持有天數到期"
            if exit_reason:
                trades.append({
                    "進場日": entry_date,
                    "出場日": date,
                    "持有天數": holding_days,
                    "出場原因": exit_reason,
                    "交易報酬%": (equity / entry_equity - 1) * 100,
                })
                in_pos = False
                entry_date = None
                entry_equity = None
                holding_days = 0
        else:
            if signal:
                in_pos = True
                entry_date = date
                entry_equity = equity
                holding_days = 0

        benchmark *= 1 + bench_ret
        records.append({"Date": date, "策略淨值": equity, "PCB買進持有": benchmark, "是否持倉": in_pos})

    curve = pd.DataFrame(records).set_index("Date")
    trade_log = pd.DataFrame(trades)
    metrics = {
        "總報酬%": (curve["策略淨值"].iloc[-1] - 1) * 100 if not curve.empty else float("nan"),
        "CAGR%": cagr(curve["策略淨值"]) * 100 if not curve.empty else float("nan"),
        "MDD%": max_drawdown(curve["策略淨值"]) * 100 if not curve.empty else float("nan"),
        "交易次數": float(len(trade_log)),
        "勝率%": (trade_log["交易報酬%"].gt(0).mean() * 100) if not trade_log.empty else float("nan"),
        "平均單筆%": trade_log["交易報酬%"].mean() if not trade_log.empty else float("nan"),
        "PCB買進持有報酬%": (curve["PCB買進持有"].iloc[-1] - 1) * 100 if not curve.empty else float("nan"),
    }
    return curve, trade_log, metrics


with st.spinner("正在抓取台股資料，第一次載入可能較慢……"):
    table = build_stock_table()

leader_r = group_return(table, "載板")
ccl_r = group_return(table, "CCL")
pcb_r = group_return(table, "PCB")
leader_60 = group_return(table, "載板", "60日漲跌幅%")
ccl_60 = group_return(table, "CCL", "60日漲跌幅%")
pcb_60 = group_return(table, "PCB", "60日漲跌幅%")
state, explanation, level = determine_state(leader_r, ccl_r, pcb_r)

init_learning_state()

tab_dashboard, tab_chain, tab_learning, tab_strategy, tab_backtest = st.tabs(["📈 儀表板", "🧭 產業鏈", "🎓 學習", "🎯 策略", "📊 回測"])

with tab_dashboard:
    st.header("市場狀態")
    if level == "success": st.success(f"{state}\n\n{explanation}")
    elif level == "info": st.info(f"{state}\n\n{explanation}")
    else: st.warning(f"{state}\n\n{explanation}")

    st.subheader("核心族群 20日 / 60日 強弱")
    col1, col2, col3 = st.columns(3)
    col1.metric("載板 Leader", fmt_pct(leader_r), delta=f"60日 {fmt_pct(leader_60)}")
    col2.metric("CCL Core", fmt_pct(ccl_r), delta=f"60日 {fmt_pct(ccl_60)}")
    col3.metric("PCB Follower", fmt_pct(pcb_r), delta=f"60日 {fmt_pct(pcb_60)}")

    st.subheader("動能象限圖（X=60日中期趨勢，Y=20日短期動能）")
    quadrant_df = table.dropna(subset=["20日漲跌幅%", "60日漲跌幅%"]).copy()
    if quadrant_df.empty:
        st.info("目前無法取得足夠資料繪製象限圖。")
    else:
        fig_q = px.scatter(quadrant_df, x="60日漲跌幅%", y="20日漲跌幅%", color="類別", text="公司", hover_data=["股號", "實際Ticker", "是否有CB", "動能象限", "權力分級"], title="20日 vs 60日 動能象限")
        fig_q.update_traces(textposition="top center", marker=dict(size=12, opacity=0.85))
        fig_q.update_layout(xaxis_title="60日漲跌幅（中期趨勢）", yaxis_title="20日漲跌幅（短期動能）", height=620)
        st.plotly_chart(add_quadrant_shapes(fig_q, quadrant_df), use_container_width=True)

    st.subheader("核心族群走勢（等權指數，起點=100）")
    group_indices = build_all_group_indices(period="1y")
    if not group_indices.empty:
        long_df = group_indices.reset_index().melt(id_vars="Date", var_name="族群", value_name="指數")
        st.plotly_chart(px.line(long_df, x="Date", y="指數", color="族群", title="載板 / CCL / PCB 族群相對走勢"), use_container_width=True)

    st.subheader("20日強弱排名（含60日中期趨勢）")
    rank_display = table.dropna(subset=["20日漲跌幅%"]).sort_values("20日漲跌幅%", ascending=False)[["類別", "公司", "股號", "實際Ticker", "20日漲跌幅%", "60日漲跌幅%", "動能象限", "是否有CB", "權力分級"]].copy()
    rank_display["20日漲跌幅%"] = pd.to_numeric(rank_display["20日漲跌幅%"], errors="coerce").round(2)
    rank_display["60日漲跌幅%"] = pd.to_numeric(rank_display["60日漲跌幅%"], errors="coerce").round(2)
    st.dataframe(rank_display, use_container_width=True, hide_index=True)

with tab_chain:
    st.header("供應鏈結構")
    st.markdown("**上游材料**（銅箔 / 玻纖 / PI） → **CCL 銅箔基板** → **PCB 硬板 / 軟板 / 載板**  \n\n旁支：**設備廠**不在材料鏈內，而是吃擴產與製程升級的 Capex 循環。")
    display_table = table.copy()
    display_table["20日漲跌幅%"] = pd.to_numeric(display_table["20日漲跌幅%"], errors="coerce").round(2)
    display_table["60日漲跌幅%"] = pd.to_numeric(display_table["60日漲跌幅%"], errors="coerce").round(2)
    selected_category = st.selectbox("篩選類別", ["全部"] + list(STOCKS.keys()))
    if selected_category != "全部": display_table = display_table[display_table["類別"] == selected_category]
    st.dataframe(display_table[["類別", "公司", "股號", "實際Ticker", "20日漲跌幅%", "60日漲跌幅%", "動能象限", "是否有CB", "產業定位", "權力分級"]], use_container_width=True, hide_index=True)
    failed = display_table[display_table["實際Ticker"] == "抓取失敗"]
    if not failed.empty: st.warning("以下股票 yfinance 仍抓取失敗，可改用其他資料源補強：" + "、".join(failed["公司"].tolist()))

with tab_learning:
    st.header("互動學習")
    correct, total = learning_score()
    st.metric("本次答題紀錄", f"{correct}/{total}")
    selected = st.selectbox("選擇一題練習", [item["q"] for item in QUESTION_BANK])
    item = next(x for x in QUESTION_BANK if x["q"] == selected)
    answer = st.radio(item["q"], item["options"], index=None, key=f"radio_{selected}")
    if answer:
        is_correct = answer == item["answer"]
        record_answer(item["q"], is_correct)
        st.success("✔ 正確") if is_correct else st.error(f"❌ 這題答案是：{item['answer']}")
        st.write(item["explain"])
    if st.button("清除本次答題紀錄"):
        st.session_state.answered = {}; st.rerun()

with tab_strategy:
    st.header("策略判讀")
    st.markdown("""
    v2.4 的目標不是自動交易，而是幫你建立判斷節奏：
    1. 先看權力端：載板是否強？  
    2. 再看材料端：CCL 是否跟上？  
    3. 最後看製造端：PCB 是否落後且仍有補漲可能？  
    4. 用60日確認中期趨勢，並用象限圖分辨初動、主升、回檔與弱勢。
    """)
    st.write(f"狀態：{state}")
    st.write(explanation)

with tab_backtest:
    st.header("輪動策略回測 v1")
    st.markdown("""
    **規則**：前一日滿足「載板20日 > CCL20日 > PCB20日，且 PCB20日 > 0、PCB60日 < 0」時，隔日進場 PCB 等權指數。  
    **出場**：PCB20日轉弱，或持有滿指定天數。這是研究雛形，尚未納入交易成本、滑價與股息稅。
    """)
    c1, c2, c3 = st.columns(3)
    period = c1.selectbox("回測資料期間", ["2y", "5y", "10y"], index=1)
    max_hold = c2.slider("最多持有天數", 5, 60, 20, 5)
    exposure = c3.slider("單次曝險比例", 0.1, 1.0, 0.5, 0.1)

    if st.button("執行回測"):
        with st.spinner("正在回測，資料期間越長越慢……"):
            curve, trade_log, metrics = run_rotation_backtest(period, max_hold, exposure)
        if curve.empty:
            st.error("資料不足，無法完成回測。")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("總報酬", fmt_pct(metrics.get("總報酬%", float("nan"))))
            m2.metric("CAGR", fmt_pct(metrics.get("CAGR%", float("nan"))))
            m3.metric("MDD", fmt_pct(metrics.get("MDD%", float("nan"))))
            m4.metric("交易次數", int(metrics.get("交易次數", 0)))
            m5, m6, m7 = st.columns(3)
            m5.metric("勝率", fmt_pct(metrics.get("勝率%", float("nan"))))
            m6.metric("平均單筆", fmt_pct(metrics.get("平均單筆%", float("nan"))))
            m7.metric("PCB買進持有", fmt_pct(metrics.get("PCB買進持有報酬%", float("nan"))))

            plot_df = curve.reset_index().melt(id_vars="Date", value_vars=["策略淨值", "PCB買進持有"], var_name="項目", value_name="淨值")
            st.plotly_chart(px.line(plot_df, x="Date", y="淨值", color="項目", title="策略淨值 vs PCB買進持有"), use_container_width=True)
            st.subheader("交易紀錄")
            if trade_log.empty:
                st.info("此參數下沒有完成交易。")
            else:
                show_log = trade_log.copy()
                show_log["交易報酬%"] = pd.to_numeric(show_log["交易報酬%"], errors="coerce").round(2)
                st.dataframe(show_log, use_container_width=True, hide_index=True)
                st.download_button("下載交易紀錄 CSV", show_log.to_csv(index=False).encode("utf-8-sig"), "trade_log.csv", "text/csv")

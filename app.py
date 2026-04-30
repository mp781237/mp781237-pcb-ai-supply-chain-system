import math
from typing import Dict, Any, List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="PCB 供應鏈 AI 系統", layout="wide")

st.title("📊 PCB 供應鏈 AI 學習與分析系統 v2")
st.caption("學習產業鏈位置、可轉債資訊、族群相對強弱與輪動策略。僅供研究與學習，非投資建議。")

STOCKS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "載板": {
        "欣興": {"ticker": "3037.TW", "cb": True, "role": "ABF / BT 載板，AI GPU / ASIC 需求相關", "power": "Tier 1：AI 需求核心"},
        "南電": {"ticker": "8046.TW", "cb": True, "role": "ABF 載板，半導體封裝材料鏈", "power": "Tier 1：AI 需求核心"},
        "景碩": {"ticker": "3189.TW", "cb": True, "role": "載板，封裝基板族群", "power": "Tier 1：AI 需求核心"},
    },
    "CCL": {
        "台光電": {"ticker": "2383.TW", "cb": False, "role": "高階 CCL，高頻高速材料", "power": "Tier 1：材料定價權"},
        "台燿": {"ticker": "6274.TW", "cb": True, "role": "高階 CCL，AI 伺服器材料", "power": "Tier 1：材料定價權"},
        "聯茂": {"ticker": "6213.TW", "cb": True, "role": "CCL，銅箔基板材料", "power": "Tier 2：材料循環"},
    },
    "PCB": {
        "金像電": {"ticker": "2368.TW", "cb": True, "role": "伺服器 / AI PCB 硬板", "power": "Tier 3：接單與補漲"},
        "華通": {"ticker": "2313.TW", "cb": True, "role": "PCB 硬板，消費電子與通訊板", "power": "Tier 3：接單與補漲"},
        "健鼎": {"ticker": "3044.TW", "cb": False, "role": "PCB 硬板，規模型製造", "power": "Tier 3：接單與補漲"},
    },
    "上游材料": {
        "金居": {"ticker": "8358.TW", "cb": True, "role": "銅箔，導電材料", "power": "Tier 2：漲價循環"},
        "富喬": {"ticker": "1815.TW", "cb": True, "role": "玻纖布，PCB 結構材料", "power": "Tier 2：供需循環"},
        "台玻": {"ticker": "1802.TW", "cb": False, "role": "玻纖與玻璃材料", "power": "Tier 2：材料循環"},
        "達邁": {"ticker": "3645.TW", "cb": True, "role": "PI 材料，軟板上游", "power": "Tier 2：軟板材料"},
    },
    "設備": {
        "志聖": {"ticker": "2467.TW", "cb": True, "role": "PCB / 載板製程設備，吃 Capex 循環", "power": "Capex 前置訊號"},
        "由田": {"ticker": "3455.TW", "cb": True, "role": "檢測設備，PCB / 半導體相關", "power": "Capex 前置訊號"},
        "群翊": {"ticker": "6664.TW", "cb": True, "role": "製程設備，PCB / 載板相關", "power": "Capex 前置訊號"},
    },
}

CORE_GROUPS = ["載板", "CCL", "PCB"]
LOOKBACK_DAYS = 20

QUESTION_BANK = [
    {
        "q": "台光電屬於哪一層？",
        "options": ["上游材料", "CCL", "PCB", "載板"],
        "answer": "CCL",
        "explain": "台光電主要是 CCL 銅箔基板，高頻高速材料是 AI 伺服器題材中的重要位置。",
    },
    {
        "q": "志聖（2467）在 PCB 供應鏈中比較像哪一種角色？",
        "options": ["銅箔", "CCL", "設備", "載板"],
        "answer": "設備",
        "explain": "志聖屬於設備廠，主要吃擴產、製程升級與 Capex 循環，不是材料本身。",
    },
    {
        "q": "如果載板 > CCL > PCB，通常代表什麼？",
        "options": ["標準輪動", "假輪動", "空頭", "資料錯誤"],
        "answer": "標準輪動",
        "explain": "這代表權力端或需求端先反應，再傳導到 CCL 與 PCB，屬於比較健康的輪動結構。",
    },
    {
        "q": "CB 是什麼？",
        "options": ["公司債", "可轉債", "普通股", "ETF"],
        "answer": "可轉債",
        "explain": "CB 是 Convertible Bond，可轉換公司債。對高波動股票而言，可能形成轉債套利或偏防守的參與方式。",
    },
    {
        "q": "哪一類公司通常比較接近『擴產前置訊號』？",
        "options": ["設備", "PCB", "普通股", "ETF"],
        "answer": "設備",
        "explain": "設備廠常在擴產預期或 Capex 循環中先有反應，但波動也可能很大。",
    },
    {
        "q": "若 PCB 大漲，但載板與 CCL 沒跟上，較可能是哪種狀態？",
        "options": ["標準輪動", "假輪動或末端補漲", "長期低估", "必然主升段"],
        "answer": "假輪動或末端補漲",
        "explain": "若權力端沒有同步轉強，只是 PCB 製造端先噴，常見於短線題材或末端補漲。",
    },
]


@st.cache_data(ttl=60 * 30, show_spinner=False)
def download_price(ticker: str, period: str = "1y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            close = df.xs("Close", axis=1, level=0).iloc[:, 0]
        elif "Adj Close" in df.columns.get_level_values(0):
            close = df.xs("Adj Close", axis=1, level=0).iloc[:, 0]
        else:
            return pd.DataFrame()
        out = pd.DataFrame({"Close": close})
    else:
        price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        if price_col not in df.columns:
            return pd.DataFrame()
        out = pd.DataFrame({"Close": df[price_col]})
    return out.dropna()


def safe_return(ticker: str, lookback: int = LOOKBACK_DAYS) -> float:
    df = download_price(ticker, period="6mo")
    if len(df) <= lookback:
        return float("nan")
    latest = float(df["Close"].iloc[-1])
    past = float(df["Close"].iloc[-lookback])
    if past == 0:
        return float("nan")
    return (latest / past - 1) * 100


def build_stock_table() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for category, group in STOCKS.items():
        for name, info in group.items():
            rows.append(
                {
                    "類別": category,
                    "公司": name,
                    "股號": info["ticker"].replace(".TW", ""),
                    "Ticker": info["ticker"],
                    "20日漲跌幅%": safe_return(info["ticker"]),
                    "是否有CB": "有" if info["cb"] else "無",
                    "產業定位": info["role"],
                    "權力分級": info["power"],
                }
            )
    return pd.DataFrame(rows)


def group_return(table: pd.DataFrame, category: str) -> float:
    values = table.loc[table["類別"] == category, "20日漲跌幅%"].dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


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
        df = download_price(info["ticker"], period=period)
        if df.empty:
            continue
        s = df["Close"].copy()
        s.name = name
        series_list.append(s)
    if not series_list:
        return pd.DataFrame()
    prices = pd.concat(series_list, axis=1).dropna(how="all").ffill().dropna()
    indexed = prices / prices.iloc[0] * 100
    out = pd.DataFrame({category: indexed.mean(axis=1)})
    return out


def build_all_group_indices(period: str = "1y") -> pd.DataFrame:
    frames = [build_group_index(group, period) for group in CORE_GROUPS]
    frames = [x for x in frames if not x.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).dropna()


def init_learning_state() -> None:
    if "answered" not in st.session_state:
        st.session_state.answered = {}


def record_answer(question: str, correct: bool) -> None:
    st.session_state.answered[question] = correct


def learning_score() -> Tuple[int, int]:
    total = len(st.session_state.answered)
    correct = sum(1 for v in st.session_state.answered.values() if v)
    return correct, total


with st.spinner("正在抓取台股資料，第一次載入可能較慢……"):
    table = build_stock_table()

leader_r = group_return(table, "載板")
ccl_r = group_return(table, "CCL")
pcb_r = group_return(table, "PCB")
state, explanation, level = determine_state(leader_r, ccl_r, pcb_r)

init_learning_state()

tab_dashboard, tab_chain, tab_learning, tab_strategy = st.tabs(["📈 儀表板", "🧭 產業鏈", "🎓 學習", "🎯 策略"])

with tab_dashboard:
    st.header("市場狀態")
    if level == "success":
        st.success(f"{state}\n\n{explanation}")
    elif level == "info":
        st.info(f"{state}\n\n{explanation}")
    else:
        st.warning(f"{state}\n\n{explanation}")

    col1, col2, col3 = st.columns(3)
    col1.metric("載板 Leader", fmt_pct(leader_r))
    col2.metric("CCL Core", fmt_pct(ccl_r))
    col3.metric("PCB Follower", fmt_pct(pcb_r))

    st.subheader("核心族群走勢（等權指數，起點=100）")
    group_indices = build_all_group_indices(period="1y")
    if group_indices.empty:
        st.info("目前無法取得足夠價格資料繪圖。")
    else:
        long_df = group_indices.reset_index().melt(id_vars="Date", var_name="族群", value_name="指數")
        fig = px.line(long_df, x="Date", y="指數", color="族群", title="載板 / CCL / PCB 族群相對走勢")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("20日強弱排名")
    rank_table = table.dropna(subset=["20日漲跌幅%"]).sort_values("20日漲跌幅%", ascending=False)
    rank_display = rank_table[["類別", "公司", "股號", "20日漲跌幅%", "是否有CB", "權力分級"]].copy()
    rank_display["20日漲跌幅%"] = rank_display["20日漲跌幅%"].round(2)
    st.dataframe(rank_display, use_container_width=True, hide_index=True)

with tab_chain:
    st.header("供應鏈結構")
    st.markdown(
        """
        **上游材料**（銅箔 / 玻纖 / PI） → **CCL 銅箔基板** → **PCB 硬板 / 軟板 / 載板**  
        旁支：**設備廠**不在材料鏈內，而是吃擴產與製程升級的 Capex 循環。
        """
    )

    display_table = table.copy()
    display_table["20日漲跌幅%"] = display_table["20日漲跌幅%"].map(lambda x: None if pd.isna(x) else round(float(x), 2))
    selected_category = st.selectbox("篩選類別", ["全部"] + list(STOCKS.keys()))
    if selected_category != "全部":
        display_table = display_table[display_table["類別"] == selected_category]
    st.dataframe(
        display_table[["類別", "公司", "股號", "20日漲跌幅%", "是否有CB", "產業定位", "權力分級"]],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("CB 觀察提醒")
    st.markdown(
        """
        - **有 CB**：代表未來可進一步研究轉債溢價率、轉換價、剩餘年限與套利空間。  
        - **無 CB**：只能用現股或其他工具參與，沒有直接轉債觀察角度。  
        - **高波動 + 有 CB**：通常比較值得納入可轉債觀察清單，但仍需看實際轉債條件。
        """
    )

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
        if is_correct:
            st.success("✔ 正確")
        else:
            st.error(f"❌ 這題答案是：{item['answer']}")
        st.write(item["explain"])

    if st.button("清除本次答題紀錄"):
        st.session_state.answered = {}
        st.rerun()

with tab_strategy:
    st.header("策略判讀")
    st.markdown(
        """
        v2 的目標不是自動交易，而是幫你建立判斷節奏：

        1. **先看權力端**：載板是否強？  
        2. **再看材料端**：CCL 是否跟上？  
        3. **最後看製造端**：PCB 是否落後且仍有補漲可能？
        """
    )

    st.subheader("目前狀態解讀")
    st.write(f"狀態：{state}")
    st.write(explanation)

    st.subheader("初步行動框架")
    if level == "success":
        st.success("可觀察 PCB 補漲，但仍需搭配個股型態、成交量與風控。")
    elif pcb_r > leader_r if not any(math.isnan(x) for x in [pcb_r, leader_r]) else False:
        st.warning("PCB 已強於載板，較像補漲末端或假輪動，避免只因漲幅追高。")
    else:
        st.info("訊號未明，先當作學習與觀察，不急著形成交易結論。")

    st.subheader("下一版可加入")
    st.markdown(
        """
        - 回測模組：驗證「載板 > CCL > PCB」是否真的有補漲效果。  
        - CB 模組：輸入轉換價、市價、債價後，計算溢價率。  
        - 個人學習紀錄：把答題紀錄存成檔案或資料庫。
        """
    )

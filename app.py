import math
from typing import Dict, Any, List

import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="PCB 供應鏈 AI 系統", layout="wide")

st.title("📊 PCB 供應鏈 AI 學習與分析系統 v1")
st.caption("用來引導理解 PCB / CCL / 載板產業鏈、可轉債資訊與族群相對強弱。非投資建議。")

STOCKS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "載板": {
        "欣興": {"ticker": "3037.TW", "cb": True, "role": "ABF / BT 載板，AI GPU / ASIC 需求相關"},
        "南電": {"ticker": "8046.TW", "cb": True, "role": "ABF 載板，半導體封裝材料鏈"},
        "景碩": {"ticker": "3189.TW", "cb": True, "role": "載板，封裝基板族群"},
    },
    "CCL": {
        "台光電": {"ticker": "2383.TW", "cb": False, "role": "高階 CCL，高頻高速材料"},
        "台燿": {"ticker": "6274.TW", "cb": True, "role": "高階 CCL，AI 伺服器材料"},
        "聯茂": {"ticker": "6213.TW", "cb": True, "role": "CCL，銅箔基板材料"},
    },
    "PCB": {
        "金像電": {"ticker": "2368.TW", "cb": True, "role": "伺服器 / AI PCB 硬板"},
        "華通": {"ticker": "2313.TW", "cb": True, "role": "PCB 硬板，消費電子與通訊板"},
        "健鼎": {"ticker": "3044.TW", "cb": False, "role": "PCB 硬板，規模型製造"},
    },
    "上游材料": {
        "金居": {"ticker": "8358.TW", "cb": True, "role": "銅箔，導電材料"},
        "富喬": {"ticker": "1815.TW", "cb": True, "role": "玻纖布，PCB 結構材料"},
        "台玻": {"ticker": "1802.TW", "cb": False, "role": "玻纖與玻璃材料"},
        "達邁": {"ticker": "3645.TW", "cb": True, "role": "PI 材料，軟板上游"},
    },
    "設備": {
        "志聖": {"ticker": "2467.TW", "cb": True, "role": "PCB / 載板製程設備，吃 Capex 循環"},
        "由田": {"ticker": "3455.TW", "cb": True, "role": "檢測設備，PCB / 半導體相關"},
        "群翊": {"ticker": "6664.TW", "cb": True, "role": "製程設備，PCB / 載板相關"},
    },
}

CORE_GROUPS = ["載板", "CCL", "PCB"]
LOOKBACK_DAYS = 20


@st.cache_data(ttl=60 * 30, show_spinner=False)
def download_price(ticker: str) -> pd.DataFrame:
    """Download recent price data and normalize columns for single-ticker yfinance output."""
    df = yf.download(ticker, period="6mo", auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance sometimes returns MultiIndex columns. Normalize to simple columns.
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

    out = out.dropna()
    return out


def safe_return(ticker: str, lookback: int = LOOKBACK_DAYS) -> float:
    df = download_price(ticker)
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
            ret = safe_return(info["ticker"])
            rows.append(
                {
                    "類別": category,
                    "公司": name,
                    "股號": info["ticker"].replace(".TW", ""),
                    "Ticker": info["ticker"],
                    "20日漲跌幅%": ret,
                    "是否有CB": "有" if info["cb"] else "無",
                    "產業定位": info["role"],
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


def determine_state(leader: float, ccl: float, pcb: float) -> tuple[str, str, str]:
    if any(math.isnan(x) for x in [leader, ccl, pcb]):
        return "⚪ 資料不足", "部分股價資料抓取失敗，先不要判斷。", "info"
    if leader > ccl > pcb:
        return "🔥 AI 主升段 / 標準輪動", "載板 > CCL > PCB，代表權力端先動、PCB 仍可能補漲。", "success"
    if pcb > leader:
        return "⚠️ 假輪動或末端補漲", "PCB 強於載板，可能是短線補漲或題材末端，不宜只看表面強勢。", "warning"
    if leader > 0 and ccl > 0 and pcb > 0:
        return "🟡 同步上漲", "三者都偏強，但輪動順序不明顯，策略訊號較弱。", "warning"
    return "🟡 混沌 / 觀望", "目前沒有清楚的載板 → CCL → PCB 輪動結構。", "warning"


with st.spinner("正在抓取台股資料，第一次載入可能較慢……"):
    table = build_stock_table()

leader_r = group_return(table, "載板")
ccl_r = group_return(table, "CCL")
pcb_r = group_return(table, "PCB")
state, explanation, level = determine_state(leader_r, ccl_r, pcb_r)

st.header("📈 市場狀態")
if level == "success":
    st.success(f"{state}\n\n{explanation}")
elif level == "info":
    st.info(f"{state}\n\n{explanation}")
else:
    st.warning(f"{state}\n\n{explanation}")

st.header("📊 核心族群 20 日強弱")
col1, col2, col3 = st.columns(3)
col1.metric("載板 Leader", fmt_pct(leader_r))
col2.metric("CCL Core", fmt_pct(ccl_r))
col3.metric("PCB Follower", fmt_pct(pcb_r))

st.header("🧭 供應鏈結構")
st.markdown(
    """
    **上游材料**（銅箔 / 玻纖 / PI） → **CCL 銅箔基板** → **PCB 硬板 / 軟板 / 載板**  
    旁支：**設備廠**不在材料鏈內，而是吃擴產與製程升級的 Capex 循環。
    """
)

st.header("🏢 公司資料與 CB 狀態")
display_table = table.copy()
display_table["20日漲跌幅%"] = display_table["20日漲跌幅%"].map(lambda x: None if pd.isna(x) else round(float(x), 2))
st.dataframe(
    display_table[["類別", "公司", "股號", "20日漲跌幅%", "是否有CB", "產業定位"]],
    use_container_width=True,
    hide_index=True,
)

st.header("🎓 學習模式")
question_bank = [
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
]

selected = st.selectbox("選擇一題練習", [item["q"] for item in question_bank])
item = next(x for x in question_bank if x["q"] == selected)
answer = st.radio(item["q"], item["options"], index=None)
if answer:
    if answer == item["answer"]:
        st.success("✔ 正確")
    else:
        st.error(f"❌ 這題答案是：{item['answer']}")
    st.write(item["explain"])

st.header("🎯 策略提醒")
st.markdown(
    """
    這個 v1 只做**學習與觀察**，不是自動交易。  
    初步判斷框架：
    - 載板 > CCL > PCB：標準輪動，可觀察 PCB 是否補漲。
    - PCB > 載板：可能是假輪動或末端補漲。
    - 三者同步：題材擴散，但時間差策略優勢較弱。
    """
)

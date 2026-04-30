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
    }
}

LOOKBACK_DAYS = 20

@st.cache_data(ttl=60 * 30, show_spinner=False)
def download_price(ticker: str, period: str = "1y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if "Adj Close" in df.columns:
        return df[["Adj Close"]].rename(columns={"Adj Close": "Close"})
    return df[["Close"]]


def safe_return(ticker: str, lookback: int) -> float:
    df = download_price(ticker)
    if len(df) <= lookback:
        return float("nan")
    return (df["Close"].iloc[-1] / df["Close"].iloc[-lookback] - 1) * 100


def build_stock_table():
    rows = []
    for cat, group in STOCKS.items():
        for name, info in group.items():
            rows.append({
                "類別": cat,
                "公司": name,
                "股號": info["ticker"].replace(".TW", ""),
                "20日漲跌幅%": safe_return(info["ticker"], 20),
                "60日漲跌幅%": safe_return(info["ticker"], 60),
                "CB": "有" if info["cb"] else "無"
            })
    return pd.DataFrame(rows)


table = build_stock_table()

st.header("📊 強弱（含60日）")

clean = table.copy()
clean["20日漲跌幅%"] = clean["20日漲跌幅%"].round(2)
clean["60日漲跌幅%"] = clean["60日漲跌幅%"].round(2)

st.dataframe(clean, use_container_width=True)

st.info("✔ 已加入60日漲幅，用來觀察中期趨勢 vs 短期動能")

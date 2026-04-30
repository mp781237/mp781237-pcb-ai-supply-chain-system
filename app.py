import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(layout="wide")

st.title("📊 PCB供應鏈 AI 系統 v1")

stocks = {
    "載板": {
        "欣興": {"ticker": "3037.TW", "cb": True},
        "南電": {"ticker": "8046.TW", "cb": True},
        "景碩": {"ticker": "3189.TW", "cb": True},
    },
    "CCL": {
        "台光電": {"ticker": "2383.TW", "cb": False},
        "台燿": {"ticker": "6274.TW", "cb": True},
        "聯茂": {"ticker": "6213.TW", "cb": True},
    },
    "PCB": {
        "金像電": {"ticker": "2368.TW", "cb": True},
        "華通": {"ticker": "2313.TW", "cb": True},
        "健鼎": {"ticker": "3044.TW", "cb": False},
    }
}

def get_return(ticker):
    df = yf.download(ticker, period="3mo", progress=False)
    if len(df) < 20:
        return 0
    return (df["Close"].iloc[-1] / df["Close"].iloc[-20] - 1) * 100


def calc_group(group):
    returns = []
    for k,v in group.items():
        r = get_return(v["ticker"])
        returns.append(r)
    return sum(returns)/len(returns)

leader = calc_group(stocks["載板"])
ccl = calc_group(stocks["CCL"])
pcb = calc_group(stocks["PCB"])

st.header("📈 市場狀態")

if leader > ccl > pcb:
    state = "🔥 AI主升段"
    action = "👉 可布局 PCB"
elif pcb > leader:
    state = "⚠️ 假輪動"
    action = "👉 不建議進場"
else:
    state = "🟡 混沌"
    action = "👉 觀望"

st.subheader(state)
st.write(action)

st.header("📊 強弱")
col1, col2, col3 = st.columns(3)
col1.metric("載板", f"{leader:.2f}%")
col2.metric("CCL", f"{ccl:.2f}%")
col3.metric("PCB", f"{pcb:.2f}%")

st.header("🎓 學習模式")

q = st.radio("台光電屬於哪一層？",
             ["材料","CCL","PCB","載板"])

if q:
    if q == "CCL":
        st.success("✔ 正確")
        st.write("上游：銅箔 / 玻纖 / 樹脂")
        st.write("下游：PCB / 載板")
        st.write("投資意義：有定價權")
    else:
        st.error("❌ 再想一下")

st.header("🏢 公司資料")

for cat, group in stocks.items():
    st.subheader(cat)
    for name, info in group.items():
        st.write(f"{name} ({info['ticker']})")
        st.write(f"CB：{'有' if info['cb'] else '無'}")

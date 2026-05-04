import math
from typing import Dict, Any, List, Tuple
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="PCB 供應鏈 AI 系統 v3", layout="wide")
st.title("📊 PCB 供應鏈 AI 學習與分析系統 v3")
st.caption("v2.5 個股選股回測 + v3 嚴格條件等權回測（含敏感度測試）。僅供研究，非投資建議。")

STOCKS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "載板": {"欣興":{"code":"3037","cb":True,"role":"ABF/BT載板，AI需求核心"},"南電":{"code":"8046","cb":True,"role":"ABF載板"},"景碩":{"code":"3189","cb":True,"role":"封裝基板"}},
    "CCL": {"台光電":{"code":"2383","cb":False,"role":"高階CCL"},"台燿":{"code":"6274","cb":True,"role":"高階CCL"},"聯茂":{"code":"6213","cb":True,"role":"CCL材料"}},
    "PCB": {"金像電":{"code":"2368","cb":True,"role":"AI伺服器PCB"},"華通":{"code":"2313","cb":True,"role":"PCB硬板"},"健鼎":{"code":"3044","cb":False,"role":"PCB硬板"}},
    "上游材料": {"金居":{"code":"8358","cb":True,"role":"銅箔"},"富喬":{"code":"1815","cb":True,"role":"玻纖布"},"台玻":{"code":"1802","cb":False,"role":"玻纖"},"達邁":{"code":"3645","cb":True,"role":"PI材料"}},
    "設備": {"志聖":{"code":"2467","cb":True,"role":"PCB/載板設備"},"由田":{"code":"3455","cb":True,"role":"檢測設備"},"群翊":{"code":"6664","cb":True,"role":"製程設備"}}
}
CORE = ["載板","CCL","PCB"]
BENCH_CODE = "0050"
FEE_BUY = 0.001425; FEE_SELL = 0.001425; TAX_SELL = 0.003

@st.cache_data(ttl=1800, show_spinner=False)
def dl_ticker(ticker: str, period: str="5y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        k = "Adj Close" if "Adj Close" in df.columns.get_level_values(0) else "Close"
        if k not in df.columns.get_level_values(0): return pd.DataFrame()
        s = df.xs(k, axis=1, level=0).iloc[:,0]
    else:
        k = "Adj Close" if "Adj Close" in df.columns else "Close"
        if k not in df.columns: return pd.DataFrame()
        s = df[k]
    return pd.DataFrame({"Close": pd.to_numeric(s, errors="coerce")}).dropna()

@st.cache_data(ttl=1800, show_spinner=False)
def dl(code: str, period: str="5y") -> Tuple[pd.DataFrame, str]:
    for t in [f"{code}.TW", f"{code}.TWO"]:
        df = dl_ticker(t, period)
        if not df.empty: return df, t
    return pd.DataFrame(), "抓取失敗"

def ret(code: str, n: int) -> Tuple[float,str]:
    df,t = dl(code,"1y")
    if len(df)<=n: return float("nan"), t
    return float((df.Close.iloc[-1]/df.Close.iloc[-n]-1)*100), t

def momentum(r20, r60):
    if pd.isna(r20) or pd.isna(r60): return "資料不足"
    if r20>=0 and r60<0: return "🔥 初動/轉強"
    if r20>=0 and r60>=0: return "🚀 主升/強勢"
    if r20<0 and r60>=0: return "🟡 強勢回檔"
    return "❄️ 弱勢"

def stock_table():
    rows=[]
    for cat,g in STOCKS.items():
        for name,info in g.items():
            r20,t = ret(info["code"],20); r60,t60 = ret(info["code"],60)
            rows.append({"類別":cat,"公司":name,"股號":info["code"],"實際Ticker":t if t!="抓取失敗" else t60,"20日漲跌幅%":r20,"60日漲跌幅%":r60,"動能象限":momentum(r20,r60),"是否有CB":"有" if info["cb"] else "無","產業定位":info["role"]})
    return pd.DataFrame(rows)

def group_index(cat: str, period: str="5y"):
    arr=[]
    for name,info in STOCKS[cat].items():
        df,_=dl(info["code"],period)
        if not df.empty:
            s=df.Close.copy(); s.name=name; arr.append(s)
    if not arr: return pd.DataFrame()
    p=pd.concat(arr,axis=1).dropna(how="all").ffill().dropna()
    return pd.DataFrame({cat:(p/p.iloc[0]*100).mean(axis=1)})

def all_indices(period="1y"):
    frames=[group_index(c,period) for c in CORE]
    frames=[x for x in frames if not x.empty]
    return pd.concat(frames,axis=1).dropna() if frames else pd.DataFrame()

def pct(v):
    return "資料不足" if v is None or (isinstance(v,float) and math.isnan(v)) else f"{v:.2f}%"

def mdd(eq):
    return float((eq/eq.cummax()-1).min()) if not eq.empty else float("nan")

def cagr(eq):
    if eq.empty or len(eq)<2: return float("nan")
    days=max((eq.index[-1]-eq.index[0]).days,1)
    return float((eq.iloc[-1]/eq.iloc[0])**(365/days)-1)

def stock_prices(cat, period):
    arr=[]
    for name,info in STOCKS[cat].items():
        df,_=dl(info["code"],period)
        if not df.empty:
            s=df.Close.copy(); s.name=name; arr.append(s)
    return pd.concat(arr,axis=1).dropna(how="all").ffill().dropna() if arr else pd.DataFrame()

def run_bt(period, max_hold, exposure, top_n, stop_loss, take_profit, require_initial):
    gi=all_indices(period); pxs=stock_prices("PCB",period)
    if gi.empty or pxs.empty: return pd.DataFrame(), pd.DataFrame(), {}
    idx=gi.index.intersection(pxs.index); gi=gi.loc[idx]; pxs=pxs.loc[idx]
    g20=pd.DataFrame({c:gi[c]/gi[c].shift(20)-1 for c in CORE}).dropna()
    r20=pxs/pxs.shift(20)-1; r60=pxs/pxs.shift(60)-1; dr=pxs.pct_change().fillna(0)
    idx=g20.index.intersection(r20.dropna().index).intersection(r60.dropna().index)
    equity=1.0; bench=1.0; pos={}; trades=[]; rec=[]; realized=0.0
    for i in range(1,len(idx)):
        d=idx[i]; prev=idx[i-1]
        daily_port=0.0; exits=[]
        for s,p in pos.items():
            tr=float(dr.loc[d,s]); p["ret"]=(1+p["ret"])*(1+tr)-1; p["hold"]+=1
            reason=None
            if p["ret"]<=-abs(stop_loss): reason="停損"
            elif p["ret"]>=abs(take_profit): reason="停利"
            elif p["hold"]>=max_hold: reason="時間出場"
            elif float(r20.loc[prev,s])<0: reason="20日轉弱"
            if reason:
                trades.append({"股票":s,"進場日":p["entry"],"出場日":d,"持有天數":p["hold"],"出場原因":reason,"交易報酬%":p["ret"]*100})
                realized += (exposure/max(1,top_n))*p["ret"]
                exits.append(s)
            else:
                daily_port += (exposure/max(1,top_n))*tr
        for s in exits: pos.pop(s,None)
        equity *= (1+daily_port)
        bench *= (1+float(dr.loc[d].mean()))
        signal = bool(g20.loc[prev,"載板"] > g20.loc[prev,"CCL"] > g20.loc[prev,"PCB"])
        if signal:
            scores=r20.loc[prev].dropna().sort_values(ascending=False)
            for s in scores.index:
                if len(pos)>=top_n: break
                if s in pos: continue
                if float(r20.loc[prev,s])<=0: continue
                if require_initial and not (float(r20.loc[prev,s])>0 and float(r60.loc[prev,s])<0): continue
                pos[s]={"entry":d,"hold":0,"ret":0.0}
        rec.append({"Date":d,"策略淨值":equity,"PCB等權買進持有":bench,"持股數":len(pos)})
    curve=pd.DataFrame(rec).set_index("Date") if rec else pd.DataFrame()
    log=pd.DataFrame(trades)
    metrics={"總報酬%":(curve["策略淨值"].iloc[-1]-1)*100 if not curve.empty else float("nan"),"CAGR%":cagr(curve["策略淨值"])*100 if not curve.empty else float("nan"),"MDD%":mdd(curve["策略淨值"])*100 if not curve.empty else float("nan"),"交易次數":float(len(log)),"勝率%":log["交易報酬%"].gt(0).mean()*100 if not log.empty else float("nan"),"平均單筆%":log["交易報酬%"].mean() if not log.empty else float("nan"),"PCB等權買進持有%":(curve["PCB等權買進持有"].iloc[-1]-1)*100 if not curve.empty else float("nan")}
    return curve,log,metrics

def run_strict_bt(period, lookback, hold, tp, sl, leader_min, core_min, follower_max, bench_min, position_size):
    """v3 嚴格回測：6 條件全部成立 → 進場 PCB 三檔等權，含交易成本。"""
    gi=all_indices(period); pxs=stock_prices("PCB",period); bdf,_=dl(BENCH_CODE,period)
    if gi.empty or pxs.empty or bdf.empty: return pd.DataFrame(),pd.DataFrame(),{},pd.DataFrame()
    bench=bdf.Close
    idx=gi.index.intersection(pxs.index).intersection(bench.index)
    if len(idx)<lookback+5: return pd.DataFrame(),pd.DataFrame(),{},pd.DataFrame()
    gi=gi.loc[idx]; pxs=pxs.loc[idx]; bench=bench.loc[idx]
    glb=pd.DataFrame({c:gi[c]/gi[c].shift(lookback)-1 for c in CORE})
    blb=bench/bench.shift(lookback)-1
    sig=((glb["載板"]>leader_min)&(glb["CCL"]>core_min)&(glb["載板"]>glb["CCL"])
         &(glb["CCL"]>glb["PCB"])&(glb["PCB"]<follower_max)&(blb>bench_min))
    cash=1.0; in_pos=False; entry_i=None; units={}; entry_px={}; entry_inv=0.0
    cooldown=False; trades=[]; rec=[]
    codes=list(pxs.columns)
    bench_eq=bench/bench.iloc[0]; foll_eq=(pxs/pxs.iloc[0]).mean(axis=1)
    for i,d in enumerate(idx):
        mv=sum(units[c]*pxs.at[d,c] for c in units) if in_pos else 0.0
        if in_pos:
            held=i-entry_i
            basis=sum(units[c]*entry_px[c] for c in units)
            gret=mv/basis-1 if basis>0 else 0.0
            reason=None
            if held>=hold: reason=f"持滿{hold}日"
            elif gret>=tp: reason="停利"
            elif gret<=sl: reason="停損"
            if reason:
                proceeds=mv*(1-FEE_SELL-TAX_SELL)
                cash+=proceeds
                nret=proceeds/entry_inv-1
                trades.append({"進場日":idx[entry_i].date(),"出場日":d.date(),"持有天數":held,"出場原因":reason,
                               "Leader_LB%":float(glb.loc[idx[entry_i-1],"載板"])*100,"Core_LB%":float(glb.loc[idx[entry_i-1],"CCL"])*100,
                               "Follower_LB%":float(glb.loc[idx[entry_i-1],"PCB"])*100,"Bench_LB%":float(blb.loc[idx[entry_i-1]])*100,
                               "毛報酬%":gret*100,"淨報酬%":nret*100,"出場後淨值":cash})
                in_pos=False; units={}; entry_px={}; cooldown=True
                rec.append({"Date":d,"策略淨值":cash,"0050買進持有":float(bench_eq.loc[d]),"PCB等權買進持有":float(foll_eq.loc[d])})
                continue
        if (not in_pos) and i>=1 and (not cooldown) and bool(sig.iloc[i-1]) and pd.notna(sig.iloc[i-1]):
            buy=cash*position_size
            per=buy/len(codes)
            ok=True; tu={}; te={}
            for c in codes:
                px=pxs.at[d,c]
                if pd.isna(px) or px<=0: ok=False; break
                shares=(per/(1+FEE_BUY))/px
                tu[c]=shares; te[c]=float(px)
            if ok:
                units=tu; entry_px=te; entry_inv=buy
                cash-=buy; in_pos=True; entry_i=i
        cooldown=False
        eq=cash+(sum(units[c]*pxs.at[d,c] for c in units) if in_pos else 0.0)
        rec.append({"Date":d,"策略淨值":eq,"0050買進持有":float(bench_eq.loc[d]),"PCB等權買進持有":float(foll_eq.loc[d])})
    curve=pd.DataFrame(rec).drop_duplicates("Date",keep="last").set_index("Date")
    log=pd.DataFrame(trades)
    eq=curve["策略淨值"]
    metrics={
        "總報酬%":(eq.iloc[-1]-1)*100 if not eq.empty else float("nan"),
        "CAGR%":cagr(eq)*100,
        "MDD%":mdd(eq)*100,
        "Sharpe":float(eq.pct_change().mean()*252/(eq.pct_change().std()*math.sqrt(252))) if eq.pct_change().std()>0 else float("nan"),
        "交易次數":float(len(log)),
        "勝率%":log["淨報酬%"].gt(0).mean()*100 if not log.empty else float("nan"),
        "平均單筆%":log["淨報酬%"].mean() if not log.empty else float("nan"),
        "0050買進持有%":(curve["0050買進持有"].iloc[-1]-1)*100,
        "PCB等權買進持有%":(curve["PCB等權買進持有"].iloc[-1]-1)*100,
    }
    sig_df=pd.DataFrame({"Date":idx,"訊號":sig.values}).set_index("Date")
    return curve,log,metrics,sig_df

def run_sensitivity(period, leader_min, core_min, follower_max, bench_min, position_size,
                    lookbacks, holdings, tps, sls):
    rows=[]
    total=len(lookbacks)*len(holdings)*len(tps)*len(sls); cnt=0
    bar=st.progress(0.0, text=f"敏感度測試 0/{total}")
    for lb in lookbacks:
        for hd in holdings:
            for tp in tps:
                for sl in sls:
                    cnt+=1; bar.progress(cnt/total, text=f"敏感度測試 {cnt}/{total}")
                    _,log,m,_=run_strict_bt(period,lb,hd,tp,sl,leader_min,core_min,follower_max,bench_min,position_size)
                    rows.append({"lookback":lb,"hold":hd,"TP":tp,"SL":sl,
                                 "CAGR%":m.get("CAGR%",float("nan")),"MDD%":m.get("MDD%",float("nan")),
                                 "Sharpe":m.get("Sharpe",float("nan")),"勝率%":m.get("勝率%",float("nan")),
                                 "交易次數":int(m.get("交易次數",0))})
    bar.empty()
    return pd.DataFrame(rows)

def add_quadrants(fig,df):
    fig.add_hline(y=0,line_dash="dash",line_color="gray"); fig.add_vline(x=0,line_dash="dash",line_color="gray")
    fig.add_annotation(x=1,y=1,xref="paper",yref="paper",text="🚀 主升/強勢",showarrow=False,xanchor="right")
    fig.add_annotation(x=0,y=1,xref="paper",yref="paper",text="🔥 初動/轉強",showarrow=False,xanchor="left")
    return fig

with st.spinner("正在抓取台股資料……"):
    table=stock_table()

leader=table[table.類別=="載板"]["20日漲跌幅%"].mean(); ccl=table[table.類別=="CCL"]["20日漲跌幅%"].mean(); pcb=table[table.類別=="PCB"]["20日漲跌幅%"].mean()
state="🔥 標準輪動" if leader>ccl>pcb else ("⚠️ PCB強於載板" if pcb>leader else "🟡 觀望")

t1,t2,t3,t4,t5=st.tabs(["📈 儀表板","🧭 產業鏈","🎓 學習","📊 個股回測v2","📐 嚴格回測v3"])
with t1:
    st.header("市場狀態"); st.info(state)
    c1,c2,c3=st.columns(3); c1.metric("載板20日",pct(leader)); c2.metric("CCL20日",pct(ccl)); c3.metric("PCB20日",pct(pcb))
    q=table.dropna(subset=["20日漲跌幅%","60日漲跌幅%"]).copy()
    if not q.empty:
        fig=px.scatter(q,x="60日漲跌幅%",y="20日漲跌幅%",color="類別",text="公司",hover_data=["股號","是否有CB","動能象限"])
        fig.update_traces(textposition="top center",marker=dict(size=12)); fig.update_layout(height=600)
        st.plotly_chart(add_quadrants(fig,q),use_container_width=True)
    show=table.copy(); show["20日漲跌幅%"]=pd.to_numeric(show["20日漲跌幅%"],errors="coerce").round(2); show["60日漲跌幅%"]=pd.to_numeric(show["60日漲跌幅%"],errors="coerce").round(2)
    st.dataframe(show.sort_values("20日漲跌幅%",ascending=False),use_container_width=True,hide_index=True)
with t2:
    cat=st.selectbox("類別",["全部"]+list(STOCKS.keys()))
    d=show if cat=="全部" else show[show.類別==cat]
    st.dataframe(d,use_container_width=True,hide_index=True)
with t3:
    st.write("台光電屬於哪一層？")
    a=st.radio("答案",["上游材料","CCL","PCB","載板"],index=None)
    if a: st.success("正確，台光電是 CCL。") if a=="CCL" else st.error("答案是 CCL。")
with t4:
    st.header("個股輪動回測 v2")
    st.markdown("輪動成立後，從 PCB 個股挑 20日動能最強者；加入停損、停利、持有天數與曝險控制。")
    a,b,c=st.columns(3); period=a.selectbox("期間",["2y","5y","10y"],index=1); hold=b.slider("最多持有天數",5,60,20,5); topn=c.slider("最多持股",1,3,1)
    d,e,f=st.columns(3); exp=d.slider("總曝險",0.1,1.0,0.5,0.1); sl=e.slider("停損",0.03,0.20,0.08,0.01); tp=f.slider("停利",0.05,0.50,0.20,0.05)
    initial=st.checkbox("只買初動股（20日>0且60日<0）",value=False)
    if st.button("執行v2回測"):
        curve,log,met=run_bt(period,hold,exp,topn,sl,tp,initial)
        if curve.empty: st.error("資料不足或無法回測。")
        else:
            m1,m2,m3,m4=st.columns(4); m1.metric("總報酬",pct(met["總報酬%"])); m2.metric("CAGR",pct(met["CAGR%"])); m3.metric("MDD",pct(met["MDD%"])); m4.metric("交易次數",int(met["交易次數"]))
            m5,m6,m7=st.columns(3); m5.metric("勝率",pct(met["勝率%"])); m6.metric("平均單筆",pct(met["平均單筆%"])); m7.metric("PCB等權買進持有",pct(met["PCB等權買進持有%"]));
            st.plotly_chart(px.line(curve.reset_index().melt(id_vars="Date",value_vars=["策略淨值","PCB等權買進持有"],var_name="項目",value_name="淨值"),x="Date",y="淨值",color="項目"),use_container_width=True)
            if log.empty: st.info("沒有完成交易。")
            else:
                log["交易報酬%"] = pd.to_numeric(log["交易報酬%"],errors="coerce").round(2); st.dataframe(log,use_container_width=True,hide_index=True); st.download_button("下載交易紀錄",log.to_csv(index=False).encode("utf-8-sig"),"trade_log_v2.csv")
with t5:
    st.header("📐 嚴格條件回測 v3")
    st.markdown("**6 條件全部成立才進場**：①Leader_LB > leader_min ②Core_LB > core_min ③Leader>Core ④Core>Follower ⑤Follower<follower_max ⑥0050_LB>bench_min。進場買 PCB 三檔等權（金像電/華通/健鼎），含交易成本（買 0.1425% / 賣 0.1425% + 證交稅 0.3%）。")
    cA,cB,cC=st.columns(3)
    period_v3=cA.selectbox("期間",["2y","5y","10y","max"],index=2,key="period_v3")
    lookback_v3=cB.slider("Lookback (天)",10,60,20,5,key="lb_v3")
    hold_v3=cC.slider("最多持有 (天)",10,60,20,5,key="hd_v3")
    cD,cE,cF=cA,cB,cC
    cD,cE,cF=st.columns(3)
    tp_v3=cD.slider("停利",0.05,0.40,0.15,0.05,key="tp_v3")
    sl_v3=cE.slider("停損 (絕對值)",0.05,0.20,0.10,0.01,key="sl_v3")
    pos_v3=cF.slider("進場部位 (% 現金)",0.1,1.0,0.5,0.1,key="pos_v3")
    cG,cH,cI,cJ=st.columns(4)
    leader_min=cG.slider("Leader_min",0.0,0.30,0.10,0.01,key="lm")
    core_min=cH.slider("Core_min",0.0,0.20,0.05,0.01,key="cm")
    follower_max=cI.slider("Follower_max",0.0,0.20,0.05,0.01,key="fm")
    bench_min=cJ.slider("Bench_min",-0.10,0.10,0.0,0.01,key="bm")
    sl_v3=-abs(sl_v3)
    if st.button("執行 v3 回測",key="run_v3"):
        with st.spinner("回測中…"):
            curve,log,met,sigdf=run_strict_bt(period_v3,lookback_v3,hold_v3,tp_v3,sl_v3,
                                              leader_min,core_min,follower_max,bench_min,pos_v3)
        if curve.empty: st.error("資料不足或回測失敗。")
        else:
            n_sig=int(sigdf["訊號"].fillna(False).astype(bool).sum())
            st.caption(f"期間訊號日數：{n_sig} 天 / 總交易日 {len(sigdf)} 天")
            m1,m2,m3,m4=st.columns(4)
            m1.metric("總報酬",pct(met["總報酬%"])); m2.metric("CAGR",pct(met["CAGR%"]))
            m3.metric("MDD",pct(met["MDD%"])); m4.metric("Sharpe",f'{met["Sharpe"]:.2f}' if not math.isnan(met["Sharpe"]) else "—")
            m5,m6,m7,m8=st.columns(4)
            m5.metric("交易次數",int(met["交易次數"])); m6.metric("勝率",pct(met["勝率%"]))
            m7.metric("平均單筆",pct(met["平均單筆%"])); m8.metric("0050同期",pct(met["0050買進持有%"]))
            comp=curve.reset_index().melt(id_vars="Date",value_vars=["策略淨值","0050買進持有","PCB等權買進持有"],var_name="項目",value_name="淨值")
            fig=px.line(comp,x="Date",y="淨值",color="項目",title="策略 vs 0050 vs PCB 等權")
            st.plotly_chart(fig,use_container_width=True)
            dd=curve["策略淨值"]/curve["策略淨值"].cummax()-1
            fig2=px.area(dd.reset_index(),x="Date",y="策略淨值",title="策略回撤")
            fig2.update_traces(fillcolor="rgba(220,60,60,0.4)",line_color="rgb(220,60,60)")
            fig2.update_yaxes(tickformat=".0%"); st.plotly_chart(fig2,use_container_width=True)
            if log.empty:
                st.info("沒有完成交易。建議放寬條件（降低 leader_min 或 core_min）試試。")
            else:
                for c in ["Leader_LB%","Core_LB%","Follower_LB%","Bench_LB%","毛報酬%","淨報酬%"]:
                    if c in log.columns: log[c]=pd.to_numeric(log[c],errors="coerce").round(2)
                st.dataframe(log,use_container_width=True,hide_index=True)
                st.download_button("下載交易紀錄",log.to_csv(index=False).encode("utf-8-sig"),"trade_log_v3.csv",key="dl_v3")
    with st.expander("🧪 敏感度測試（81 組參數，較慢）"):
        st.caption("跑 lookback × holding × TP × SL = 3×3×3×3 = 81 組，每組約 1-3 秒。")
        if st.button("執行敏感度測試",key="run_sens"):
            with st.spinner("跑 81 組參數中…"):
                sens=run_sensitivity(period_v3,leader_min,core_min,follower_max,bench_min,pos_v3,
                                     [10,20,40],[10,20,40],[0.10,0.15,0.20],[-0.05,-0.10,-0.15])
            for c in ["CAGR%","MDD%","Sharpe","勝率%"]: sens[c]=pd.to_numeric(sens[c],errors="coerce").round(2)
            st.dataframe(sens.sort_values("CAGR%",ascending=False),use_container_width=True,hide_index=True)
            st.download_button("下載敏感度結果",sens.to_csv(index=False).encode("utf-8-sig"),"sensitivity_v3.csv",key="dl_sens")

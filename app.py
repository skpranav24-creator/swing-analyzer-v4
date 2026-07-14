import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests, json, os
from datetime import datetime

st.set_page_config(page_title="Swing Analyzer V4", page_icon="🔔", layout="wide")
st.title("🔔 Swing Analyzer V4")
st.caption("Rule-based NSE swing analyzer with watchlist scanning and Telegram signal-change alerts.")

STATE_FILE = "signal_state.json"

def secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.getenv(name, default)

def load_state():
    if not os.path.exists(STATE_FILE): return {}
    try:
        with open(STATE_FILE, "r") as f: return json.load(f)
    except Exception: return {}

def save_state(x):
    with open(STATE_FILE, "w") as f: json.dump(x, f, indent=2)

def send_telegram(message):
    token = secret("TELEGRAM_BOT_TOKEN")
    chat_id = secret("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False, "Telegram token/chat ID not configured."
    try:
        u = f"https://api.telegram.org/bot{token}/sendMessage"
        res = requests.post(u, data={"chat_id": chat_id, "text": message}, timeout=15)
        res.raise_for_status()
        return True, "Sent"
    except Exception as e:
        return False, str(e)

@st.cache_data(ttl=60)
def get_data(t, period="5y"):
    d = yf.download(
        t,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False
    )
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d[["Open","High","Low","Close","Volume"]].dropna()

def calc_rsi(s, n=14):
    delta=s.diff()
    gain=delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss=(-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs=gain/loss.replace(0,np.nan)
    return 100-100/(1+rs)

def enrich(x):
    d=x.copy()
    d["EMA20"]=d["Close"].ewm(span=20,adjust=False).mean()
    d["EMA50"]=d["Close"].ewm(span=50,adjust=False).mean()
    d["EMA200"]=d["Close"].ewm(span=200,adjust=False).mean()
    d["RSI"]=calc_rsi(d["Close"])
    e12=d["Close"].ewm(span=12,adjust=False).mean()
    e26=d["Close"].ewm(span=26,adjust=False).mean()
    d["MACD"]=e12-e26
    d["MACDS"]=d["MACD"].ewm(span=9,adjust=False).mean()
    d["VOL20"]=d["Volume"].rolling(20).mean()
    tr=pd.concat([(d["High"]-d["Low"]),(d["High"]-d["Close"].shift()).abs(),
                  (d["Low"]-d["Close"].shift()).abs()],axis=1).max(axis=1)
    d["ATR"]=tr.rolling(14).mean()
    up=d["High"].diff(); down=-d["Low"].diff()
    pdm=pd.Series(np.where((up>down)&(up>0),up,0.0),index=d.index)
    mdm=pd.Series(np.where((down>up)&(down>0),down,0.0),index=d.index)
    atrs=tr.ewm(alpha=1/14,adjust=False).mean()
    pdi=100*pdm.ewm(alpha=1/14,adjust=False).mean()/atrs.replace(0,np.nan)
    mdi=100*mdm.ewm(alpha=1/14,adjust=False).mean()/atrs.replace(0,np.nan)
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    d["ADX"]=dx.ewm(alpha=1/14,adjust=False).mean()
    d["RES20"]=d["High"].shift(1).rolling(20).max()
    return d

def score_row(r):
    score=0; reasons=[]
    def add(ok,pts,good,bad):
        nonlocal score
        if bool(ok): score+=pts; reasons.append(("✅",good))
        else: reasons.append(("❌",bad))
    add(r.Close>r.EMA20,20,"Price is above EMA20","Price is below EMA20")
    add(r.EMA20>r.EMA50,20,"EMA20 is above EMA50","EMA20 is below EMA50")
    add(r.Close>r.EMA200,15,"Price is above EMA200","Price is below EMA200")
    add(50<=r.RSI<=70,15,"RSI is in the constructive 50–70 zone",f"RSI is {r.RSI:.1f}, outside preferred entry zone")
    add(r.MACD>r.MACDS,10,"MACD is bullish","MACD is not bullish")
    add(r.ADX>=20,10,"ADX shows meaningful trend strength","ADX shows weak trend strength")
    add(r.Volume>r.VOL20*1.2,5,"Volume confirms the move","No strong volume confirmation")
    add(r.Close>r.RES20,5,"20-day breakout confirmed","No 20-day breakout")
    return score,reasons

def new_entry_decision(score):
    if score>=85: return "🟢 STRONG BUY"
    if score>=70: return "🟢 BUY"
    if score>=55: return "🟠 WATCH"
    if score>=35: return "🟡 WAIT"
    return "🔴 AVOID"

def owned_decision(score,r):
    if r.Close<r.EMA50 and r.RSI<40: return "🔴 EXIT / REVIEW"
    if r.Close<r.EMA20 or r.MACD<r.MACDS: return "🟠 REDUCE / TIGHTEN STOP"
    if score>=55: return "🔵 HOLD"
    return "🟡 HOLD WITH CAUTION"

def analyze(symbol):
    ticker = symbol.strip().upper()
    if not ticker.endswith(".NS"):
        ticker += ".NS"

    raw = get_data(ticker)

    if len(raw) < 200:
        raise ValueError(f"Not enough historical data — received only {len(raw)} rows")

    df = enrich(raw)
    r = df.iloc[-1]
    score, reasons = score_row(r)

    price = float(r.Close)
    atr = float(r.ATR)
    stop = max(0.01, price - 1.5 * atr)
    risk = max(price - stop, 0.01)

    return ticker, df, r, score, reasons, price, stop, price + 2 * risk, price + 3 * risk

st.sidebar.header("Trade setup")
symbol=st.sidebar.text_input("NSE symbol","SILVERBEES").strip().upper()
position=st.sidebar.radio("Do you already own it?",["No — New entry","Yes — I own it"])
avg_price=None
if position.startswith("Yes"):
    avg_price=st.sidebar.number_input("Average buy price (₹)",min_value=0.01,value=200.0,step=1.0)
capital=st.sidebar.number_input("Available capital (₹)",1000,10000000,200000,5000)
risk_pct=st.sidebar.slider("Maximum risk per trade (%)",0.25,2.0,0.5,0.25)

st.sidebar.divider()
st.sidebar.header("Watchlist alerts")
watch_text=st.sidebar.text_area("Symbols, comma separated","SILVERBEES,GOLDBEES,TATAMOTORS")
notify_changes=st.sidebar.checkbox("Send Telegram alert when signal changes",value=True)
scan=st.sidebar.button("🔔 Scan watchlist now",use_container_width=True)
test=st.sidebar.button("Send test Telegram message",use_container_width=True)

if test:
    ok,msg=send_telegram("✅ Swing Analyzer V4 Telegram connection is working.")
    st.sidebar.success(msg) if ok else st.sidebar.error(msg)

if scan:
    state=load_state(); rows=[]; changed=[]
    for s in [x.strip().upper() for x in watch_text.split(",") if x.strip()]:
        try:
            t,df0,r0,sc,rea,p,sl,t1,t2=analyze(s)
            sig=new_entry_decision(sc)
            old=state.get(t)
            rows.append({"Symbol":t,"Price":round(p,2),"Score":sc,"Signal":sig,"Previous":old or "First scan"})
            if old is not None and old!=sig:
                changed.append(f"{t}: {old} → {sig}\nPrice ₹{p:.2f} | Score {sc}/100\nStop ₹{sl:.2f} | 2R ₹{t1:.2f}")
            state[t]=sig
        except Exception as e:
            rows.append({"Symbol":s,"Price":"—","Score":"—","Signal":f"Error: {e}","Previous":"—"})
    save_state(state)
    st.subheader("Watchlist scan")
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    if notify_changes and changed:
        ok,msg=send_telegram("🔔 SWING SIGNAL CHANGE\n\n"+"\n\n".join(changed))
        st.success("Telegram alert sent.") if ok else st.warning(msg)
    elif notify_changes:
        st.info("No signal changed, so no duplicate Telegram alert was sent.")

try:
    ticker,df,r,score,reasons,price,stop,target1,target2=analyze(symbol)
except Exception as e:
    st.error(f"Could not analyze {symbol}: {e}"); st.stop()

owned=position.startswith("Yes")
call=owned_decision(score,r) if owned else new_entry_decision(score)
st.subheader(f"{ticker} — {call}")
st.caption(f"Data fetched: {datetime.now().strftime('%d %b %Y, %I:%M:%S %p')} • Cached up to 60 seconds")

c1,c2,c3,c4,c5,c6=st.columns(6)
c1.metric("Price",f"₹{price:.2f}"); c2.metric("EMA20",f"₹{r.EMA20:.2f}")
c3.metric("EMA50",f"₹{r.EMA50:.2f}"); c4.metric("RSI",f"{r.RSI:.1f}")
c5.metric("ADX",f"{r.ADX:.1f}"); c6.metric("Setup score",f"{score}/100")

if owned and avg_price:
    pnl=(price/avg_price-1)*100
    st.metric("Unrealised P/L",f"{pnl:+.2f}%",f"Average ₹{avg_price:.2f}")

risk_per_share=max(price-stop,0.01); risk_budget=capital*risk_pct/100
qty=min(int(risk_budget//risk_per_share),int(capital//price))
st.subheader("Action plan")
a,b,c,d=st.columns(4)
a.metric("ATR-based stop",f"₹{stop:.2f}"); b.metric("2R target",f"₹{target1:.2f}")
c.metric("3R target",f"₹{target2:.2f}"); d.metric("Max quantity by risk",str(qty))

st.subheader("Why V4 gave this decision")
for icon,text in reasons: st.write(icon,text)
st.subheader("Price trend"); st.line_chart(df[["Close","EMA20","EMA50","EMA200"]].tail(250))
st.subheader("Momentum"); st.line_chart(df[["RSI"]].tail(250))
st.warning("V4 is a rule-based research aid, not a guarantee or personalized investment advice. Yahoo Finance data may be delayed. Verify live prices with your broker.")

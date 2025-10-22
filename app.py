# app.py
# Streamlit A+E Forecasting & Policy Demo
# pip install -r requirements.txt
import streamlit as st
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

# Optional Prophet
try:
    from prophet import Prophet
    HAVE_PROPHET = True
except Exception:
    HAVE_PROPHET = False

from sklearn.ensemble import GradientBoostingRegressor

st.set_page_config(page_title="A+E Forecasting & Policy", layout="wide")

# ---------------- Utils ----------------
ALIAS = {
    "A+E O&O Fast Channels": "FAST",
    "Partner O&O Fast Channels": "FAST",
    "MVPD Apps & Sites": "MVPD VOD",
    "STB VOD": "STB VOD 4+",
    "O+O": "On-Domain VOD",
}
VOD_FAST_BUCKETS = ["FAST", "MVPD VOD", "vMVPD VOD", "STB VOD 4+", "On-Domain VOD"]

def ensure_month(dt_series):
    s = pd.to_datetime(dt_series, errors="coerce")
    return s.dt.to_period("M").dt.to_timestamp()

def intercept_blend(last_actual: float, yhat: pd.Series, decay=(1.0, 0.6, 0.3)) -> pd.Series:
    if len(yhat) == 0: return yhat
    d = last_actual - yhat.iloc[0]
    out = yhat.copy()
    for i, w in enumerate(decay):
        if i < len(out): out.iloc[i] += w*d
    return out

def wape(y, yhat):
    y = np.array(y, float); yhat = np.array(yhat, float)
    return 100 * np.sum(np.abs(y - yhat)) / (np.sum(np.abs(y)) + 1e-9)

def mape(y, yhat):
    y = np.array(y, float); yhat = np.array(yhat, float)
    return 100 * np.mean(np.abs((y - yhat)/(np.maximum(np.abs(y),1e-9))))

def smape(y, yhat):
    y = np.array(y, float); yhat = np.array(yhat, float)
    denom = (np.abs(y)+np.abs(yhat))/2.0
    return 100 * np.mean(np.abs(y - yhat)/np.maximum(denom,1e-9))

def naive12_forecast(y: pd.Series, horizon: int) -> pd.Series:
    y = y.sort_index()
    out_idx = pd.date_range(y.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    if len(y) >= 12:
        template = y.iloc[-12:].to_numpy()
        yhat = np.resize(template, horizon)
    else:
        yhat = np.repeat(y.iloc[-1] if len(y) else 0.0, horizon)
    return pd.Series(yhat, index=out_idx)

def make_supervised(y: pd.Series) -> pd.DataFrame:
    df = y.to_frame("y").copy()
    df["ds"] = df.index
    df["month"]   = df["ds"].dt.month
    df["quarter"] = df["ds"].dt.quarter
    df["year"]    = df["ds"].dt.year
    for k in [1,2,3,6,12]:
        df[f"lag{k}"] = df["y"].shift(k)
    df["roll3"]  = df["y"].rolling(3).mean().shift(1)
    df["roll6"]  = df["y"].rolling(6).mean().shift(1)
    df["roll12"] = df["y"].rolling(12).mean().shift(1)
    df["diff1"]  = df["y"].diff(1).shift(1)
    df["diff12"] = df["y"].diff(12).shift(1)
    return df.dropna()

def gbm_forecast(y: pd.Series, horizon: int) -> pd.Series:
    df = make_supervised(y)
    if len(df) < 6:
        return naive12_forecast(y, horizon)
    X = df.drop(columns=["y","ds"])
    model = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.9, random_state=42
    )
    model.fit(X, df["y"])
    # recursive multi-step using generated preds
    hist = y.copy()
    out_idx = pd.date_range(hist.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    preds = []
    for i, ds_i in enumerate(out_idx, start=1):
        s = hist.sort_index()
        row = {
            "month": ds_i.month,
            "quarter": (ds_i.month-1)//3 + 1,
            "year": ds_i.year,
            "lag1": s.iloc[-1] if len(s)>=1 else np.nan,
            "lag2": s.iloc[-2] if len(s)>=2 else np.nan,
            "lag3": s.iloc[-3] if len(s)>=3 else np.nan,
            "lag6": s.iloc[-6] if len(s)>=6 else np.nan,
            "lag12": s.iloc[-12] if len(s)>=12 else np.nan,
            "roll3":  s.tail(3).mean() if len(s)>=3 else s.mean() if len(s)>0 else 0.0,
            "roll6":  s.tail(6).mean() if len(s)>=6 else s.mean() if len(s)>0 else 0.0,
            "roll12": s.tail(12).mean() if len(s)>=12 else s.mean() if len(s)>0 else 0.0,
            "diff1":  (s.iloc[-1]-s.iloc[-2]) if len(s)>=2 else 0.0,
            "diff12": (s.iloc[-1]-s.iloc[-13]) if len(s)>=13 else 0.0,
        }
        Xi = pd.DataFrame([row]).fillna(0.0)
        yhat = float(model.predict(Xi)[0])
        preds.append(yhat)
        hist.loc[ds_i] = max(yhat, 0.0)
    return pd.Series(preds, index=out_idx)

def prophet_forecast(y: pd.Series, horizon: int) -> pd.Series:
    if not HAVE_PROPHET or len(y) < 6:
        return pd.Series(dtype=float)
    df = y.reset_index().rename(columns={"index":"ds", 0:"y", y.name:"y"})
    df.columns = ["ds","y"]
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, interval_width=0.8)
    m.fit(df)
    fut = m.make_future_dataframe(periods=horizon, freq="MS")
    fc = m.predict(fut)
    fc = fc[fc["ds"] > df["ds"].max()]
    return pd.Series(fc["yhat"].values, index=fc["ds"].values)

def natural_gap(l7_hist: pd.Series, fast_hist: pd.Series,
                fast_fc: pd.Series, pct=0.05, lookahead=24):
    if l7_hist.empty or (fast_hist.empty and fast_fc.empty): return np.nan, None, None
    t_peak = l7_hist.idxmax()
    future = pd.concat([fast_hist, fast_fc]).sort_index()
    post = future[(future.index >= t_peak) & (future.index <= t_peak + pd.DateOffset(months=lookahead))]
    if post.empty or post.max() <= 0: return np.nan, t_peak, None
    thresh = pct * post.max()
    ge = post[post >= thresh]
    first = ge.index.min() if not ge.empty else None
    if first is None: return np.nan, t_peak, None
    gap = (first.to_period("M") - t_peak.to_period("M")).n
    return gap, t_peak, first

def rec_from_gap(g, floor3=True):
    if pd.isna(g): return "No FAST signal yet"
    g = int(g)
    natural = "0M (concurrent)" if g<=0 else "1M" if g<=2 else "3M" if g<=5 else "6M"
    if floor3 and natural in {"0M (concurrent)","1M"}:
        return "3M"
    return natural

# ---------------- Sidebar: Data & Controls ----------------
st.sidebar.header("Data")
data_file = st.sidebar.file_uploader("Upload monthly data (CSV)", type=["csv"])
st.sidebar.caption("Required columns (or aliases): Program, bucket/bucket_work, Month or Year+Month, Minutes Viewed.")

H = st.sidebar.number_input("Forecast horizon (months)", min_value=3, max_value=36, value=12)
use_anchor = st.sidebar.checkbox("Smooth join (intercept correction)", value=True)
threshold = st.sidebar.slider("FAST threshold (% of post-peak FAST max)", 1, 50, 5) / 100.0
floor3 = st.sidebar.checkbox("Apply 3-month floor", value=True)

if data_file is None:
    st.info("Upload your monthly dataset to begin.")
    st.stop()

df = pd.read_csv(data_file)
# Normalize columns
prog_col = next((c for c in ["program_key","Program - Series Name","Program","Partner Program","program"] if c in df.columns), None)
bucket_col = next((c for c in ["analysis_bucket","bucket_work","bucket","dist_bucket"] if c in df.columns), None)
min_col = next((c for c in ["Minutes Viewed","minutes_viewed","minutes","Total Minutes","y","MIN"] if c in df.columns), None)
time_col = next((c for c in ["month_dt","Year and Month","Display Interval","time","Month","date"] if c in df.columns), None)
if not all([prog_col, bucket_col, min_col, time_col]):
    st.error("Missing required columns. Need a program name, bucket, minutes, and a month/date column.")
    st.stop()

df[min_col] = pd.to_numeric(df[min_col], errors="coerce").fillna(0)
df["ds"] = ensure_month(df[time_col])
df["bucket"] = df[bucket_col].replace(ALIAS)
df = df.rename(columns={prog_col:"program", min_col:"y"})

programs = sorted(df["program"].dropna().unique().tolist())
buckets = sorted(df["bucket"].dropna().unique().tolist())

st.sidebar.header("Selections")
prog = st.sidebar.selectbox("Program", programs)
bucket = st.sidebar.selectbox("Bucket", buckets)
model_choice = st.sidebar.selectbox("Model", ["Naive-12","GBM (+calendar/lag)","Prophet (diagnostic)"])

sub = (df[(df["program"]==prog) & (df["bucket"]==bucket)]
         .groupby("ds", as_index=False)["y"].sum()
         .sort_values("ds"))
if sub.empty:
    st.warning("No rows for that selection.")
    st.stop()

y = sub.set_index("ds")["y"].asfreq("MS").fillna(0.0)

# ---------------- Forecast ----------------
if model_choice.startswith("Naive"):
    fc = naive12_forecast(y, H)
elif model_choice.startswith("GBM"):
    fc = gbm_forecast(y, H)
else:
    fc = prophet_forecast(y, H)
    if fc.empty:
        st.warning("Prophet not available or not enough data; falling back to Naive-12.")
        fc = naive12_forecast(y, H)

if use_anchor and len(fc)>0:
    fc = intercept_blend(y.iloc[-1], fc)

# ---------------- Policy (Live+7 vs FAST only) ----------------
if bucket == "FAST":
    # need Live+7 hist for the same program
    l7 = (df[(df["program"]==prog) & (df["bucket"]=="Live+7")]
            .groupby("ds", as_index=False)["y"].sum()
            .set_index("ds")["y"].asfreq("MS").fillna(0.0))
    g, t_peak, first = natural_gap(l7, y, fc, pct=threshold)
    rec = rec_from_gap(g, floor3=floor3)
else:
    g, t_peak, first, rec = (np.nan, None, None, "— (policy derives from Live+7 vs FAST)")

# ---------------- Charts ----------------
c1, c2 = st.columns([2,1])

with c1:
    st.subheader(f"{prog} — {bucket}: Actuals & Forecast")
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(y.index, y.values, label="Actual")
    ax.plot(fc.index, fc.values, "--", label=f"Forecast ({model_choice})")
    ax.set_xlabel("Month"); ax.set_ylabel("Minutes")
    ax.legend(); fig.tight_layout()
    st.pyplot(fig)

with c2:
    st.subheader("Policy Recommendation")
    if bucket=="FAST" and not pd.isna(g):
        st.metric("Natural Gap (months)", int(g))
        st.metric("Recommendation", rec)
        st.caption(f"Threshold: {int(threshold*100)}% of post-peak FAST max; 3M floor: {floor3}")
    else:
        st.write(rec)

# ---------------- Backtest (rolling-origin, quick) ----------------
st.markdown("---")
st.subheader("Quick Backtest (last 6 months holdout)")

TEST_H = 6
if len(y) <= TEST_H + 6:
    st.info("Not enough history to backtest (need ~12+ months).")
else:
    train = y.iloc[:-TEST_H]; test = y.iloc[-TEST_H:]
    preds = {}

    # baselines
    preds["NaiveLast"] = pd.Series(np.full(len(test), train.iloc[-1]), index=test.index)
    preds["MA3"] = pd.Series(np.full(len(test), train.tail(3).mean()), index=test.index)
    preds["SeasonalNaive"] = (test.index.to_series()
                              .apply(lambda d: y.get(d - pd.DateOffset(years=1), train.iloc[-1])))

    # chosen model
    if model_choice.startswith("Naive"):
        preds["Naive-12"] = naive12_forecast(train, TEST_H)
    elif model_choice.startswith("GBM"):
        preds["GBM"] = gbm_forecast(train, TEST_H)
    else:
        pf = prophet_forecast(train, TEST_H)
        preds["Prophet"] = pf if not pf.empty else naive12_forecast(train, TEST_H)

    rows = []
    for name, p in preds.items():
        yy, pp = test.values, p.reindex(test.index).values
        rows.append({"Model": name, "WAPE%": wape(yy, pp), "MAPE%": mape(yy, pp), "sMAPE%": smape(yy, pp)})
    bt = pd.DataFrame(rows).sort_values("WAPE%")
    st.dataframe(bt, use_container_width=True)

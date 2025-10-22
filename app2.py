# app.py (v2)
# Streamlit A+E Forecasting & Policy — Full pipeline
# pip install -r requirements.txt
import streamlit as st
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Optional Prophet
try:
    from prophet import Prophet
    HAVE_PROPHET = True
except Exception:
    HAVE_PROPHET = False

# Optional scikit-learn (GBM)
try:
    from sklearn.ensemble import GradientBoostingRegressor
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

st.set_page_config(page_title="A+E Forecasting & Policy — Full", layout="wide")

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
def rmse(y, yhat):
    y = np.array(y, float); yhat = np.array(yhat, float)
    return float(np.sqrt(np.mean((y - yhat)**2)))

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
    if not HAVE_SKLEARN:
        return naive12_forecast(y, horizon)
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

def monthly_by_bucket(df):
    out = (df.groupby(["bucket","ds"], dropna=False)["y"]
             .sum().reset_index().sort_values(["bucket","ds"]))
    # fill gaps per bucket
    rows = []
    for b, g in out.groupby("bucket"):
        idx = pd.date_range(g["ds"].min(), g["ds"].max(), freq="MS")
        gg = g.set_index("ds").reindex(idx).fillna(0.0).rename_axis("ds").reset_index()
        gg["bucket"] = b
        rows.append(gg)
    return pd.concat(rows, ignore_index=True)

def seasonal_naive(test_idx, full_series, fallback):
    return test_idx.to_series().apply(
        lambda d: full_series.get(d - pd.DateOffset(years=1), fallback)
    )

def rolling_backtests_all(df, test_h=6):
    mf = monthly_by_bucket(df)
    results = []
    for b, g in mf.groupby("bucket"):
        g = g.sort_values("ds")
        if len(g) < test_h + 6: 
            continue
        train = g.iloc[:-test_h].set_index("ds")["y"]
        test  = g.iloc[-test_h:].set_index("ds")["y"]

        cands = {
            "NaiveLast": pd.Series(np.full(len(test), train.iloc[-1]), index=test.index),
            "MA3": pd.Series(np.full(len(test), train.tail(3).mean()), index=test.index),
            "SeasonalNaive": seasonal_naive(test.index, g.set_index("ds")["y"], train.iloc[-1]),
            "Naive-12": naive12_forecast(train, test_h),
            "GBM": gbm_forecast(train, test_h),
        }
        pf = prophet_forecast(train, test_h)
        if not pf.empty: cands["Prophet"] = pf

        for name, pred in cands.items():
            yy = test.values
            pp = pred.reindex(test.index).values
            results.append({
                "bucket": b, "model": name,
                "RMSE": rmse(yy, pp),
                "WAPE_%": wape(yy, pp), "sMAPE_%": smape(yy, pp), "MAPE_%": mape(yy, pp),
                "n_points": len(test)
            })
    bt = pd.DataFrame(results).sort_values(["bucket","WAPE_%"])
    winners = (bt.sort_values("WAPE_%")
                 .groupby("bucket", as_index=False).first()[["bucket","model","WAPE_%","MAPE_%","sMAPE_%","RMSE"]])
    return bt, winners

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

def policy_sensitivity(df, thresholds=(0.03,0.05,0.07,0.10), floor3=True):
    prog_col = "program"
    out = []
    # top programs by Live+7 (fallback to total if none)
    live7 = df[df["bucket"]=="Live+7"]
    if live7.empty:
        top_programs = df.groupby(prog_col)["y"].sum().sort_values(ascending=False).head(12).index.tolist()
    else:
        top_programs = live7.groupby(prog_col)["y"].sum().sort_values(ascending=False).head(12).index.tolist()
    for p in top_programs:
        sub = df[df["program"]==p].copy()
        pm = (sub.groupby(["bucket","ds"])["y"].sum().reset_index())
        for t in thresholds:
            live7 = pm[pm["bucket"]=="Live+7"].set_index("ds")["y"].sort_index()
            fast  = pm[pm["bucket"]=="FAST"].set_index("ds")["y"].sort_index()
            # no forecast here—use history only for sensitivity
            g, _, _ = natural_gap(live7, fast, pd.Series(dtype=float), pct=t)
            natural = "0M (concurrent)" if (not pd.isna(g) and int(g)<=0) else \
                      "1M" if (not pd.isna(g) and int(g)<=2) else \
                      "3M" if (not pd.isna(g) and int(g)<=5) else \
                      ("6M" if not pd.isna(g) else "No FAST signal yet")
            rec = "3M" if floor3 and natural in {"0M (concurrent)","1M"} else natural
            out.append({"program_key": p, "threshold": f"{int(t*100)}%", 
                        "natural_gap_months": (None if pd.isna(g) else int(g)),
                        "natural_recommendation": natural, "with_3M_floor": rec})
    sens = pd.DataFrame(out)
    if sens.empty:
        return sens, sens
    sens_pivot = sens.pivot(index="program_key", columns="threshold", values="with_3M_floor")
    return sens, sens_pivot

def per_title_summary(df):
    prog_col = "program"
    rows = []
    for p, sub in df.groupby(prog_col):
        totals = sub.groupby("bucket")["y"].sum()
        live7_total = totals.get("Live+7", 0.0)
        vod_fast_total = totals.reindex(VOD_FAST_BUCKETS).fillna(0).sum()
        denom = live7_total + vod_fast_total
        l7_share = (live7_total / denom) if denom > 0 else np.nan
        # natural gap @ 5%
        pm = (sub.groupby(["bucket","ds"])["y"].sum().reset_index())
        live7 = pm[pm["bucket"]=="Live+7"].set_index("ds")["y"].sort_index()
        fast  = pm[pm["bucket"]=="FAST"].set_index("ds")["y"].sort_index()
        g, _, _ = natural_gap(live7, fast, pd.Series(dtype=float), pct=0.05)
        natural = "0M (concurrent)" if (not pd.isna(g) and int(g)<=0) else \
                  "1M" if (not pd.isna(g) and int(g)<=2) else \
                  "3M" if (not pd.isna(g) and int(g)<=5) else \
                  ("6M" if not pd.isna(g) else "No FAST signal yet")
        policy3 = "3M" if natural in {"0M (concurrent)","1M"} else natural
        rows.append({
            "program_key": p,
            "Live+7 Share": (f"{l7_share*100:,.1f}%" if not pd.isna(l7_share) else "NA"),
            "VOD/FAST Share": (f"{(1 - l7_share)*100:,.1f}%" if not pd.isna(l7_share) else "NA"),
            "Natural Gap": ("NA" if pd.isna(g) else f"{int(g)} months"),
            "Recommendation": policy3
        })
    return pd.DataFrame(rows)

def corr_heatmap(df, program=None):
    # Build a pivot: rows=month, cols=bucket, values=sum(minutes)
    sub = df[df["program"]==program] if program else df
    pivot = (sub.groupby(["ds","bucket"])["y"].sum().unstack(fill_value=0.0))
    if pivot.shape[1] < 2:
        return None, None
    corr = pivot.corr()
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(corr.values, origin="lower")
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)));  ax.set_yticklabels(corr.index)
    ax.set_title("Bucket Correlation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, corr

def df_download_button(df, filename, label="Download CSV"):
    if df is None or len(df)==0:
        st.button(label, disabled=True)
        return
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, file_name=filename, mime="text/csv")

# ---------------- Sidebar: Data & Controls ----------------
st.sidebar.header("Data")
data_file = st.sidebar.file_uploader("Upload monthly data (CSV)", type=["csv"])
st.sidebar.caption("Required columns (aliases OK): Program, bucket/bucket_work, Month or Year+Month, Minutes Viewed.")

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

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Forecast", "Backtests", "Policy sensitivity", "Per-title summary", "Correlation"])

with tab1:
    # Forecast
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
        if bucket=="FAST":
            l7 = (df[(df["program"]==prog) & (df["bucket"]=="Live+7")]
                    .groupby("ds", as_index=False)["y"].sum()
                    .set_index("ds")["y"].asfreq("MS").fillna(0.0))
            g, t_peak, first = natural_gap(l7, y, fc, pct=threshold)
            rec = rec_from_gap(g, floor3=floor3)
            if not pd.isna(g):
                st.metric("Natural Gap (months)", int(g))
            st.metric("Recommendation", rec)
            st.caption(f"Threshold: {int(threshold*100)}% of post-peak FAST max; 3M floor: {floor3}")
        else:
            st.write("— (policy derives from Live+7 vs FAST)")

    # export forecast
    fc_df = pd.DataFrame({"ds": fc.index, "yhat": fc.values})
    df_download_button(fc_df, f"forecast_{prog}_{bucket}.csv", "Download forecast CSV")

with tab2:
    st.subheader("Rolling backtests by bucket (last 6 months holdout)")
    bt_all, winners = rolling_backtests_all(df)
    c1, c2 = st.columns([2,1])
    with c1:
        st.dataframe(bt_all, use_container_width=True)
        df_download_button(bt_all, "backtest_metrics.csv", "Download backtest metrics")
    with c2:
        st.markdown("**Per-bucket winners (lowest WAPE)**")
        st.dataframe(winners, use_container_width=True)

with tab3:
    st.subheader("Policy threshold sensitivity (3/5/7/10%) — with optional 3M floor")
    sens, sens_wide = policy_sensitivity(df, thresholds=(0.03,0.05,0.07,0.10), floor3=True)
    if sens_wide is None or len(sens)==0:
        st.info("Not enough Live+7/FAST data to compute sensitivity.")
    else:
        st.dataframe(sens_wide, use_container_width=True)
        df_download_button(sens, "policy_sensitivity_long.csv", "Download (long)")
        df_download_button(sens_wide.reset_index(), "policy_sensitivity_wide.csv", "Download (wide)")

with tab4:
    st.subheader("Per-title summary")
    summary = per_title_summary(df)
    st.dataframe(summary, use_container_width=True)
    df_download_button(summary, "per_title_summary.csv", "Download per-title summary")

with tab5:
    st.subheader("Correlation heatmap")
    fig, corr = corr_heatmap(df, program=prog)
    if fig is None:
        st.info("Need ≥2 buckets with data to draw correlation.")
    else:
        st.pyplot(fig)
        df_download_button(corr.reset_index(), "bucket_correlation.csv", "Download correlation")

st.caption("Tip: Each git push to main will auto-redeploy on Streamlit Cloud.")

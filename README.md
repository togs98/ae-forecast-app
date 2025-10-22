# A+E Forecasting & Policy (Streamlit)

Interactive dashboard to forecast A+E viewing by Program/Bucket and derive policy recommendations (natural gap with 3‑month floor).

**Models:** Naive‑12, GBM(+calendar/lag), Prophet (diagnostic).  
**Features:** join smoothing (intercept correction), quick 6‑month backtest (WAPE/MAPE/sMAPE), threshold sensitivity.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## CSV schema (aliases allowed)
- Program: `program_key` / `Program - Series Name` / `Program` / `Partner Program` / `program`
- Bucket: `analysis_bucket` / `bucket_work` / `bucket` / `dist_bucket`
- Month:  `month_dt` / `Year and Month` / `Display Interval` / `time` / `Month` / `date`
- Minutes: `Minutes Viewed` / `minutes_viewed` / `minutes` / `Total Minutes` / `y` / `MIN`

## Sample
See `sample_data.csv` for format.

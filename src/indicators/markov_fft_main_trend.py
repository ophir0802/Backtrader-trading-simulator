"""
Detect market regimes in the S&P 500 using FFT-smoothed trend returns +
2-regime Markov model.

Output
------
CSV  markov_fft_main_trend.csv  with column:
    regime_signal_fftfeat  ∈ {+1, 0, -1}
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# ---------------------------------------------------------------------------
# DB connection
# ---------------------------------------------------------------------------
env_path = Path(__file__).resolve().parents[2] / "Algo_env" / ".env"
load_dotenv(env_path)
engine = create_engine(os.getenv("DATABASE_URL"))

query = """
SELECT date, close, market_trend
FROM sp500_index
WHERE date BETWEEN '2014-01-01' AND '2024-12-31'
ORDER BY date ASC;
"""
df = pd.read_sql(query, engine, parse_dates=["date"]).set_index("date")

# ---------------------------------------------------------------------------
# FFT low-pass filter → trend series
# ---------------------------------------------------------------------------
def low_pass_fft(series: pd.Series, cutoff: float = 0.05) -> pd.Series:
    n = len(series)
    fft_vals = np.fft.fft(series.values)
    freqs = np.fft.fftfreq(n, d=1.0)
    fft_vals[np.abs(freqs) > cutoff] = 0
    return pd.Series(np.fft.ifft(fft_vals).real, index=series.index, name="trend")

df["trend"] = low_pass_fft(df["close"])
df["trend_returns"] = df["trend"].pct_change()

# ---------------------------------------------------------------------------
# 2-regime Markov on trend returns
# ---------------------------------------------------------------------------
work = df.dropna(subset=["trend_returns", "market_trend"]).copy()

model = MarkovRegression(
    work["trend_returns"], exog=work[["market_trend"]], k_regimes=2,
    switching_variance=True
)
res = model.fit()

work[["prob_0", "prob_1"]] = res.smoothed_marginal_probabilities

bull_regime = 0
if work.loc[work["prob_1"] > 0.5, "trend_returns"].mean() > \
   work.loc[work["prob_0"] > 0.5, "trend_returns"].mean():
    bull_regime = 1
bear_regime = 1 - bull_regime

work["dom_prob"]    = work[["prob_0", "prob_1"]].max(axis=1)
work["dom_regime"]  = np.where(work["prob_0"] > work["prob_1"], 0, 1)

signal = np.zeros(len(work), dtype=int)
conf   = 0.80
signal[(work["dom_regime"] == bull_regime) & (work["dom_prob"] >= conf)]  =  1
signal[(work["dom_regime"] == bear_regime) & (work["dom_prob"] >= conf)]  = -1
work["regime_signal_fftfeat"] = signal

# ---------------------------------------------------------------------------
# Merge back & save
# ---------------------------------------------------------------------------
df = df.join(work[["regime_signal_fftfeat"]])
out_path = Path("markov_fft_main_trend.csv")
df.to_csv(out_path)

print("✅ CSV written to", out_path.resolve())
print(df[["close", "regime_signal_fftfeat"]].tail(8))

import pandas as pd
from indicators.bb_indicator import calculate_bb_trend
from indicators.ma200_indicator import calculate_ma200
from indicators.macd_indicator import calculate_macd
from indicators.td_indicator import calculate_td_trend

# Load market dataset
df = pd.read_csv("../data/full_dataset_with_split.csv")

# Compute indicators
df['bb_trend'] = calculate_bb_trend(df)
df['ma200'] = calculate_ma200(df['close'])
df['macd'], df['macd_signal'] = calculate_macd(df['close'])
df['td_trend'] = calculate_td_trend(df['close'], lambda_threshold=0.15)

# Save results
df.to_csv("../data/market_with_indicators.csv", index=False)

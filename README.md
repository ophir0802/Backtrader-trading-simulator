# AlgoTrading Project


## Description

This project assumes historical market data is stored in a PostgreSQL database (accessible via the `DATABASE_URL` environment variable). The dataset consists of daily OHLCV for S&P 500 index and constituent stocks. The repository provides:

- A set of reusable indicators and market‑regime utilities
- A strategy layer designed for quick iteration (plug‑and‑play parameters and components)
- A simple, reproducible way to evaluate strategies across chosen time windows (train/validation/test)

The goal is to make it easy to implement new ideas rapidly, test them over realistic market periods, and assess whether a strategy is robust rather than overfitted.

## Assumptions

- `src/` – source code
- `data/` – stock data files
-  `env` holds postGres url namer "DATABASE_URL"

## Installation

```bash
python -m venv Algo_env
.\Algo_env\Scripts\activate
pip install -r requirements.txt



generate_sector_indicators.py
This script calculates Relative Strength indicators for sector-level analysis relative to the S&P 500.
It generates the following metrics and exports them to data/sector_indicators_train.csv:

Classic RS: RS_21, RS_55, RS_123

Anchored RS: Relative strength since a specific anchor date (e.g., YTD)

RS Ribbon: RSribbon_8, RSribbon_21, RSribbon_42 based on EMA momentum

Momentum Slope: slope_21, slope_42 to detect acceleration/deceleration in RS

Output is used in Phase 2: Sector vs Market Trend Analysis
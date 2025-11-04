import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Assuming all user imports are correct and working
from DBintegration.models import DailyStockData, SP500Index
from DBintegration.db_utils import model_to_dataframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import matplotlib.style as style
def calc_beta_grouped(df_all, window=250):
    """
   computes beta_index for each stock
    """
    out = []
    for symbol, df in df_all.groupby('symbol'):
        beta_df = calculate_beta_index(df.copy(), window)
        beta_df['symbol'] = symbol          
        out.append(beta_df)

    return pd.concat(out).sort_index()
def calculate_beta_index(df: pd.DataFrame, window: int = 250):
    """
    Calculates ‘beta_pos’, ‘beta_neg’ and ‘beta_index’ on a rolling window.
    Works whether the DataFrame comes in with a ‘date’ column or with a
    DatetimeIndex.
    """
    # --- 0. Normalise the date axis ----------------------------------------
    if 'date' in df.columns:                     # ❶ “date” is a column
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    elif isinstance(df.index, pd.DatetimeIndex): # ❷ “date” is already index
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    else:
        raise ValueError(
            "DataFrame must have a 'date' column or a DatetimeIndex."
        )

    # keep just the price-series columns for the rolling calculations
    data = df[['return_daily', 'sp_return']]

    # --- 1. rolling computations -------------------------------------------
    results = {'beta_pos': [], 'beta_neg': [],
               'num_up':   [], 'num_down': []}

    for end in range(len(data)):
        start = max(0, end + 1 - window)
        win   = data.iloc[start:end + 1]

        up   = win[ win['sp_return']  > 0]
        down = win[ win['sp_return'] <= 0]

        # helpers
        def beta(block):
            if len(block) > 1 and block['sp_return'].var():
                return block['return_daily'].cov(block['sp_return']) \
                       / block['sp_return'].var()
            return np.nan

        results['beta_pos'].append(beta(up))
        results['beta_neg'].append(beta(down))
        results['num_up'].append(len(up))
        results['num_down'].append(len(down))

    # --- 2. stitch the results back onto the original frame ----------------
    res = pd.DataFrame(results, index=data.index)
    df = pd.concat([df, res], axis=1)

    df['beta_index'] = (
        res['num_up']   * res['beta_pos']
        - res['num_down'] * res['beta_neg']
    ) / window
    return df
################




def split_dataframe_by_dates(
    df: pd.DataFrame,
    d1: str = '1.1.2013',
    d2: str = '1.1.2024',
    d3: str = '1.1.2022',
    d4: str = '1.1.2024'
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into three parts based on specified date ranges.

    The function assumes the input DataFrame has a column named 'date'.
    The date ranges are inclusive. Note that data on the boundary dates
    (d2 and d3) will be included in two DataFrames.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'date' column.
        d1 (str): The start date for the first DataFrame ('D.M.YYYY').
        d2 (str): The end date for the first and start for the second DataFrame.
        d3 (str): The end date for the second and start for the third DataFrame.
        d4 (str): The end date for the third DataFrame.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the
        three resulting DataFrames (df1, df2, df3).
    """
    # Create a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()
    if 'date' not in df_copy.columns:
        df_copy = df_copy.reset_index()  # מחזיר את האינדקס כעמודה 'date'
    # --- Step 1: Convert all date inputs to datetime objects ---
    # Convert the string dates from the function arguments
    # dayfirst=True correctly interprets formats like '1.1.2013'
    date1 = pd.to_datetime(d1, dayfirst=True)
    date2 = pd.to_datetime(d2, dayfirst=True)
    date3 = pd.to_datetime(d3, dayfirst=True)
    date4 = pd.to_datetime(d4, dayfirst=True)

    # Ensure the DataFrame's 'date' column is also in datetime format
    try:
        df_copy['date'] = pd.to_datetime(df_copy['date'])
    except Exception as e:
        print(f"Error converting the 'date' column to datetime: {e}")
        # Return empty dataframes if conversion fails
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


    # --- Step 2: Filter the DataFrame for each period ---
    # DataFrame 1: from d1 to d2 (inclusive)
    mask1 = (df_copy['date'] >= date1) & (df_copy['date'] <= date2)
    df1 = df_copy.loc[mask1]

    # DataFrame 2: from d2 to d3 (inclusive)
    mask2 = (df_copy['date'] >= date2) & (df_copy['date'] <= date3)
    df2 = df_copy.loc[mask2]

    # DataFrame 3: from d3 to d4 (inclusive)
    mask3 = (df_copy['date'] >= date3) & (df_copy['date'] <= date4)
    df3 = df_copy.loc[mask3]

    return df1, df2, df3
import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DBintegration.db_utils import model_to_dataframe
from DBintegration.db_utils import update_data
from DBintegration.models import DailyStockData
from sqlalchemy.orm import sessionmaker
from DBintegration.models import Base
from pathlib import Path
import pandas as pd
from datetime import datetime
"""functions List:
count_symbols(df) -----> count unique symbols.
"""
def count_symbols(dataframe: pd.DataFrame) -> int:
    """
    Counts the number of unique symbols in a pandas DataFrame.

    Args:
        dataframe (pd.DataFrame): The input DataFrame, expected to have a 'symbol' column.

    Returns:
        int: The number of unique symbols.
    """
    if 'symbol' in dataframe.columns:
        return dataframe['symbol'].nunique()
    else:
        print("Warning: 'symbol' column not found in the DataFrame.")
        return 0

# Example usage of the function with your 'df'



# def split_dataframe_by_dates(
#     df: pd.DataFrame,
#     d1: str = '1.1.2013',
#     d2: str = '1.1.2021',
#     d3: str = '1.1.2022',
#     d4: str = '1.1.2024'
# ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     """
#     Splits a DataFrame into three parts based on specified date ranges.

#     The function assumes the input DataFrame has a column named 'date'.
#     The date ranges are inclusive. Note that data on the boundary dates
#     (d2 and d3) will be included in two DataFrames.

#     Args:
#         df (pd.DataFrame): The input DataFrame with a 'date' column.
#         d1 (str): The start date for the first DataFrame ('D.M.YYYY').
#         d2 (str): The end date for the first and start for the second DataFrame.
#         d3 (str): The end date for the second and start for the third DataFrame.
#         d4 (str): The end date for the third DataFrame.

#     Returns:
#         tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the
#         three resulting DataFrames (df1, df2, df3).
#     """
#     # Create a copy to avoid SettingWithCopyWarning
#     df_copy = df.copy()
#     if 'date' not in df_copy.columns:
#         df_copy = df_copy.reset_index()  # return index as a 'date' column
#     # --- Step 1: Convert all date inputs to datetime objects ---
#     # Convert the string dates from the function arguments
#     # dayfirst=True correctly interprets formats like '1.1.2013'
#     date1 = pd.to_datetime(d1, dayfirst=True)
#     date2 = pd.to_datetime(d2, dayfirst=True)
#     date3 = pd.to_datetime(d3, dayfirst=True)
#     date4 = pd.to_datetime(d4, dayfirst=True)

#     # Ensure the DataFrame's 'date' column is also in datetime format
#     try:
#         df_copy['date'] = pd.to_datetime(df_copy['date'])
#     except Exception as e:
#         print(f"Error converting the 'date' column to datetime: {e}")
#         # Return empty dataframes if conversion fails
#         return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


#     # --- Step 2: Filter the DataFrame for each period ---
#     # DataFrame 1: from d1 to d2 (inclusive)
#     mask1 = (df_copy['date'] >= date1) & (df_copy['date'] <= date2)
#     df1 = df_copy.loc[mask1]

#     # DataFrame 2: from d2 to d3 (inclusive)
#     mask2 = (df_copy['date'] >= date2) & (df_copy['date'] <= date3)
#     df2 = df_copy.loc[mask2]

#     # DataFrame 3: from d3 to d4 (inclusive)
#     mask3 = (df_copy['date'] >= date3) & (df_copy['date'] <= date4)
#     df3 = df_copy.loc[mask3]

#     return df1, df2, df3

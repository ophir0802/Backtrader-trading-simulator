
#%%
import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DBintegration.models import DailyStockData
from DBintegration.models import SP500Index# or any other model you want to use
from DBintegration.models import *
from pathlib import Path
from Indicators.df_utils import count_symbols
from DBintegration.db_utils import *
from sqlalchemy.orm import sessionmaker
import backtrader as bt
import pandas as pd
import os
from ophir.btIndicators import *
from ophir.strategy import *
from ophir.utils import *
from ophir.utils import split_dataframe_by_dates
import matplotlib
import numpy as np
from pandas.tseries.offsets import BDay
import backtrader as bt
import pandas as pd
import numpy as np
import backtrader.analyzers as btanalyzers
from datetime import datetime, timedelta
import os

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# model_to_dataframe is an external function that downloads data from cloud
# - First call (DailyStockData) returns: DataFrame with columns [date, symbol, open, high, low, close, volume]
#   where stocks may have different time intervals
# - Second call (SP500Index) returns: DataFrame with columns [date, open, high, low, close, volume] 
#   (no 'symbol' column)

# Your actual model classes for the external data function
# DailyStockData and SP500Index are passed to model_to_dataframe()

def split_dataframe_by_dates(
    df: pd.DataFrame,
    d1: str = '1.1.2013',
    d2: str = '1.1.2019',
    d3: str = '1.1.2022',
    d4: str = '1.1.2024'
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into three parts based on specified date ranges.
    """
    df_copy = df.copy()
    if 'date' not in df_copy.columns:
        df_copy = df_copy.reset_index()
    
    # Convert dates
    date1 = pd.to_datetime(d1, dayfirst=True)
    date2 = pd.to_datetime(d2, dayfirst=True)
    date3 = pd.to_datetime(d3, dayfirst=True)
    date4 = pd.to_datetime(d4, dayfirst=True)
    
    try:
        df_copy['date'] = pd.to_datetime(df_copy['date'])
    except Exception as e:
        print(f"Error converting the 'date' column to datetime: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Filter dataframes
    mask1 = (df_copy['date'] >= date1) & (df_copy['date'] <= date2)
    df1 = df_copy.loc[mask1]
    
    mask2 = (df_copy['date'] >= date2) & (df_copy['date'] <= date3)
    df2 = df_copy.loc[mask2]
    
    mask3 = (df_copy['date'] >= date3) & (df_copy['date'] <= date4)
    df3 = df_copy.loc[mask3]
    
    return df1, df2, df3

def clear_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only OHLCV columns plus symbol and date.
    """
    keep = ['symbol', 'open', 'high', 'low', 'close', 'volume']
    
    if 'date' in df.columns:
        keep = ['date'] + keep
    
    keep = [c for c in keep if c in df.columns]
    return df.loc[:, keep].copy()

# =============================================================================
# INDICATORS
# =============================================================================

class BetaStrength(bt.Indicator):
    """
    BetaStrength – relative strength indicator for the last N days.
    """
    lines = ('score', 'p_up', 'p_down', 'diff_up', 'diff_down')
    params = dict(
        period=30,
        eps=1e-9
    )

    def __init__(self):
        super().__init__()
        
        # Ensure we have market data
        if len(self.datas) < 2:
            raise ValueError("BetaStrength requires both stock and market data feeds")
        
        # Daily returns
        stock_ret = (self.data0 / self.data0(-1)) - 1.0
        market_ret = (self.data1 / self.data1(-1)) - 1.0
        
        # Masks for up/down market days
        up_mask = market_ret > 0
        down_mask = market_ret <= 0
        
        period = self.p.period
        eps = self.p.eps
        
        # Count of days
        n_up = bt.ind.SumN(up_mask, period=period) + eps
        n_down = bt.ind.SumN(down_mask, period=period) + eps
        
        # Fraction of days
        p_up = n_up / float(period)
        p_down = n_down / float(period)
        
        # Return sums
        sum_stock_up = bt.ind.SumN(stock_ret * up_mask, period=period)
        sum_market_up = bt.ind.SumN(market_ret * up_mask, period=period)
        
        sum_stock_down = bt.ind.SumN(stock_ret * down_mask, period=period)
        sum_market_down = bt.ind.SumN(market_ret * down_mask, period=period)
        
        # Return deltas
        diff_up = sum_stock_up - sum_market_up
        diff_down = sum_market_down - sum_stock_down
        
        # Final score
        score = p_up * diff_up - p_down * diff_down
        
        # Assign output lines
        self.lines.score = score
        self.lines.p_up = p_up
        self.lines.p_down = p_down
        self.lines.diff_up = diff_up
        self.lines.diff_down = diff_down
        
        self.addminperiod(period)

class BetaIndex(bt.Indicator):
    """
    Beta Index indicator with separate up and down market calculations.
    """
    lines = ('beta_up_long', 'beta_down_long', 'beta_up_short', 'beta_down_short', 
             'beta_index', 'beta_index_recent')
    params = dict(
        period=360,
        short_window=20,
        func=lambda b_up, b_down, n_up, n_down, p:
             (n_up * b_up - n_down**3 * b_down) / p if p > 0 else 0.0,
    )

    def __init__(self):
        super().__init__()
        
        if len(self.datas) < 2:
            raise ValueError("BetaIndex requires both stock and market data feeds")
        
        stock_ret = (self.data0 / self.data0(-1)) - 1
        market_ret = (self.data1 / self.data1(-1)) - 1
        
        beta_up_long, beta_down_long, index_long = self._calculate_for_period(
            stock_ret, market_ret, self.p.period)
        beta_up_short, beta_down_short, index_short = self._calculate_for_period(
            stock_ret, market_ret, self.p.short_window)
        
        self.lines.beta_up_long = beta_up_long
        self.lines.beta_down_long = beta_down_long
        self.lines.beta_up_short = beta_up_short
        self.lines.beta_down_short = beta_down_short
        self.lines.beta_index = index_long
        self.lines.beta_index_recent = index_short
        
        self.addminperiod(max(self.p.period, self.p.short_window))

    def _calculate_for_period(self, stock_ret, market_ret, period):
        """Helper method to calculate beta for a given period."""
        market_up_days = market_ret > 0
        market_down_days = market_ret <= 0
        
        # Beta up calculation
        n_up = bt.ind.SumN(market_up_days, period=period) + 1e-9
        
        mean_stock_ret_up = bt.ind.SumN(stock_ret * market_up_days, period=period) / n_up
        mean_market_ret_up = bt.ind.SumN(market_ret * market_up_days, period=period) / n_up
        
        mean_prod_up = bt.ind.SumN(stock_ret * market_ret * market_up_days, period=period) / n_up
        cov_up = mean_prod_up - (mean_stock_ret_up * mean_market_ret_up)
        
        mean_sq_market_ret_up = bt.ind.SumN((market_ret * market_ret) * market_up_days, period=period) / n_up
        var_up = mean_sq_market_ret_up - (mean_market_ret_up * mean_market_ret_up)
        
        beta_up = cov_up / (var_up + 1e-9)
        
        # Beta down calculation
        n_down = bt.ind.SumN(market_down_days, period=period) + 1e-9
        
        mean_stock_ret_down = bt.ind.SumN(stock_ret * market_down_days, period=period) / n_down
        mean_market_ret_down = bt.ind.SumN(market_ret * market_down_days, period=period) / n_down
        
        mean_prod_down = bt.ind.SumN(stock_ret * market_ret * market_down_days, period=period) / n_down
        cov_down = mean_prod_down - (mean_stock_ret_down * mean_market_ret_down)
        
        mean_sq_market_ret_down = bt.ind.SumN((market_ret * market_ret) * market_down_days, period=period) / n_down
        var_down = mean_sq_market_ret_down - (mean_market_ret_down * mean_market_ret_down)
        
        beta_down = cov_down / (var_down + 1e-9)
        
        # Final index
        final_index = self.p.func(beta_up, beta_down, n_up, n_down, period)
        
        return beta_up, beta_down, final_index

class ScoreIndicator(bt.Indicator):
    """Simple indicator that passes through a score from data feed."""
    lines = ('score',)
    params = ()

    def __init__(self):
        if hasattr(self.data, 'score'):
            self.lines.score = self.data.score
        else:
            # Fallback to zero if no score available
            self.lines.score = 0
        self.addminperiod(1)

# =============================================================================
# CUSTOM DATA FEED
# =============================================================================

class SP500IndexWithScore(bt.feeds.PandasData):
    """Custom data feed for S&P 500 index that includes score line."""
    lines = ('score',)
    params = (
        ('score', -1),  # -1 means auto-detect column position
    )
    
    def __init__(self):
        super().__init__()
        # Validate that score column exists
        if hasattr(self.p.dataname, 'columns') and 'score' not in self.p.dataname.columns:
            print("Warning: 'score' column not found in data. Setting score to 0.")

# =============================================================================
# STRATEGIES
# =============================================================================

class CombinedStrategy(bt.Strategy):
    """Multi-stock strategy with proper error handling and data validation."""
    
    params = dict(
        strength_period=30,
        beta_long_period=360,
        beta_short_period=20,
        printlog=True,
        long_threshold_entry=0.2,
        short_threshold_entry=-0.1,  # Fixed: should be negative
        long_exit_score=0.1,         # Lower than entry for hysteresis
        short_exit_score=-0.05,      # Higher than entry for hysteresis
        trail_perc=0.08,
        min_position_value=1000,     # Minimum position size in dollars
    )

    def __init__(self):
        # Validate data feeds
        if len(self.datas) < 2:
            raise ValueError("Strategy requires at least one stock feed and one market feed")
        
        # Separate market feed from stock feeds
        self.market = self.datas[-1]
        self.stocks = self.datas[:-1]
        
        # Create indicators for market
        self.market_scorer = ScoreIndicator(self.market)
        
        # Create indicators for each stock
        self.strengths = []
        self.beta_indices = []
        
        for d in self.stocks:
            try:
                strength_ind = BetaStrength(d, self.market, period=self.p.strength_period)
                self.strengths.append(strength_ind)
                
                beta_ind = BetaIndex(d, self.market, period=self.p.beta_long_period, 
                                   short_window=self.p.beta_short_period)
                self.beta_indices.append(beta_ind)
            except Exception as e:
                print(f"Error creating indicators for {getattr(d, '_name', 'unknown')}: {e}")
                # Add None placeholders to maintain index alignment
                self.strengths.append(None)
                self.beta_indices.append(None)
        
        # Track exit orders
        self.exit_orders = {}
        self.entry_prices = {}

    def log(self, txt, dt=None):
        """Logging function."""
        if self.p.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()} - {txt}')

    def notify_order(self, order):
        """Handle order notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                self.log(f'BUY EXECUTED [{order.data._name}], Price: {order.executed.price:.2f}, '
                        f'Size: {order.executed.size:.2f}')
                self.entry_prices[order.data] = order.executed.price
            elif order.issell():
                self.log(f'SELL EXECUTED [{order.data._name}], Price: {order.executed.price:.2f}, '
                        f'Size: {order.executed.size:.2f}')
                if order.data not in self.entry_prices:
                    self.entry_prices[order.data] = order.executed.price

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order {order.status} for {order.data._name}')
            # Remove failed exit orders from tracking
            if order.data in self.exit_orders:
                try:
                    if self.exit_orders[order.data].ref == order.ref:
                        del self.exit_orders[order.data]
                except (AttributeError, KeyError):
                    pass

    def notify_trade(self, trade):
        """Handle trade notifications."""
        if trade.isclosed:
            self.log(f'TRADE CLOSED [{trade.data._name}] PnL: {trade.pnlcomm:.2f}')
            # Clean up tracking dictionaries
            if trade.data in self.exit_orders:
                del self.exit_orders[trade.data]
            if trade.data in self.entry_prices:
                del self.entry_prices[trade.data]

    def next(self):
        """Main strategy logic."""
        try:
            market_score = self.market_scorer.score[0]
            
            for i, d in enumerate(self.stocks):
                # Skip if indicators failed to create
                if self.strengths[i] is None or self.beta_indices[i] is None:
                    continue
                
                position = self.getposition(d)
                
                # Get indicator values with error checking
                try:
                    beta_strength_score = self.strengths[i].score[0]
                    if pd.isna(beta_strength_score):
                        continue
                except (IndexError, AttributeError):
                    continue
                
                # Skip if price data is invalid
                if pd.isna(d.close[0]) or d.close[0] <= 0:
                    continue
                
                # Entry logic
                if position.size == 0:
                    self._handle_entry(d, beta_strength_score, market_score)
                else:
                    # Exit logic
                    self._handle_exit(d, position, beta_strength_score)
                    
        except Exception as e:
            self.log(f"Error in strategy execution: {e}")

    def _handle_entry(self, data, beta_strength_score, market_score):
        """Handle entry logic for a stock."""
        try:
            # Long entry
            if (beta_strength_score > self.p.long_threshold_entry and market_score >= 1):
                position_pct = 0.01 if market_score == 1 else 0.02
                target_value = self.broker.getvalue() * position_pct
                
                if target_value >= self.p.min_position_value:
                    size = target_value / data.close[0]
                    if size >= 1:  # Ensure at least 1 share
                        self.buy(data=data, size=int(size))
                        self.log(f'BUY SIGNAL [{data._name}] Score: {beta_strength_score:.3f}, '
                               f'Market: {market_score}, Size: {int(size)}')
            
            # Short entry
            elif (beta_strength_score < self.p.short_threshold_entry and market_score <= -1):
                position_pct = 0.01 if market_score == -1 else 0.02
                target_value = self.broker.getvalue() * position_pct
                
                if target_value >= self.p.min_position_value:
                    size = target_value / data.close[0]
                    if size >= 1:
                        self.sell(data=data, size=int(size))
                        self.log(f'SELL SIGNAL [{data._name}] Score: {beta_strength_score:.3f}, '
                               f'Market: {market_score}, Size: {int(size)}')
                        
        except Exception as e:
            self.log(f"Error in entry logic for {data._name}: {e}")

    def _handle_exit(self, data, position, beta_strength_score):
        """Handle exit logic for a stock."""
        try:
            close_reason = None
            
            # Check for score-based exit
            if position.size > 0 and beta_strength_score < self.p.long_exit_score:
                close_reason = f'Long Exit Score ({beta_strength_score:.3f})'
            elif position.size < 0 and beta_strength_score > self.p.short_exit_score:
                close_reason = f'Short Exit Score ({beta_strength_score:.3f})'
            
            if close_reason:
                self.log(f'CLOSE SIGNAL [{data._name}]: {close_reason}')
                self.close(data=data)
            else:
                # Set trailing stop if not already set
                if data not in self.exit_orders:
                    try:
                        if position.size > 0:
                            order = self.sell(data=data, exectype=bt.Order.StopTrail, 
                                            trailpercent=self.p.trail_perc)
                        else:
                            order = self.buy(data=data, exectype=bt.Order.StopTrail, 
                                           trailpercent=self.p.trail_perc)
                        
                        if order:
                            self.exit_orders[data] = order
                            
                    except Exception as e:
                        self.log(f"Error setting trailing stop for {data._name}: {e}")
                        
        except Exception as e:
            self.log(f"Error in exit logic for {data._name}: {e}")

    def stop(self):
        """Called when strategy stops."""
        if self.p.printlog:
            self.log(f'Final Portfolio Value: {self.broker.getvalue():.2f}')

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def run_backtest():
    """Main function to run the backtest."""
    
    print("=== Starting Backtest ===")
    
    # Load data using your external function
    print("Loading stock data from cloud...")
    df_main = model_to_dataframe(DailyStockData)  # Returns DF with 'symbol' column
    print(f"Loaded {len(df_main)} stock data rows")
    
    print("Loading S&P 500 data from cloud...")
    df_sp500 = model_to_dataframe(SP500Index)     # Returns DF without 'symbol' column
    print(f"Loaded {len(df_sp500)} S&P 500 data rows")
    
    # Prepare data
    print("Splitting data by date ranges...")
    df_train, _, _ = split_dataframe_by_dates(df_main)
    df_sp500_train, _, _ = split_dataframe_by_dates(df_sp500)
    
    # Clean data
    df_train = clear_columns(df_train)
    df_sp500_train = clear_columns(df_sp500_train)
    
    print(f"Training data: {len(df_train)} stock rows, {len(df_sp500_train)} S&P 500 rows")
    
    # Load market regime scores from CSV
    print("Loading market regime data...")
    try:
        scores_df = pd.read_csv(
            'markov_fft_com_old_data.csv',
            parse_dates=['date'],
            index_col='date'
        ).rename(columns={'regime_signal_combined': 'score'})
        print(f"Loaded {len(scores_df)} market regime scores")
    except FileNotFoundError:
        print("Warning: markov_fft_com_old_data.csv not found. Creating mock regime scores...")
        # Create mock regime scores as fallback
        date_range = pd.date_range(
            start=df_sp500_train['date'].min() if 'date' in df_sp500_train.columns else df_sp500_train.index.min(),
            end=df_sp500_train['date'].max() if 'date' in df_sp500_train.columns else df_sp500_train.index.max(),
            freq='D'
        )
        scores_data = []
        for date in date_range:
            score = np.random.choice([-2, -1, 0, 1, 2], p=[0.1, 0.2, 0.4, 0.2, 0.1])
            scores_data.append({'date': date, 'score': score})
        scores_df = pd.DataFrame(scores_data).set_index('date')
    
    # Add scores to SP500 data
    df_sp500_train = df_sp500_train.join(scores_df[['score']], how='left')
    df_sp500_train['score'] = df_sp500_train['score'].fillna(0).astype(int)
    
    # Setup Cerebro
    cerebro = bt.Cerebro()
    
    # Add stock data feeds
    if 'symbol' not in df_train.columns:
        raise ValueError("'symbol' column not found in stock data from model_to_dataframe(DailyStockData)")
    
    stock_symbols = df_train['symbol'].unique()
    print(f"Found {len(stock_symbols)} unique stock symbols")
    
    # Limit number of stocks for performance (remove this limit if you want all stocks)
    max_stocks = 20
    if len(stock_symbols) > max_stocks:
        print(f"Limiting to first {max_stocks} stocks for performance")
        stock_symbols = stock_symbols[:max_stocks]
    
    added_stocks = 0
    for symbol in stock_symbols:
        stock_df = df_train[df_train['symbol'] == symbol].copy()
        
        # Skip stocks with insufficient data
        min_data_points = 252  # About 1 year of trading days
        if len(stock_df) < min_data_points:
            print(f"Skipping {symbol} - insufficient data ({len(stock_df)} rows, need {min_data_points})")
            continue
        
        # Prepare stock dataframe
        if 'date' in stock_df.columns:
            stock_df = stock_df.set_index('date')
        
        # Remove symbol column for backtrader (it expects OHLCV only)
        stock_df = stock_df.drop('symbol', axis=1, errors='ignore')
        stock_df.sort_index(inplace=True)
        
        # Check for data quality issues
        if stock_df.isnull().any().any():
            print(f"Warning: {symbol} has missing values, forward filling...")
            stock_df = stock_df.fillna(method='ffill').fillna(method='bfill')
        
        # Add to cerebro
        try:
            feed = bt.feeds.PandasData(dataname=stock_df)
            cerebro.adddata(feed, name=symbol)
            added_stocks += 1
        except Exception as e:
            print(f"Error adding {symbol} to cerebro: {e}")
            continue
    
    print(f"Successfully added {added_stocks} stock feeds to cerebro")
    
    if added_stocks == 0:
        raise ValueError("No stock feeds were successfully added to cerebro")
    
    # Add market feed (must be last)
    if 'date' in df_sp500_train.columns:
        df_sp500_train = df_sp500_train.set_index('date')
    
    df_sp500_train.sort_index(inplace=True)
    
    # Check S&P 500 data quality
    if df_sp500_train.isnull().any().any():
        print("Warning: S&P 500 data has missing values, forward filling...")
        df_sp500_train = df_sp500_train.fillna(method='ffill').fillna(method='bfill')
    
    market_feed = SP500IndexWithScore(dataname=df_sp500_train)
    cerebro.adddata(market_feed, name='SP500_Market')
    
    print("Market feed added successfully")
    
    # Add strategy and configure broker
    cerebro.addstrategy(CombinedStrategy)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission
    
    # Add analyzers
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
    
    # Run backtest
    print("\n" + "="*50)
    print("STARTING BACKTEST EXECUTION")
    print("="*50)
    
    starting_value = cerebro.broker.getvalue()
    print(f'Starting Portfolio Value: ${starting_value:,.2f}')
    
    try:
        results = cerebro.run()
        
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - starting_value) / starting_value * 100
        
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f'Final Portfolio Value: ${final_value:,.2f}')
        print(f'Total Return: {total_return:.2f}%')
        
        # Print analyzer results
        strat = results[0]
        
        # Sharpe ratio
        try:
            sharpe_analysis = strat.analyzers.sharpe.get_analysis()
            sharpe_ratio = sharpe_analysis.get('sharperatio', None)
            if sharpe_ratio:
                print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
            else:
                print("Sharpe Ratio: Not available")
        except Exception as e:
            print(f"Error calculating Sharpe ratio: {e}")
        
        # Drawdown
        try:
            dd_analysis = strat.analyzers.drawdown.get_analysis()
            max_dd = dd_analysis.max.drawdown
            print(f"Max Drawdown: {max_dd:.2f}%")
        except Exception as e:
            print(f"Error calculating drawdown: {e}")
        
        # Trade statistics
        try:
            trade_analysis = strat.analyzers.trades.get_analysis()
            if hasattr(trade_analysis, 'total') and trade_analysis.total.total > 0:
                total_trades = trade_analysis.total.total
                won_trades = getattr(trade_analysis.won, 'total', 0) if hasattr(trade_analysis, 'won') else 0
                lost_trades = getattr(trade_analysis.lost, 'total', 0) if hasattr(trade_analysis, 'lost') else 0
                win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
                
                print(f"Total Trades: {total_trades}")
                print(f"Winning Trades: {won_trades}")
                print(f"Losing Trades: {lost_trades}")
                print(f"Win Rate: {win_rate:.1f}%")
                
                if hasattr(trade_analysis.won, 'pnl') and hasattr(trade_analysis.lost, 'pnl'):
                    avg_win = trade_analysis.won.pnl.average if won_trades > 0 else 0
                    avg_loss = trade_analysis.lost.pnl.average if lost_trades > 0 else 0
                    print(f"Average Win: ${avg_win:.2f}")
                    print(f"Average Loss: ${avg_loss:.2f}")
                    if avg_loss != 0:
                        profit_factor = abs(avg_win / avg_loss)
                        print(f"Profit Factor: {profit_factor:.2f}")
            else:
                print("No trades executed during backtest")
        except Exception as e:
            print(f"Error calculating trade statistics: {e}")
        
        print("="*50)
        
        return cerebro, results
        
    except Exception as e:
        print(f"Error during backtest execution: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    cerebro, results = run_backtest()
    
    if cerebro and results:
        print("\nBacktest completed successfully!")
        # Uncomment the next line to show plots (requires matplotlib)
        # cerebro.plot(style='candlestick')
    else:
        print("Backtest failed!")




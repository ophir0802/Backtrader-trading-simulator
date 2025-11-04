import backtrader as bt
import numpy as np
import pandas as pd

# premises: data0 = stockFeed, data1 = marketFeed
class BetaIndex(bt.Indicator):
    """
    An OPTIMIZED version of the BetaIndex indicator.
    It calculates all values in a single pass to avoid redundant data fetching.
    """
    
    lines = ('beta_up_long', 'beta_down_long','beta_up_short','beta_down_short', 'beta_index', 'beta_index_recent',)
    params = ( #dictionary
        ('period', 360),
        ('short_window', 20),
        ('func', lambda b_up, b_down, n_up, n_down, p: 
                 (n_up * b_up - n_down**3 * b_down) / p if p > 0 else 0.0),
    )
    def __init__(self):
        super().__init__()
        
        stock_ret  = (self.data0 / self.data0(-1)) - 1
        market_ret = (self.data1 / self.data1(-1)) - 1
        
        beta_up_long, beta_down_long, index_long = self._calculate_for_period(stock_ret, market_ret, self.p.period)
        beta_up_short, beta_down_short, index_short = self._calculate_for_period(stock_ret, market_ret, self.p.short_window)
        # 4. set results to indicators output lines
        self.lines.beta_up_long = beta_up_long
        self.lines.beta_down_long = beta_down_long
        self.lines.beta_up_short = beta_up_short
        self.lines.beta_down_short = beta_down_short
        self.lines.beta_index = index_long
        self.lines.beta_index_recent = index_short # short term index
        self.addminperiod(max(self.p.period, self.p.short_window))
        
    def _calculate_for_period(self, stock_ret, market_ret, period):
        """
        Helper method to define the calculation chain for a given period.
        This method doesn't calculate values, it builds the indicator graph.
        """
        #boolean masks 
        market_up_days = market_ret > 0
        market_down_days = market_ret <= 0
        #beta_up
        n_up = bt.ind.SumN(market_up_days, period=period) + 1e-9 # 
        # E[X] = Sum(X) / N
        mean_stock_ret_up = bt.ind.SumN(stock_ret * market_up_days, period=period) / n_up
        mean_market_ret_up = bt.ind.SumN(market_ret * market_up_days, period=period) / n_up
        # Cov(X,Y) = E[XY] - E[X]E[Y]
        mean_prod_up = bt.ind.SumN(stock_ret * market_ret * market_up_days, period=period) / n_up
        cov_up = mean_prod_up - (mean_stock_ret_up * mean_market_ret_up)
        # Var(X) = E[X^2] - (E[X])^2
        mean_sq_market_ret_up = bt.ind.SumN((market_ret*market_ret) * market_up_days, period=period) / n_up
        var_up = mean_sq_market_ret_up - (mean_market_ret_up*mean_market_ret_up)
        
        beta_up = cov_up / (var_up + 1e-9) # avoid deviding by 0: 

        #beta_down
        n_down = bt.ind.SumN(market_down_days, period=period) + 1e-9
        
        mean_stock_ret_down = bt.ind.SumN(stock_ret * market_down_days, period=period) / n_down
        mean_market_ret_down = bt.ind.SumN(market_ret * market_down_days, period=period) / n_down
        
        mean_prod_down = bt.ind.SumN(stock_ret * market_ret * market_down_days, period=period) / n_down
        cov_down = mean_prod_down - (mean_stock_ret_down * mean_market_ret_down)
        
        mean_sq_market_ret_down = bt.ind.SumN((market_ret* market_ret) * market_down_days, period=period) / n_down
        var_down = mean_sq_market_ret_down - (mean_market_ret_down* mean_market_ret_down)
        
        beta_down = cov_down / (var_down + 1e-9)

        # beta_index
        final_index = self.p.func(beta_up, beta_down, n_up, n_down, period)
        
        return beta_up, beta_down, final_index
    
    
class ScoreIndicator(bt.Indicator):
    lines = ('score',)          # indicator outputs one line
    params = ()                 # no params needed

    def __init__(self):
        # wire the feed’s score line straight through
        self.lines.score = self.data.score
        self.addminperiod(1)    # indicator is valid from bar 0
        


    


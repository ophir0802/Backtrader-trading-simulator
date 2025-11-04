import backtrader as bt
import pandas as pd
import datetime
from ophir.utils import *
from ophir.btIndicators import BetaIndex
from ophir.btIndicators import ScoreIndicator
import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class RelativeBIDX(bt.Strategy):
    params = (
        ('beta_period', 30),
        ('beta_short_window', 20),
        ('end_dates', None),
        ('high_percentage', 0.1),
        ('low_percentage', 0.12),
    )

    def __init__(self):
        print("--- Initializing Multi-Stock Strategy ---")
        self.market_feed   = self.datas[-1]       # index feed
        self.stock_feeds   = self.datas[:-1]      # equities
        self.market_score = ScoreIndicator(self.market_feed)
        self.beta_indicators = {}
        self.orders_by_stock = {}
        self.sell_reasons     = {}                #  <- REASON DICT (key = symbol)

        for d in self.stock_feeds:
            s = d._name
            self.beta_indicators[s] = BetaIndex(
                d, self.market_feed,
                period=self.p.beta_period,
                short_window=self.p.beta_short_window
            )
            self.orders_by_stock[s] = None
        print("--- Strategy Initialization Complete ---")

    def next(self):
        #rank stocks:
        ranked = []
        for d in self.stock_feeds:
            if len(d) > self.p.beta_period:
                s   = d._name
                val = self.beta_indicators[s].beta_index_recent[0]
                if not pd.isna(val):
                    ranked.append({'symbol': s, 'beta': val, 'data': d})

        ranked.sort(key=lambda x: x['beta'], reverse=True)
        n   = len(ranked)
        top = int(n * self.p.high_percentage)
        mid = int(n * self.p.low_percentage)
        to_buy  = {r['symbol'] for r in ranked[:top]}
        to_hold = {r['symbol'] for r in ranked[:mid]}
        sp_score_today = self.market_score[0]
        for d in self.stock_feeds:
            s = d._name
            if self.orders_by_stock.get(s): #order available
                continue
            if pd.isna(d.close[0]) or len(d) <= self.p.beta_period: #data allready "over"
                continue
            pos = self.getposition(d)
            if pos: #already in position -> sell checks
                end_dt = self.p.end_dates.get(s)
                today  = d.datetime.date(0)
                # forced last-day exit
                if end_dt and today == end_dt.date():
                    o = self.close(data=d)
                    self.sell_reasons[s] = 'forced'
                    continue
                # logic exit: 
                if s not in to_hold:
                    o = self.close(data=d)
                    self.orders_by_stock[s] = o
                    self.sell_reasons[s] = 'logic'
                continue
            beta_recent = self.beta_indicators[s].beta_index_recent[0]
            if (s in to_buy) and (sp_score_today in (1, 2)):                                            # long
                pct = 2 if sp_score_today == 2 else 1
                size = int(self.broker.getvalue() * pct / 100 / d.close[0])
                if size < 1: continue
                buy_o = self.buy(data=d, size=size); self.orders_by_stock[s] = buy_o
                self.sell(data=d, parent=buy_o, exectype=bt.Order.StopTrail, trailpercent=0.08); self.sell_reasons[s] = 'TrailingStop'
            elif (s in to_buy) and (sp_score_today in (-1, -2)):                                         # short
                pct = 1 if sp_score_today == -2 else 0.5
                size = int(self.broker.getvalue() * pct / 100 / d.close[0])
                if size < 1: continue
                sell_o = self.sell(data=d, size=size); self.orders_by_stock[s] = sell_o
                self.buy(data=d, parent=sell_o, exectype=bt.Order.StopTrail, trailpercent=0.08); self.sell_reasons[s] = 'TrailingStopShort'
    # ------------------------------------------------------------------
    def notify_order(self, order):
        s = order.data._name
        if order.status in [order.Submitted, order.Accepted]:
            return
        self.orders_by_stock[s] = None       # free slot

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        s = trade.data._name
        reason = self.sell_reasons.pop(s, 'no reason recorded')
        odt = trade.open_datetime().strftime('%Y-%m-%d %H:%M:%S')
        cdt = trade.close_datetime().strftime('%Y-%m-%d %H:%M:%S')
        print(f'--- TRADE CLOSED for {s} ---')
        print(f'OPENED: {odt}  CLOSED: {cdt}  REASON: {reason}')
        print(f'PnL: Gross {trade.pnl:.2f}  Net {trade.pnlcomm:.2f}')
        print('-----------------------------------------')

    def stop(self):
        
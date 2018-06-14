"""
#
# Simple, Long-Short, Trend Following Strategy with 30/60/90 day lookback periods.
# +1/3 weight for bullish trend, -1/3 weight for bearish trend within the respective lookback period.
#
"""

import protos_edge as pe

"""
#
# Settings:
#
"""
# Data
start = '2016-12-31'
end = '2018-05-25'
tickers=['bitcoin','bitcoin-cash','ethereum','litecoin','ripple']


# Backtest Parameters
init_balance=100
rebalance_period=1
spread=0.002
fees=0

# Strategy Parameters
max_lookback=90
weights=[(1/3),(1/3),(1/3)]
normalize_vol=True
long_only=False
short_only=False


if __name__ == '__main__':
    
    data = pe.Data_Selected(start,end,frequency=1,tickers=tickers)
    prices = data.load_data(table="close")
    prices = data.clean_data(prices)
    
    portfolio = pe.Daily_Portfolio(init_balance)
    
    backtest = pe.Daily_Backtest(rebalance_period, spread, fees)
    
    strategy = pe.Trend_Following(max_lookback, weights,normalize_vol,
                                  long_only, short_only)
    
    performance = backtest.run_backtest(prices,portfolio,strategy)
    
    statistics = backtest.collect_statistics(performance)
    

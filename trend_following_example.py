import base_classes as bc

"""
#
# Settings:
#
"""
# Data
start = '2016-12-31'
end = '2018-05-25'

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
    
    data = bc.Trend_Data(start,end,frequency=1, tickers=[])
    prices = data.load_data()
    
    prices = data.clean_data(prices)
        
    portfolio = bc.Daily_Portfolio(init_balance)
    
    backtest = bc.Daily_Backtest(rebalance_period, spread, fees)
    
    strategy = bc.Trend_Following(max_lookback, weights,normalize_vol,
                                  long_only, short_only)
    
    performance = backtest.run_backtest(prices,portfolio,strategy)
    
    
    backtest.collect_statistics(performance)

    

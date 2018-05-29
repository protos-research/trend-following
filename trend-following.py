from datetime import datetime
import pandas as pd
import numpy as np
import gspread as gs
from gspread_dataframe import get_as_dataframe
from oauth2client.service_account import ServiceAccountCredentials
import math
import matplotlib.pyplot as plt

##############################################################################
def loadGS():
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('keyfile.json', scope)
    client = gs.authorize(creds)

    ss_price = client.open("Backtesting Data").worksheet('Price')
    
    df_price = get_as_dataframe(ss_price)
    
    df_price_tickers = df_price[['Date','bitcoin','bitcoin-cash','ethereum','litecoin','ripple']]
    
    return df_price_tickers

def cleanData(df_price, start_date, end_date):
    price_filter = (df_price['Date'] >= start_date) & (df_price['Date'] <= end_date)
   
    price = df_price[price_filter]
    
    price.set_index('Date', inplace=True)
    
    price.index = pd.to_datetime(price.index)

    price.fillna('NaN')
    
    price = price.apply(pd.to_numeric, errors='coerce')
    
    return price

def Strategy(prices,max_lookback, weights, 
             normalize_vol, long_only, short_only):
    
    last_row = prices.shape[0]-1
    
    # 3 lookback periods including max-lookback
    lb1 = int(max_lookback/3)
    lb2 = int(2*max_lookback/3)
    
    
    
    # As soon as singals are fully calculated
    if(last_row >= max_lookback):
                 
        l_mask_1 = prices.iloc[last_row,:]>=prices.iloc[last_row-lb1,:]
        l_mask_1 = l_mask_1*weights[0]
        l_mask_1.mask(l_mask_1==0,other=(-weights[0]), inplace=True)
                
        l_mask_2 = prices.iloc[last_row,:]>=prices.iloc[last_row-lb2,:]
        l_mask_2 = l_mask_2*weights[1]
        l_mask_2.mask(l_mask_2==0,other=(-weights[1]), inplace=True)

        l_mask_3 = prices.iloc[last_row,:]>=prices.iloc[last_row-max_lookback,:]
        l_mask_3 = l_mask_3*weights[2]
        l_mask_3.mask(l_mask_3==False,other=(-weights[2]), inplace=True)

        #### Short Masks
        
        s_mask_1 = prices.iloc[last_row,:]<prices.iloc[last_row-lb1,:]
        s_mask_1 = s_mask_1*(-weights[0])
        s_mask_1.mask(s_mask_1==0,other=(weights[0]), inplace=True)
        
        s_mask_2 = prices.iloc[last_row,:]<prices.iloc[last_row-lb2,:]
        s_mask_2 = s_mask_2*(-weights[1])
        s_mask_2.mask(s_mask_2==0,other=(weights[1]), inplace=True)

        s_mask_3 = prices.iloc[last_row,:]<prices.iloc[last_row-max_lookback,:]
        s_mask_3 = s_mask_3*(-weights[2])
        s_mask_3.mask(s_mask_3==0,other=(weights[2]), inplace=True)

        for index, i in enumerate(prices.iloc[last_row-max_lookback,:]):
            if(math.isnan(i)): 
                l_mask_1[index] = np.NAN
                l_mask_2[index] = np.NAN
                l_mask_3[index] = np.NAN
                s_mask_1[index] = np.NAN
                s_mask_2[index] = np.NAN
                s_mask_3[index] = np.NAN
                      
        # Long-Only or Long-Short   
        if(long_only):
            mask = l_mask_1 + l_mask_2 + l_mask_3
            mask.mask(mask < 0, other=0, inplace=True)
        elif(short_only):
            mask = s_mask_1 +s_mask_2 + s_mask_3
            mask.mask(mask > 0, other=0, inplace=True)
        else:
            mask = l_mask_1 + l_mask_2 + l_mask_3 


 
    else:
        mask = prices.iloc[last_row,:]
        mask = (mask*0).fillna(0)
    
    ewma_ann = [0,0,0,0,0]
    # Normalize for Volatility as well:
    vol_lb = 90
    if(last_row+1 >= vol_lb):
        if(normalize_vol):
            returns = prices.pct_change().replace(np.inf, np.nan)
            ewma0 = returns.iloc[:vol_lb,:].std(axis=0)**2
            if(last_row>0):
                for i in range(vol_lb,last_row+1):#returns.shape[0]-vol_lb .... vol_lb+i
                    ewma0 = 0.94*ewma0.squeeze() + 0.06*((returns.iloc[i,:].rename())**2).T.squeeze()
                
            ewma_ann = np.sqrt(ewma0)*np.sqrt(365)
            ewma = ewma_ann.sum()/ewma_ann
            ewma_norm = ewma/ewma.sum()
            mask = mask*ewma_norm
            
    
    
    # Normalize the mask - max single position risk = 1/(nr of tickers active)
    if(normalize_vol): mask_norm = mask  
    else: mask_norm = mask/mask.count()
    #Replace NaN with 0 
    mask_norm = mask_norm.fillna(0)
    
    return mask_norm
###############################################################################
"""
#
#
#
# MAIN CODE STARTS HERE!
#
#
#
"""    

"""
#
# Load Data
#
"""    
start_date = '2016-12-31'
end_date = '2018-05-25'
data = loadGS()
prices_d = cleanData(data,start_date, end_date)
# prices_d: daily prices for the 5 tickers: bitcoin, ethereum, ...
prices_d = prices_d.fillna(np.NAN)

"""
#
### Settings for the Trend Strategy
#
# Enter your settings here
#
"""  
# max_lookback in days for the trend signal (gets divided in 3 pieces)
max_lookback = 90
#max_lookback_short = 30
# Weights used for the strategy
s_weights = [(1/3), (1/3), (1/3)]
# Long-short or long-only
long_only=False
# If short_only=True, then set long_only=False !!!
short_only= False
# Normalize weights with rolling volatility?
normalize_vol = True
# Rebalance portfolio every x days
rebalance_period = 8
# Half-Spread: bps above / below market price
spread = 0.01
# Transaction Fees in [USD] - Change balance down below when using this!
transaction_fees = 0

#weights = Strategy(prices_d.iloc[0:100,:], max_lookback,s_weights, normalize_vol, long_only, short_only)
#print(weights)



"""
#
# Initializing the portfolio for the backtest
#
"""  
# Portfolio quantities: how many quantities of each ticker are in the Portfolio
portfolio_q = []
# Portfolio balance: this is where we calculate sharpe, returns, etc from
portfolio_balance = []
# Initial Balance allocated to the strategy (number can be chosen arbitrarily)
# also acts as a counter throughout the for loop
balance = 100

trading = []

tweights = []

ewma_b = []
ewma_r = []



"""
#
# Start the backtest
#
# Loop through all days where we have price data
"""

for i in range(1,prices_d.shape[0]): 
    if(i == 1): print(prices_d.iloc[i,:].name)
    if(i == prices_d.shape[0]-1): print(prices_d.iloc[i,:].name)

    ### What happened to our portfolio during the timestep?
    # Add returns to balance, if we had a non-empty portfolio allocation
        
    if(i > 1):
        # update current balance only when portfolio has allocations
        # for the first days, there are no trend-signals == no allocation
        if(abs(portfolio_q[len(portfolio_q)-1]).sum() != 0):
            # add returns of each ticker for this timestep
            # quantity of ticker * price_delta (quantity is neg for short pos)
            balance += (portfolio_q[len(portfolio_q)-1]*(prices_d.iloc[i-1,:]-prices_d.iloc[i-2,:])).sum()
        
    ### How should we react to new prices?
        
    # get new weights
    weights = Strategy(prices_d.iloc[0:i,:], max_lookback, s_weights, normalize_vol, long_only, short_only)

        
    # calculate target allocation
    t_alloc = weights*balance
    tweights.append(weights)
    # calculate target quantity allocation        
    q_alloc = (t_alloc/prices_d.iloc[i-1,:]).fillna(0)
    # change division by zero (no prices available, etc.) to 0
    q_alloc = q_alloc.replace(np.inf, 0)
        
    # change quantity allocation of our portfolio 
    # i%7 == 0 every seven days! On all other days, portfolio allocation stays unchanged
    if(i == 1): portfolio_q.append(q_alloc)
    if(i%rebalance_period == 0):
        # Append new allocation to portfolio every x=rebalancing_period days                      
        portfolio_q.append(q_alloc)
        # Subtract transaction fees and market spread
        trades = portfolio_q[len(portfolio_q)-1]-portfolio_q[len(portfolio_q)-2]
        trading.append(trades)
        balance -= (abs(trading[len(trading)-1])*prices_d.iloc[i-1,:]*spread).sum()
        #balance -= transaction_fees
            
            
        
    # add current days new balance (calculated above) as soon as signals start to come in (i > lookback-period for trend signals)
    if(i >= max_lookback):
        portfolio_balance.append(balance)


        
        
"""
# 
# Show some plots and statistics
#
"""      

# transform to pd.DataFrame() for plotts, etc.


portfolio_q = pd.DataFrame(portfolio_q, 
                           columns=prices_d.columns)

#portfolio_q.iloc[:,0].plot()

portfolio_balance = pd.DataFrame(portfolio_balance)


returns = portfolio_balance.pct_change()

sharpe = returns.mean()/(returns.std())*np.sqrt(365)

mean = returns.mean()*365
vol = returns.std()*np.sqrt(365)

gain_to_pain = returns.sum()/abs(returns[returns < 0].sum())

print("Expected Returns: " + str(mean.values))
print("Volatility: " + str(vol.values))
print("-------------------------------------")
print("Sharpe Ratio: " + str(sharpe.values))
print("Gain to Pain: " + str(gain_to_pain.values))
print("-------------------------------------")
#print(returns.describe())
print("-------------------------------------")
print("Final Balance: " + str(portfolio_balance.iloc[portfolio_balance.shape[0]-1].values))
portfolio_balance.plot()
#returns.hist(bins=100)


"""
#
# Create CSVs to check for errors
#
"""
portfolio_balance.to_csv("backtest.csv")

prices_d.to_csv("prices.csv")

pd.DataFrame(tweights).to_csv("weights.csv")

portfolio_q.to_csv("positions.csv")

#Exercise 0
def github() -> str:
    """
    returns the github link for this assignment
    """

    return "https://github.com/<user>/<repo>/blob/main/HW4.py"

#Exercise 1
import pandas as pd
from datetime import datetime, date, time, timedelta
import numpy as np

def load_data() -> pd.DataFrame:
    """
    returns the Tesla Stock as a data frame with indices being the dates
    """
    

    return pd.read_csv('https://lukashager.netlify.app/econ-481/data/TSLA.csv', index_col = 'Date', parse_dates = True)

#Exercise 2

import matplotlib.pyplot as plt

def plot_close(df: pd.DataFrame, start: str = '2010-06-29', end: str = '2024-04-15') -> None:
    """
    Shows a plot of the Tesla Stock between two specified dates.
    """
    data = df.loc[start:end]['Close']
    fig, ax = plt.subplots()
    data.plot(ax=ax, color="black")

    ax.set_xlim([start,end])
    ax.set_title(f"Tesla Stock Closing Prices from {start} to {end}")
    fig

#Exercise 3
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

def autoregress(df: pd.DataFrame) -> float:
    """
    returns the t-values of the OLS regression on the change of stock price of one day and the change of stock price the following day (Note: the days must be consecutive. Consecutive business days like from Friday to Monday were not considered)
    """
    Change_From_Yesterday = (df[['Close']] - df.shift(periods=1, freq='D')[['Close']]).dropna()
    D1 = Change_From_Yesterday.shift(periods=1, freq='D')
    D2 = Change_From_Yesterday.shift(periods=-1, freq='D')
    D1[['Close']]=0
    D2[['Close']]=0
    DeltaX_1 = np.transpose(np.array([(Change_From_Yesterday-D1).dropna()['Close'].tolist()]))
    DeltaX = np.transpose(np.array([(Change_From_Yesterday-D2).dropna()['Close'].tolist()]))
    model = sm.OLS(DeltaX_1,DeltaX)
    results = model.fit(cov_type='HC1')
    return results.tvalues[0]

#Exercise 4

def autoregress_logit(df: pd.DataFrame) -> float:
    """
    returns the t value of the logistic regression on whether the change in stock price 
    """
    yester = (df[['Close']] - df.shift(periods=1, freq='D')[['Close']]).dropna()
    d1 = yester.shift(periods=1, freq='D')
    d2 = yester.shift(periods=-1, freq='D')
    d1[['Close']]=0
    d2[['Close']]=0
    DX_1 = np.transpose(np.array([(yester-d1).dropna()['Close'].tolist()]))
    DX = np.transpose(np.array([(yester-d2).dropna()['Close'].tolist()]))

    Growth = ((DX_1>0).astype(int))
    model = sm.Logit(Growth,DX)
    results = model.fit()

    return results.tvalues[0]

#Exercise 5

def plot_delta(df: pd.DataFrame) -> None:
    """
    Some docstrings.
    """
    data = (df[['Close']] - df.shift(periods=1, freq='D')[['Close']]).dropna()
    fig, ax = plt.subplots()
    data.plot(ax=ax, color="black")

    ax.set_title(f"Tesla Stock Changes in Closing Price")
    ax.get_legend().remove()
    
    fig
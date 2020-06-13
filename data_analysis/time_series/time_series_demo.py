import pandas as pd
import numpy as np
import plotly.graph_objects as go


path = '/home/xiejh/PycharmProjects/Btc/huobittrade/binance.csv'
df = pd.read_csv(path,delimiter=',')
df.columns = ['datetime','open','high','low','close','volumn']
fig = go.Figure(data=[go.Candlestick(x=df['datetime'],open=df['open'], high=df['high'],
                      low=df['low'], close=df['close'])])


fig.show()
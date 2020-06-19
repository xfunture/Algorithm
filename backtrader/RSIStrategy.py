import pandas as pd
import backtrader as bt
import datetime
import os
import sys


class RsiStrategy(bt.Strategy):
    params = (('short',30),('long',70))
    def __init__(self):
        self.rsi = bt.indicators.RSI_SMA(self.data.close,period=21)
        print('rsi short:',self.params.short)
        print('rsi long:',self.params.long)

    def next(self):
        if not self.position:
            if self.rsi < self.params.short:
                self.buy(size=100)
            else:
                if self.rsi > self.params.long:
                    self.sell(size=100)


if __name__ == "__main__":
    cerebro = bt.Cerebro()
    cerebro.addstrategy(RsiStrategy)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.002)
    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    datapath = os.path.join(modpath,'')
    # data = bt.feeds.YahooFinanceCSVData(
    #     dataname = datapath,
    #     fromdate = datetime.datetime(2000,1,1),
    #     todate = datetime.datetime(2000,12,31),
    #     reverse = False
    # )
    data = bt.feeds.YahooFinanceData(dataname='AAPL',
                                     fromdate=datetime.datetime(2019,1,1),
                                     todate = datetime.datetime(2019,12,31),
                                     reverse = False)
    cerebro.adddata(data)
    print('Starting Portfolio Value:{}'.format(cerebro.broker.getvalue()))

    # cerebro.plot(style='candlestick')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio,_name='SharpeRatio')
    cerebro.addanalyzer(bt.analyzers.DrawDown,_name='DW')
    results = cerebro.run()
    strat = results[0]
    print('Final Portfolio value:{}'.format(cerebro.broker.getvalue()))
    print(strat.analyzers.SharpeRatio.get_analysis())
    print(strat.analyzers.DW.get_analysis())
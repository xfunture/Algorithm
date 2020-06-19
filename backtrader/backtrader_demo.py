from datetime import datetime
import backtrader as bt




class SmaCross(bt.Strategy):
    # list of parameters which are configurable for the strategy
    params = dict(
        pfast=10,  # period for the fast moving average
        pslow=30   # period for the slow moving average
    )

    def __init__(self):
        sma1 = bt.ind.SMA(period=self.p.pfast)  # fast moving average
        sma2 = bt.ind.SMA(period=self.p.pslow)  # slow moving average
        self.crossover = bt.ind.CrossOver(sma1, sma2)  # crossover signal

    def next(self):
        if not self.position:  # not in the market
            if self.crossover > 0:  # if fast crosses slow to the upside
                self.buy()  # enter long

        elif self.crossover < 0:  # in the market & cross to the downside
            self.close()  # close long position


cerebro = bt.Cerebro()  # create a "Cerebro" engine instance

# Create a data feed
#MSFt
#ORCL
data = bt.feeds.YahooFinanceData(dataname='orcl',
                                 fromdate=datetime(1995, 1, 1),
                                 todate=datetime(2014, 12, 31))


print(type(data))

cerebro.adddata(data)  # Add the data feed

cerebro.addstrategy(SmaCross)  # Add the trading strategy
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='SharpeRatio')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='DW')
results = cerebro.run()  # run it all
cerebro.plot()  # and plot it with a single command
strat = results[0]
print(strat.analyzers.SharpeRatio.get_analysis())
print(strat.analyzers.DW.get_analysis())
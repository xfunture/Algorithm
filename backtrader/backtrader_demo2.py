import datetime
import backtrader as bt
import backtrader.feeds as btfeed
import backtrader.indicators as btind
from csvfeeds import MyHLOC

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



class MyStrategy(bt.Strategy):
    def __init__(self):
        sma1 = btind.SimpleMovingAverage(self.data)
        ema1 = btind.ExponentialMovingAverage(self.data)

        close_over_sma = self.data.close > sma1
        close_over_ema = self.data.close > ema1
        sma_ema_diff = sma1 - ema1
        self.buy_sig = bt.And(close_over_sma,close_over_ema,sma_ema_diff>0)
        close_under_sma = self.data.close < sma1
        close_under_ema = self.data.close < ema1
        self.sell_sig = bt.And(close_under_ema,close_under_sma,(sma1 - ema1)<0)
    def next(self):
        if not self.position:
            if self.buy_sig:
                self.buy()
            else:
                self.close()
        else:
            if self.sell_sig:
                self.short()
            else:
                self.close()


class TestStrategy(bt.Strategy):
    def log(self,txt,dt=None):
        """
        Logging function fot this strategy
        """
        dt = dt or self.datas[0].datetime.date(0)
        print('%s,%s' % (dt.isoformat(),txt))

    def __init__(self):
        self.dataclose = self.datas[0].close
        #To keep track of pending orders
        self.order = None

    def notify_order(self,order):
        if order.status in [order.Submitted,order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED,%.2F' % order.executed.price)
            elif order.issell():
                self.log('SELL EXECUTED,%.2f' % order.executed.price)
            self.bar_executed = len(self)
        elif order.status in [order.Canceled,order.Margin,order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        #Write down:no pending order
        self.order = None

    def next(self):
        """
        Simply log the closing price of the series from the reference
        """
        self.log('Close,%.2f' % self.dataclose[0])

        #Check if an order is pending,if yes,cannot send a 2nd one
        if self.order:
            return
        #Check if we are in the market
        if not self.position:
            if self.dataclose[0] < self.dataclose[-1]:
                #current close less than previous close
                if self.dataclose[-1] < self.dataclose[-2]:
                    self.log('BUY CREATE,%.2f' % self.dataclose[0])

                    self.order = self.buy()
        else:
            if len(self) >= (self.bar_executed + 5):
                self.log('SELL CREATE,%.2F' % self.dataclose[0])
                self.order = self.sell()


def main():
    cerebro = bt.Cerebro()
    data = MyHLOC(dataname = '/home/rick/data/BTCUSDT_20190101_20191231_minute.csv')
    print(data)
    cerebro.adddata(data)
    cerebro.broker.setcash(10000)
    cerebro.addstrategy(TestStrategy)
    # cerebro.addstrategy(MyStrategy)
    cerebro.run()
    print('Final:{:.2f}'.format(cerebro.broker.getvalue()))
    # cerebro.plot()


if __name__ == "__main__":
    main()
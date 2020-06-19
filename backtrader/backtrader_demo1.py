from __future__ import (absolute_import,division,print_function,unicode_literals)
import backtrader as bt
import sys
import os.path
import datetime


#Create a Strategy
class TestStrategy(bt.Strategy):
    def log(self,txt,dt = None):
        """
        Logging function for this strategy
        """
        dt = dt or self.datas[0].datetime.date(0)
        print("%s, %s" % (dt.isoformat(),txt))

    def __init__(self):
        #keep a reference to the "close" line in the data[0] dataserices
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None

        self.sma = bt.ind.SMA(period = 20)

    # def next(self):
    #     #Simply log the closing price of the series from the reference
    #     self.log('Close,current price %.2f  last price %.2f' % (self.dataclose[0],self.dataclose[-1]))
    #     if self.dataclose[0] < self.dataclose[-1]:
    #         #current close less than previous close
    #
    #         if self.dataclose[-1] < self.dataclose[-2]:
    #             # previous close less than the previous close
    #             # BUY,BUY,BUY(with all possible default parameters)
    #             self.log('BUY CREATE,%2f' % self.dataclose[0])
    #             self.buy()

    def notify_order(self, order):
        if order.status in [order.Submitted,order.Accepted]:
            # Buy / Sell order submitted/accepted to/by broker - Nothing to do
            return
        #check if an order has been completed
        #Attention:broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('SELL EXECUTED, %.2f' % order.executed.price)
            elif order.issell():
                self.log('SELL EXECUTED, %2F' % order.executed.price)
            self.bar_executed = len(self)

        elif order.status in [order.Canceled,order.Margin,order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None

    def next(self):
        #Simply log closing price of the series from the reference
        self.log('Close, %.2f ' % self.dataclose[0])

        #check if an order is pending  ... if yes, we cannot send a 2nd one
        if not self.position:
            if self.dataclose[0] < self.dataclose[-1]:
                #current close less than previous close
                if self.dataclose[-1] < self.dataclose[-2]:
                    #previous close  less than the previous close
                    #Buy,Buy,Buy (with the default parameters)
                    self.log('BUY CREATE ,%.2F' % self.dataclose[0])
                    #keep track of thre created order to avoid a 2nd order
                    self.order = self.buy()
                else:
                #     Already in the market ... we might sell
                #     if len(self) >= (self.bar_executed + 5 + 5):
                    #     SELL,SELL,SELL
                        # self.log('SELL CREATE, %.2f' % self.dataclose[0])
                        # keep track of the created order to avoid a 2nd order
                    self.order = self.sell()


if __name__ == "__main__":
    cerebro = bt.Cerebro()
    cerebro.addstrategy(TestStrategy)
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
    cerebro.run()
    print('Final Portfolio value:{}'.format(cerebro.broker.getvalue()))
    cerebro.plot(style='candlestick')
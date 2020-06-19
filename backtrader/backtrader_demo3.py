from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import datetime  # For datetime objects
import os.path
import sys

import backtrader as bt
import backtrader.indicators as btind
from csvfeeds import MyHLOC


# Create a Strategy
# class TestStrategy(bt.Strategy):
#     params = (
#         ('exitbars',5),
#         ('maperiod',15)
#     )
#
#     def log(self,txt,dt=None):
#         """
#         Logging function for this strategy
#         """
#         dt = dt or self.datas[0].datetime.date(0)
#         print('%s, %s' % (dt.isoformat(),txt))
#
#     def __init__(self):
#         #keep a reference to the 'close' line in data[0] dataseries
#         self.dataclose = self.datas[0].close
#         #To keep track of pending orders
#         self.order = None
#         self.buyprice = None
#         self.buycomm = None
#
#         #Add a MovingAverageSimple indicator
#         self.sma = bt.indicators.SimpleMovingAverage(
#             self.datas[0],period = self.params.maperiod
#         )
#
#     def notify_order(self,order):
#         if order.status in [order.Submitted,order.Accepted]:
#             #Buy/Sell order submitted/accepted to/by broker - Nothing to do
#             return
#         #Check if an order has been completed
#         #Attention:broker could reject order if not enough cash
#         if order.status in [order.Completed]:
#             if order.isbuy():
#                 self.log("BUY EXECUTED,Price:%.2f Cost:%.2f Comm:%.2f" %
#                          (order.executed.price,
#                          order.executed.value,
#                          order.executed.comm))
#                 self.buyprice = order.executed.price
#                 self.buycomm = order.executed.comm
#             else:
#                 self.log('SELL EXECUTED,Price: %.2f,Cose:%.2f Comm:%.2f' %
#                          (order.executed.price,
#                          order.executed.value,
#                          order.executed.comm))
#             self.bar_executed = len(self)
#         elif order.status in [order.Canceled,order.Margin,order.Rejected]:
#             self.log('Order Canceled/Margin/Rejected')
#         self.order = None
#
#     def notify_trade(self,trade):
#         if not trade.isclosed:
#             return
#         self.log("OPERATION PROFIT, GROSS %.2f,NET %.2f" %
#                  (trade.pnl,trade.pnlcomm))
#
#     def next(self):
#         #Simply log the closing price of the series from the reference
#         self.log('Close,%.2f' % self.dataclose[0])
#
#         #Check if an order is pending ... if yes,we cannot send 2nd one
#         if self.order:
#             return
#         #Check if we are in the market
#         if not self.position:
#
#             if self.dataclose[0] < self.dataclose[-1]:
#                 #current close less than previous close
#                 if self.dataclose[-1] < self.dataclose[-2]:
#                     #previous close less than the previous close
#                     self.log('Buy CREATE, %.2f' % self.dataclose[0])
#                     self.order = self.buy()
#         else:
#             #Already in the market .. we might sell
#             if len(self) >= (self.bar_executed + self.params.exitbars):
#                 #SELL,SELL,SELL(with all possible default parameters
#                 self.log('SELL CREATE,%.2f' % self.dataclose[0])
#                 #Keep trace of the created order to avoid a 2nd order
#                 self.order = self.sell()
#


# #Create a Strategy
class TestStrategy(bt.Strategy):
    params = (
        ('exitbars', 5),
        ('maperiod', 15),
        ('printlog',False)
    )

    def log(self, txt,dt=None, doprint=None):
        """
        Logging function for this strategy
        """
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # keep a reference to the 'close' line in data[0] dataseries
        self.dataclose = self.datas[0].close
        # To keep track of pending orders
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add a MovingAverageSimple indicator
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod
        )

        # Indicators for the plotting show
        # bt.indicators.ExponentialMovingAverage(self.datas[0],period=25)
        # bt.indicators.WeightedMovingAverage(self.datas[0],period=25,subplot=True)
        # bt.indicators.MACDHisto(self.datas[0])
        # rsi= bt.indicators.RSI(self.datas[0])
        # bt.indicators.SmoothedMovingAverage(rsi,period=10)
        # bt.indicators.ATR(self.datas[0],plot=False)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return
        # Check if an order has been completed
        # Attention:broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log("BUY EXECUTED,Price:%.2f Cost:%.2f Comm:%.2f" %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log('SELL EXECUTED,Price: %.2f,Cose:%.2f Comm:%.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log("OPERATION PROFIT, GROSS %.2f,NET %.2f" %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close,%.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes,we cannot send 2nd one
        if self.order:
            return
        # Check if we are in the market
        if not self.position:

            if self.dataclose[0] > self.sma[0]:
                # current close less than previous close
                if self.dataclose[-1] < self.dataclose[-2]:
                    # previous close less than the previous close
                    self.log('Buy CREATE, %.2f' % self.dataclose[0])
                    self.order = self.buy()
        else:
            # Already in the market .. we might sell
            if self.dataclose[0] < self.sma[0]:
                # SELL SELL SELL!!! (with all possible default parameter)
                # if len(self) >= (self.bar_executed + self.params.exitbars):
                # SELL,SELL,SELL(with all possible default parameters
                self.log('SELL CREATE,%.2f' % self.dataclose[0])
                # Keep trace of the created order to avoid a 2nd order
                self.order = self.sell()

    def stop(self):
        self.log('(MA Period %2d) Ending Value %.2f' % (self.params.maperiod,self.broker.getvalue()),doprint=True)

if __name__ == "__main__":
    cerebro = bt.Cerebro()
    # cerebro.addstrategy(TestStrategy)
    # cerebro.addstrategy(MyStrategy)

    #Add a stragegy
    strats= cerebro.optstrategy(
        TestStrategy,
        maperiod=range(10,31))
    # data = bt.feeds.YahooFinanceData(dataname='orcl',
    #                                  fromdate=datetime.datetime(2000, 1, 1),
    #                                  todate=datetime.datetime(2000, 12, 31))

    data = MyHLOC(dataname='/home/rick/data/BTCUSDT_20190101_20191231_minute.csv')
    cerebro.adddata(data)
    cerebro.broker.setcash(10000.0)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=1)
    cerebro.broker.setcommission(commission=0.0)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    #Run over everything
    # cerebro.run(maxcpus=1)
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    # cerebro.plot()

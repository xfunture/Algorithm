import backtrader as bt
from datetime import datetime
import backtrader.indicators as btind
from backtrader import cerebro
from csvfeeds import MyHLOC
from datetime import timedelta
import time

def connect_broker():
    path = '/home/rick/PycharmProjects/Btc/huobittrade/binance.txt'
    f = open(path, 'r')
    all = f.readlines()
    f.close()
    apikey = all[0].strip()
    skey = all[1].strip()
    config = {'urls': {'api': 'https://api.binance.com/wapi/v3'},
              'apiKey': apikey,
              'secret': skey,
              'nonce': lambda: str(int(time.time() * 1000))
              }

    broker = bt.brokers.CCXTBroker(exchange='binance',
                                   currency='USD', config=config)
    cerebro.setbroker(broker)

    # Create data feeds
    data_ticks = bt.feeds.CCXT(exchange='binance', symbol='BTC/USDT',
                              name="btc_usdt_tick",
                              timeframe=bt.TimeFrame.Ticks,
                              compression=1, config=config)
    cerebro.adddata(data_ticks)


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




def main():
    cerebro = bt.Cerebro()

    hist_start_date = datetime.utcnow() - timedelta(minutes=10)
    data_min = bt.feeds.CCXT(exchange='binance', symbol="BTC/USDT", name="btc_usdt_min", fromdate=hist_start_date,
                             timeframe=bt.TimeFrame.Minutes)
    cerebro.adddata(data_min)
    cerebro.addstrategy(TestStrategy)
    cerebro.run()

main()
import backtrader as bt
import time
from datetime import datetime
from datetime import timedelta
import ccxt
import sys
exchange = ccxt.poloniex({
    'proxies': {
        'http': '127.0.0.1:40043',  # these proxies won't work for you, they are here for example
        'https': '127.0.0.1:40043',
    },
})


def read(path):
    f = open(path, 'r')
    all = f.readlines()
    f.close()
    apikey = all[0].strip()
    skey = all[1].strip()

    return apikey, skey


class TestStrategy(bt.Strategy):
    def notify_data(self, data, status, *args, **kwargs):
        print('*' * 5, 'DATA NOTIF:', data._getstatusname(status))

    def next(self):
        print('*' * 5, 'NEXT:',
              bt.num2date(self.data0.datetime[0]),
              self.data0._name,
              self.data0.open[0],
              self.data0.high[0],
              self.data0.low[0],
              self.data0.close[0],
              bt.TimeFrame.getname(self.data0._timeframe),
              len(self.data0))


def runstrategy(argv):
    # Create a cerebro
    cerebro = bt.Cerebro()

    # data = bt.feeds.CCXT(exchange='binance', symbol='BTC/USDT', timeframe=bt.TimeFrame.Minutes)
    hist_start_date = datetime.utcnow() - timedelta(minutes=1024)
    data_ticks = bt.feeds.CCXT(
        exchange='binance',
        symbol='BTC/USDT',
        name='BTC_USDT',
        timeframe=bt.TimeFrame.Minutes,
        fromdate=hist_start_date,
        historical='False',
        compression=1)
    # cerebro.resampledata(data, timeframe=bt.TimeFrame.Seconds)
    cerebro.adddata(data_ticks)

    # Add the strategy
    cerebro.addstrategy(TestStrategy)

    # Run the strategy
    cerebro.run()
    cerebro.plot()


if __name__ == '__main__':
    sys.exit(runstrategy(sys.argv))
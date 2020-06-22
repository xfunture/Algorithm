from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
import time

from datetime import datetime, timedelta

import backtrader as bt

from pandas import bdate_range
def read(path):
    f = open(path, 'r')
    all = f.readlines()
    f.close()
    apikey = all[0].strip()
    skey = all[1].strip()

    return apikey, skey


class TestStrategy(bt.Strategy):
    def next(self):
        for data in self.datas:
            print('*' * 5, 'NEXT:', bt.num2date(data.datetime[0]), data._name, data.open[0], data.high[0],
                  data.low[0], data.close[0], data.volume[0],
                  bt.TimeFrame.getname(data._timeframe), len(data))
            # if not self.getposition(data):
            #     order = self.buy(data, exectype=bt.Order.Limit, size=10, price=data.close[0])
            # else:
            #     order = self.sell(data, exectype=bt.Order.Limit, size=10, price=data.close[0])

    def notify_order(self, order):
        print('*' * 5, "NOTIFY ORDER", order)

def runstrategy(argv):
    # Create a cerebro
    cerebro = bt.Cerebro()
    apikey, secret = read('/home/rick/PycharmProjects/Btc/huobittrade/binance.txt')
    # Create broker
    broker_config = {
                # 'urls': {
                #          'logo': 'https://user-images.githubusercontent.com/1294454/27816857-ce7be644-6096-11e7-82d6-3c257263229c.jpg',
                #          'api': 'https://api.sandbox.gemini.com',
                #          'www': 'https://gemini.com',
                #          'doc': 'https://docs.gemini.com/rest-api',},
                     'apiKey': apikey,
                     'secret': secret,
                     'nonce': lambda: str(int(time.time() * 1000))
                    }
    broker = bt.brokers.CCXTBroker(exchange='binance', currency='USDT', config=broker_config)
    cerebro.setbroker(broker)

    # Create data feeds
    data_ticks = bt.feeds.CCXT(exchange='binance', symbol='BTC/USDT', name="btc_usd_tick",
                             timeframe=bt.TimeFrame.Ticks)
    cerebro.resampledata(data_ticks, timeframe=bt.TimeFrame.Seconds)
    cerebro.adddata(data_ticks)

    #hist_start_date = bdate_range(end=datetime.now(), periods=1)[0].to_pydatetime()
    #hist_start_date = datetime.utcnow() - timedelta(minutes=30)
    #data_min = bt.feeds.CCXT(exchange="gdax", symbol="BTC/USD", name="btc_usd_min",
    #                         timeframe=bt.TimeFrame.Minutes, fromdate=hist_start_date)
    #cerebro.adddata(data_min)

    # Add the strategy
    cerebro.addstrategy(TestStrategy)

    # Run the strategy
    cerebro.run()

if __name__ == '__main__':
    sys.exit(runstrategy(sys.argv))
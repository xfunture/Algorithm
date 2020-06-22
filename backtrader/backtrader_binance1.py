import time
import backtrader as bt
import datetime as dt
from ccxtbt import CCXTStore

DEBUG = True

def read(path):
    f = open(path, 'r')
    all = f.readlines()
    f.close()
    apikey = all[0].strip()
    skey = all[1].strip()

    return apikey, skey


class CustomStrategy(bt.Strategy):
    def __init__(self):
        self.order = None
        self.last_operation = "BUY"
        self.status = "DISCONNECTED"

    def notify_data(self, data, status, *args, **kwargs):
        self.status = data._getstatusname(status)
        if status == data.LIVE:
            self.log("LIVE DATA - Ready to trade")
        else:
            print(dt.datetime.now().strftime("%d-%m-%y %H:%M"), "NOT LIVE - %s" % self.status)

    def next(self):
        if self.status != "LIVE":
            self.log("%s - $%.2f" % (self.status, self.data0.close[0]))
            return

        if self.order:
            return

        print('*' * 5, 'NEXT:', bt.num2date(self.data0.datetime[0]), self.data0.close[0])

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected %s' % order.status)
            self.last_operation = None

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        color = 'green'
        if trade.pnl < 0:
            color = 'red'

        self.log(colored('OPERATION PROFIT, GROSS %.2f, NET %.2f' % (trade.pnl, trade.pnlcomm), color))

    def log(self, txt):
        if not DEBUG:
            return

        dt = self.data0.datetime.datetime()
        print('[%s] %s' % (dt.strftime("%d-%m-%y %H:%M"), txt))


def main():
    cerebro = bt.Cerebro(quicknotify=True)
    apikey,secret = read('/home/rick/PycharmProjects/Btc/huobittrade/binance.txt')

    broker_config = {
        'apiKey': apikey,
        'secret': secret,
        'nonce': lambda: str(int(time.time() * 1000)),
        'enableRateLimit': True,
    }
    store = CCXTStore(exchange='binance', currency='BTC', config=broker_config, retries=5, debug=True)
    broker_mapping = {
        'order_types': {
            bt.Order.Market: 'market',
            bt.Order.Limit: 'limit',
            bt.Order.Stop: 'stop-loss',  # stop-loss for kraken, stop for bitmex
            bt.Order.StopLimit: 'stop limit'
        },
        'mappings': {
            'closed_order': {
                'key': 'status',
                'value': 'closed'
            },
            'canceled_order': {
                'key': 'result',
                'value': 1
            }
        }
    }

    broker = store.getbroker(broker_mapping=broker_mapping)
    cerebro.setbroker(broker)

    hist_start_date = dt.datetime.utcnow() - dt.timedelta(minutes=2)
    data = store.getdata(
        dataname='BTC/USDT',
        name="BTC/USDT",
        timeframe=bt.TimeFrame.Minutes,
        fromdate=hist_start_date,
        compression=30,
        ohlcv_limit=99999
    )

    cerebro.adddata(data)
    cerebro.addstrategy(CustomStrategy)
    initial_value = cerebro.broker.getvalue()
    print('Starting Portfolio Value: %.2f' % initial_value)
    result = cerebro.run()
    final_value = cerebro.broker.getvalue()
    print('Final Portfolio Value: %.2f' % final_value)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("finished by user.")
        time = dt.datetime.now().strftime("%d-%m-%y %H:%M")

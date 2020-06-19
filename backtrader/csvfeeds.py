import datetime
import backtrader.feeds as btfeed


class MyHLOC(btfeed.GenericCSVData):
    params = (
        ('fromdate', datetime.datetime(2019, 1, 1)),
        ('todate', datetime.datetime(2019, 5, 30)),
        ('nullvalue', 0.0),
        # ('dtformat', ('%Y-%m-%d')),
        # ('tmformat', ('%H.%M.%S')),
        ('datetime', 0),
        ('time', -1),
        ('high', 8),
        ('low', 9),
        ('open', 7),
        ('close', 10),
        ('volume', 5),
        ('openinterest', -1)
    )
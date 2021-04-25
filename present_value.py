import numpy as np
from pandas import Timestamp, DatetimeIndex

default_start_date = Timestamp.now(tz=None).floor(freq='D')


def present_value(cashflows, dates, start_date=default_start_date, rate=.1, compound=True):
    cf = np.array(cashflows)
    dates = DatetimeIndex(dates)

    if cf.shape[0] != dates.shape[0]:
        print('Number of cashflows != number of dates. Exiting.')
        return

    pv = 0

    if compound:
        for t, date in enumerate(dates):
            delta = (date - start_date).days
            n_years = delta / 365
            pv += cf[t] / (1 + rate) ** n_years

    else:
        for t, date in enumerate(dates):
            delta = (date - start_date).days
            n_years = delta / 365
            pv += cf[t] / (1 + rate * n_years)

    return pv

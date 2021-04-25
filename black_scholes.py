import numpy as np
from scipy.stats import norm


def get_call_price(spot_price=100, strike=100, risk_free_rate=0, dividend_rate=0, volatility=0, time_to_expiration=0):
    d1 = (np.log(spot_price / strike) +
          (risk_free_rate - dividend_rate + (volatility ** 2) / 2) * time_to_expiration) / \
         (volatility * time_to_expiration ** .5)
    d2 = d1 - volatility * time_to_expiration ** .5

    standard_normal_cdf = norm(loc=0, scale=1).cdf

    N_d1 = standard_normal_cdf(d1)
    N_d2 = standard_normal_cdf(d2)

    call_price = spot_price * np.exp(- dividend_rate * time_to_expiration) * N_d1 - \
        strike * np.exp(- dividend_rate * time_to_expiration) * N_d2

    return call_price

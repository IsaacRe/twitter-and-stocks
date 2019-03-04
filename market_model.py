"""
Code for using DJIA index as a market model for stock prediction
"""
from scipy.stats import linregress
import numpy as np


class MarketModel:

    def __init__(self, alpha=0.0, beta=0.0):
        self._alpha = alpha
        self._beta = beta
        self.variance = None

    def fit(self, djia, returns):
        assert len(djia) == len(returns), "Lengths of DJIA returns don't match actual returns"
        # Mask out nan's
        mask = ~np.isnan(djia) & ~np.isnan(returns)
        self._beta, self._alpha, _, _, std_err = linregress(np.array(djia)[mask], np.array(returns)[mask])
        self.variance = std_err ** 2

    def predict(self, djia):
        mean = self._alpha + self._beta * djia
        return mean, self.variance

import pandas as pd
import numpy as np

from scipy.stats import chi2


chat_id = 734920047 # Ваш chat ID, не меняйте название переменной

def solution(p: float, x: np.array) -> tuple:
    n = len(x)
    alpha = 1 - p
    loc = x.mean()
    scale = np.sqrt(np.var(x) * 22 / n)
    chi2_left = chi2.ppf(alpha / 2, df=n-1)
    chi2_right = chi2.ppf(1 - alpha / 2, df=n-1)
    return np.sqrt(((n - 1) * np.var(x)) / chi2_right), \
           np.sqrt(((n - 1) * np.var(x)) / chi2_left)

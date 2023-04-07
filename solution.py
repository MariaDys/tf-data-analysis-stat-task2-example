import pandas as pd
import numpy as np

from scipy.stats import norm


chat_id = 734920047 # Ваш chat ID, не меняйте название переменной

def solution(p: float, x: np.array) -> tuple:
    alpha = 1 - p
    n = len(x)
    s = np.sqrt(np.var(x, ddof=1))
    chi2 = np.power(x, 2).sum() / np.power(s, 2)
    left = np.sqrt((n-1) * np.power(s, 2) / chi2) * norm.ppf(alpha/2) 
    right = np.sqrt((n-1) * np.power(s, 2) / chi2) * norm.ppf(1 - alpha/2)
    return left, right

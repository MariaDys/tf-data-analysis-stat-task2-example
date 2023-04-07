import pandas as pd
import numpy as np

from scipy.stats import norm


chat_id = 734920047 # Ваш chat ID, не меняйте название переменной

def solution(p: float, x: np.array) -> tuple:
    loc = x.mean()
    alpha = 1 - p
    n = len(x)
    if n < 30:
        n = 30
    scale = np.sqrt(np.var(x, ddof=1)) / np.sqrt(n)
    left = loc - scale * norm.ppf(1 - alpha / 2)
    right =  loc - scale * norm.ppf(alpha / 2)
    return left, right

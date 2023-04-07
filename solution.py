import pandas as pd
import numpy as np

from scipy.stats import t


chat_id = 734920047 # Ваш chat ID, не меняйте название переменной

def solution(p: float, x: np.array) -> tuple:
    n = len(x)
    x_mean = x.mean()
    s = np.sqrt(np.sum((x - x_mean)**2) / (n - 1))
    alpha = 1 - p
    t_alpha_2 = t.ppf(alpha/2, n-1)
    t_1_alpha_2 = t.ppf(1-alpha/2, n-1)
    left = x_mean - s * t_1_alpha_2 / np.sqrt(n)
    right = x_mean - s * t_alpha_2 / np.sqrt(n)
    return (left, right)

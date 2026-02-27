import numpy as np
from scipy.stats import t
 
def one_sample_t(data, mu0, alpha=0.05, alternative='two-sided'):
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)   # sample standard deviation
    df = n - 1
    # t statistic formula
    t_stat = (mean - mu0) / (std / np.sqrt(n))
    # p-value calculation
    if alternative == 'two-sided':
        p_value = 2 * (1 - t.cdf(abs(t_stat), df))
    elif alternative == 'greater':
        p_value = 1 - t.cdf(t_stat, df)
    elif alternative == 'less':
        p_value = t.cdf(t_stat, df)
    else:
        raise ValueError("Choose 'two-sided', 'greater', or 'less'")
    # Decision
    decision = "Reject H0" if p_value < alpha else "Fail to Reject H0"
    return t_stat, p_value, decision
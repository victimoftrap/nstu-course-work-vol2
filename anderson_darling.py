import math

import numpy as np
import scipy.stats as stats


def anderson_darling(xs, dist_type):
    """Тест критерием Андерсона-Дарлинга.

    :param xs: выборка
    :param dist_type: {'norm', 'rayleigh', 'cauchy'} тип распределения
    :return:
    """
    cdf_dicts = {
        'norm': stats.norm.cdf,
        'rayleigh': stats.rayleigh.cdf,
        'cauchy': stats.cauchy.cdf,
    }

    xs_sorted = np.sort(np.asarray(xs))
    n = len(xs_sorted)

    distribution_cdf_func = cdf_dicts[dist_type]

    s_value = 0
    for i in range(n):
        multiplier = (2 * i - 1.0) / (2 * n)
        dist_by_x = distribution_cdf_func(xs_sorted[i])

        s_value += (multiplier * math.log(dist_by_x)) + ((1 - multiplier) * math.log(1 - dist_by_x))
    a_value = -n - (2 * s_value)
    return a_value

import distributions_pdf as pdf

import numpy as np


def anderson_darling(xs, dist_name):
    """Тест критерием Андерсона-Дарлинга.

    :param xs: выборка
    :param dist_name: {'norm', 'relay', 'cauchy'} тип распределения
    :return:
    """
    def norm_wrapper(x):
        return pdf.norm(x, mju_mean=mju, sigma_variance=sigma)

    def relay_wrapper(x):
        return pdf.relay(x, sigma_variance=sigma)

    def cauchy_wrapper(x):
        return pdf.cauchy(x, mju_mean=mju, sigma_variance=sigma)

    pdf_dicts = {
        'norm': norm_wrapper,
        'relay': relay_wrapper,
        'cauchy': cauchy_wrapper
    }

    xs_sorted = np.sort(np.asarray(xs))
    n = len(xs_sorted)

    mju = np.mean(xs_sorted)
    sigma = np.sqrt(np.mean((xs_sorted - mju) ** 2))

    distribution_pdf_func = pdf_dicts[dist_name]

    a_value = - n
    s_value = 0
    for i in range(n):
        multiplier = (2 * i - 1.0) / 2 * n
        s_value += multiplier * np.log(distribution_pdf_func(xs_sorted[i])) \
            + (1 - multiplier) * np.log(1 - distribution_pdf_func(xs_sorted[i]))
    a_value -= 2 * s_value
    return a_value

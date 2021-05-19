from numpy import sqrt, exp, pi


def norm(x, mju_mean, sigma_variance):
    """Функция плотности, задающая нормальное распределение.
    f(x, μ, σ) = [1 / σ √(2π)] * exp{ -1/2 * (x−μ / σ)^2 }

    :param x:
    :param mju_mean: параметр сдвига
    :param sigma_variance: параетр масштаба
    :return: значение случайной величины
    """
    return (1 / (sigma_variance * sqrt(2 * pi))) * exp(-0.5 * ((x - mju_mean) / sigma_variance) ** 2)


def relay(x, sigma_variance):
    """Функция плотности, задающая распределение Рэлея.
    f(x, σ) = (x / σ^2) * exp{ -x^2 / 2σ^2 }

    :param x:
    :param sigma_variance: параетр масштаба
    :return: значение случайной величины
    """
    return (x / (sigma_variance ** 2)) * exp(- (x ** 2) / (2 * (sigma_variance ** 2)))


def cauchy(x, mju_mean, sigma_variance):
    """Функция плотности, задающая распределение Коши.
    f(x, μ, σ) = σ / π[σ^2 + (x−μ)^2]

    :param x:
    :param mju_mean: параметр сдвига
    :param sigma_variance: параетр масштаба
    :return: значение случайной величины
    """
    return sigma_variance / (pi * (sigma_variance ** 2 + (x - mju_mean) ** 2))

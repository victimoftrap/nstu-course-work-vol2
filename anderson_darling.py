import math


def anderson_darling(stat_values):
    """Тест критерием Андерсона-Дарлинга.

    :param stat_values: выборка статистик
    :return:
    """
    n = len(stat_values)

    s_value = 0
    for i in range(n):
        multiplier = (2 * i - 1.0) / (2 * n)
        dist_by_x = stat_values[i]

        s_value += (multiplier * math.log(dist_by_x)) + ((1 - multiplier) * math.log(1 - dist_by_x))
    a_value = -n - (2 * s_value)
    return a_value

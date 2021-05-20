import anderson_darling as ad
import isw

import argparse

import numpy as np


def generate_gn(n, dist_type='norm', iterations=16600):
    """Моделирование закона распределения Gn(x).

    Gn(x) моделируется с помощью метода Монте-Карло.
    Для начала генерируется выборка значений статистик критерия.
    По ней строится эмпирическая функция распределения Gn,N(x)

    :param n: объём выборки
    :param dist_type: {'norm', 'rayleigh', 'cauchy'} тип распределения
    :param iterations: количество итераций
    :return: закон распределения Gn(x)
    """
    random_sample_funcs = {
        'norm': np.random.normal,
        'rayleigh': np.random.rayleigh,
        'cauchy': np.random.standard_cauchy,
    }
    random_sample = random_sample_funcs[dist_type]

    data = [0 for i in range(iterations)]
    for i in range(iterations):
        xs = random_sample(size=n)
        criteria_stat = ad.anderson_darling(xs, dist_type)
        data[i] = criteria_stat
    g_n = np.sort(data)
    return g_n


def main(n, iters, distribution_type):
    print('nstu-course-work-vol2: ver:0.Patrego')

    gn_data = generate_gn(n, distribution_type, iters)
    isw.save_to_isw_file(
        data=gn_data,
        filename=f'ad_{distribution_type}_{n}_{iters}.dat',
        description=f'Anderson-Darling by {distribution_type}'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anderson-Darling')
    parser.add_argument('-n', dest='n', type=int, help='объём выборки')
    parser.add_argument('-N', dest='N', type=int, help='объём моделирования')
    parser.add_argument('-dist', dest='dist', type=str, help='тип закона распределения')
    parsed_args = parser.parse_args()

    n = 10
    N = 2649147
    dist = 'norm'
    main(n, N, dist)

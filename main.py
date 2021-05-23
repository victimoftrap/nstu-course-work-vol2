import anderson_darling as ad
import isw

import argparse
from datetime import datetime

import numpy as np
import scipy.stats as stats


def generate_gn(n, dist_type='norm', stats_sample_size=16600):
    """Моделирование закона распределения Gn(x).

    Gn(x) моделируется с помощью метода Монте-Карло.
    Для начала генерируется выборка значений статистик критерия.
    По ней строится эмпирическая функция распределения Gn,N(x)

    :param n: объём выборки наблюдений случайной величины
    :param dist_type: {'norm', 'rayleigh', 'cauchy'} тип распределения
    :param stats_sample_size: объём выборки значений статистик критерия
    :return: закон распределения Gn(x)
    """
    random_sample_funcs = {
        'norm': {
            'sample_func': np.random.normal,
            'stat_func': stats.norm.cdf,
        },
        'rayleigh': {
            'sample_func': np.random.rayleigh,
            'stat_func': stats.rayleigh.cdf,
        },
        'cauchy': {
            'sample_func': np.random.standard_cauchy,
            'stat_func': stats.cauchy.cdf,
        },
    }

    random_sample = random_sample_funcs[dist_type]['sample_func']
    stat_func = random_sample_funcs[dist_type]['stat_func']

    data = [0 for i in range(stats_sample_size)]
    for i in range(stats_sample_size):
        xs = random_sample(size=n)
        xs = np.sort(xs)
        stat_values = stat_func(xs)

        criteria_stat = ad.anderson_darling(stat_values)
        data[i] = criteria_stat
    g_n = np.sort(data)
    return g_n


def main(n, iters, distribution_type):
    print(f'Моделирование Gn(x) с {distribution_type} законом, n={n}, N={iters}')
    gn_data = generate_gn(n, distribution_type, iters)

    print('Сохранение в файл...')
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

    print('nstu-course-work-vol2: ver:0.1')
    n = 100
    # N = 2_649_147
    N = 1_658_944
    # N = 16_600

    # dist = 'norm'
    # dist = 'rayleigh'
    dist = 'cauchy'

    start_time = datetime.now()
    print(f"Старт программы: {start_time}\n")
    main(n, N, dist)
    end_time = datetime.now()
    print(f"\nКонец программы: {end_time}")
    print(f"Потрачено {end_time - start_time} времени!")

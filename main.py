import anderson_darling as ad
import isw

import argparse
from datetime import datetime

import numpy as np


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
        'norm': np.random.normal,
        'rayleigh': np.random.rayleigh,
        'cauchy': np.random.standard_cauchy,
    }
    random_sample = random_sample_funcs[dist_type]

    data = [0 for i in range(stats_sample_size)]
    for i in range(stats_sample_size):
        xs = random_sample(size=n)
        criteria_stat = ad.anderson_darling(xs, dist_type)
        data[i] = criteria_stat
    # g_n = np.sort(data)
    return g_n


def main(n, iters, distribution_type):
    print('nstu-course-work-vol2: ver:0.Patrego')

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

    n = 50
    # N = 2_649_147
    N = 1_658_944
    # N = 16_600

    # dist = 'norm'
    # dist = 'rayleigh'
    dist = 'cauchy'

    start_time = datetime.now()
    print(f"Старт вычислений: {start_time}")
    # main(n, N, dist)
    end_time = datetime.now()
    print(f"Конец вычислений: {end_time}")
    print(f"Я потратил {end_time - start_time} времени!")

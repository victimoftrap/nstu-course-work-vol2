import anderson_darling as ad

import  numpy as np


def generate_gn(n, iterations=16600):
    """Моделирование закона распределения Gn(x).

    Gn(x) моделируется с помощью метода Монте-Карло.
    Для начала генерируется выборка значений статистик критерия.
    По ней строится эмпирическая функция распределения Gn,N(x)

    :param n: объём выборки
    :param iterations: количество итераций
    :return: закон распределения Gn(x)
    """
    data = [0 for i in range(iterations)]
    for i in range(iterations):
        xs = np.random.normal(size=n)
        criteria_stat = ad.anderson_darling(xs, 'norm')
        print(criteria_stat)
        data[i] = criteria_stat

    g_n = [np.sum(data[i::-1]) / n for i in range(n)]
    return g_n


def main():
    print('Patrego')
    gn = generate_gn(10)
    print(f"Gn(x) = {gn}")


if __name__ == '__main__':
    main()

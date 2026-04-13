import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy, laplace, poisson, uniform



def generate_and_draw(n, a, d, name):
    """
    n - размер выборки (10, 100 или 1000)
    a - объект "оси"
    d - объект распределения
    name - название распределения
    """

    # Генерируем случайные числа на основе выбранного распределения d
    sample = d.rvs(size=n)

    # Подбор количества бинов
    if n <= 10:
        k = int(np.ceil(np.log2(n) + 1))# Формула Стерджеса
        method_name = "Стерджес"
    else:
        #используем правило Фридмана-Диакониса

        q75, q25 = np.percentile(sample, [75, 25])
        iqr = q75 - q25

        if iqr > 0:
            width = 2 * iqr * (n ** (-1 / 3))#формула
            k = int(np.ceil((np.max(sample) - np.min(sample)) / width))
        else:
            k = 30  # Защита

        if k > 50:
            k = 50
        method_name = "Фридман-Диаконис"


    # Фиксируем границы
    if "Пуассон" in name:
        plot_range = (0, 15)
        k = 15
    elif "Равномерное" in name:
        plot_range = (-2.5, 2.5)  # Равномерное от -sqrt(3) до sqrt(3)
    else:
        plot_range = (-5, 5)

    a.hist(sample, bins=k, range=plot_range, density=True, alpha=0.6,
           color='skyblue', edgecolor='black', label='Гистограмма')

    # Теоретическая кривая
    x = np.linspace(plot_range[0], plot_range[1], 1000)  # Сетка из 1000 точек

    if "Пуассон" in name:
        # Тк Пуассон дискретный, рисуем ступеньками
        x_discrete = np.arange(0, 16)
        a.step(x_discrete, d.pmf(x_discrete), where='mid', color='red', lw=2, label='Теория (PMF)')
    else:
        # Для непрерывных плавная линия
        a.plot(x, d.pdf(x), color='red', lw=2, label='Теория (PDF)')

    a.set_title(f'n = {n}\n{method_name}, k={k}', fontsize=9)
    a.legend(fontsize='x-small')
    a.grid(True, alpha=0.3)



def show_different_n(d, name):

    n_sizes = [10, 100, 1000]  # Список объемов данных


    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Распределение: {name}', fontsize=16)


    for i, n in enumerate(n_sizes):
        generate_and_draw(n, axes[i], d, name)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



if __name__ == "__main__":

    s2, s3 = np.sqrt(2), np.sqrt(3)


    dist_list = [
        (norm(0, 1), "Нормальное N(x; 0, 1)"),
        (cauchy(0, 1), "Коши C(x; 0, 1)"),
        (laplace(0, 1 / s2), "Лаплас L(x; 0, 1/√2)"),
        (poisson(5), "Пуассон P(k; 5)"),
        (uniform(-s3, 2 * s3), "Равномерное U(x; -√3, √3)")
    ]

    for dist_obj, dist_name in dist_list:
        show_different_n(dist_obj, dist_name)
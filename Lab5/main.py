import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse




# Функция расчета квадрантной корреляции
def calculate_quadrant(x, y):
    """Выборочный квадрантный коэффициент корреляции"""
    x_med, y_med = np.median(x), np.median(y)
    n = len(x)
    n1 = np.sum((x > x_med) & (y > y_med))
    n2 = np.sum((x < x_med) & (y > y_med))
    n3 = np.sum((x < x_med) & (y < y_med))
    n4 = np.sum((x > x_med) & (y < y_med))
    return ((n1 + n3) - (n2 + n4)) / n


# Функция расчета всех трех коэффициентов
def get_all_correlations(x, y):
    """Считает коэффициенты Пирсона, Спирмена и Квадрантный"""
    pearson, _ = stats.pearsonr(x, y)
    spearman, _ = stats.spearmanr(x, y)
    quadrant = calculate_quadrant(x, y)
    return pearson, spearman, quadrant


# Функция для вывода таблиц
def generate_tables(n_sizes, rho_values, iterations=1000):
    """Генерирует и выводит статистические таблицы для разных n"""
    for n in n_sizes:
        print(f"\n" + "=" * 30 + f" Таблица для n = {n} " + "=" * 30)
        results = []
        for rho in rho_values + ['mixture']:
            p_list, s_list, q_list = [], [], []
            for _ in range(iterations):
                if rho == 'mixture':
                    # Смесь: 90% корреляции 0.9 и 10% выбросов
                    d_main = np.random.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], int(n * 0.9))
                    d_out = np.random.multivariate_normal([0, 0], [[10, -0.9], [-0.9, 10]], n - int(n * 0.9))
                    data = np.vstack([d_main, d_out])
                else:
                    data = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)

                p, s, q = get_all_correlations(data[:, 0], data[:, 1])
                p_list.append(p);
                s_list.append(s);
                q_list.append(q)

            results.append({
                'p': f"ρ={rho}" if rho != 'mixture' else 'mixture',
                'Pearson μ': np.mean(p_list), 'Pearson σ²': np.var(p_list),
                'Spearman μ': np.mean(s_list), 'Spearman σ²': np.var(s_list),
                'Quadrant μ': np.mean(q_list), 'Quadrant σ²': np.var(q_list)
            })
        print(pd.DataFrame(results).to_string(index=False, float_format=lambda x: f"{x:10.6f}"))


    # Функция построения диаграмм с эллипсами
def plot_ellipses(n_sizes, rho_values, n_std=2.0):
    """Строит диаграммы рассеивания с эллипсами для каждого случая"""
    for rho in rho_values + ['mixture']:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Сравнение для {'смеси' if rho == 'mixture' else 'ρ=' + str(rho)}")

        angles_mixture = []
        for i, n in enumerate(n_sizes):
            # Генерация данных для графиков
            if rho == 'mixture':
                d_m = np.random.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], int(n * 0.9))
                d_o = np.random.multivariate_normal([0, 0], [[10, -0.9], [-0.9, 10]], n - int(n * 0.9))
                data = np.vstack([d_m, d_o])
            else:
                data = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)

            x, y = data[:, 0], data[:, 1]
            cov = np.cov(x, y)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]

            # Расчет угла и параметров эллипса
            rad = np.arctan2(*vecs[:, 0][::-1])
            if rho == 'mixture': angles_mixture.append(round(rad, 2))

            # Отрисовка
            axes[i].scatter(x, y, s=15, color='steelblue', alpha=0.6)
            ellipse = Ellipse(xy=(np.mean(x), np.mean(y)),
                              width=2 * n_std * np.sqrt(vals[0]),
                              height=2 * n_std * np.sqrt(vals[1]),
                              angle=np.degrees(rad), edgecolor='red', fc='none', ls='--')
            axes[i].add_patch(ellipse)
            axes[i].set_title(f"n={n}")
            axes[i].grid(True, alpha=0.2)

        if rho == 'mixture':
            print(f"\nУглы наклона для смеси (рад): {angles_mixture}")
        plt.tight_layout()
        plt.show()



N_SIZES = [20, 60, 100]
RHO_VALS = [0, 0.5, 0.9]

generate_tables(N_SIZES, RHO_VALS)
plot_ellipses(N_SIZES, RHO_VALS)

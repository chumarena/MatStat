import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats


def get_outliers_count(data):
    """Считает количество точек, выходящих за усы Тьюки."""
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return np.sum((data < lower_bound) | (data > upper_bound))


def get_theoretical_share(dist_type, **params):
    """Вычисляет математическую вероятность появления выброса."""
    dist_obj = getattr(stats, dist_type)(**params)
    q1, q3 = dist_obj.ppf([0.25, 0.75])
    iqr = q3 - q1
    return dist_obj.cdf(q1 - 1.5 * iqr) + dist_obj.sf(q3 + 1.5 * iqr)


def plot_boxplots(dist_list, n):
    """Визуализация боксплотов в отдельном окне."""
    plot_data = []
    for d in dist_list:
        sample = d["generator"](n)
        for val in sample:
            plot_data.append({"Распределение": d["name"], "Значение": val})

    df = pd.DataFrame(plot_data)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Распределение", y="Значение", hue="Распределение",
                palette="Set3", legend=False)

    plt.title(f"Боксплоты Тьюки (объем выборки n = {n})", fontsize=14)
    plt.ylim(-5, 15)
    plt.grid(axis='y', alpha=0.3)
    plt.show()


def boxplots_Tiuki():
    n_sizes = [20, 100]
    iterations = 1000

    dist_configs = [
        {"name": "Нормальное", "generator": lambda n: np.random.normal(0, 1, n),
         "theory_params": ("norm", {"loc": 0, "scale": 1})},
        {"name": "Коши", "generator": lambda n: np.random.standard_cauchy(n),
         "theory_params": ("cauchy", {"loc": 0, "scale": 1})},
        {"name": "Лапласа", "generator": lambda n: np.random.laplace(0, 1 / np.sqrt(2), n),
         "theory_params": ("laplace", {"loc": 0, "scale": 1 / np.sqrt(2)})},
        {"name": "Пуассона", "generator": lambda n: np.random.poisson(5, n),
         "theory_params": ("poisson", {"mu": 5})},
        {"name": "Равномерное", "generator": lambda n: np.random.uniform(-np.sqrt(3), np.sqrt(3), n),
         "theory_params": ("uniform", {"loc": -np.sqrt(3), "scale": 2 * np.sqrt(3)})}
    ]


    for n in n_sizes:
        plot_boxplots(dist_configs, n)


    print()
    header = f"{'Распределение':<15} | {'n':<4} | {'Эксперимент':<12} | {'Теория':<10}"
    print(header)
    print("-" * 55)

    for d in dist_configs:
        dist_type, params = d["theory_params"]
        theory_val = get_theoretical_share(dist_type, **params) * 100
        theory_str = f"{theory_val:.2f}%"

        for n in n_sizes:
            shares = [get_outliers_count(d["generator"](n)) / n for _ in range(iterations)]
            mean_exp_pct = np.mean(shares) * 100
            exp_str = f"{mean_exp_pct:.2f}%"


            print(f"{d['name']:<15} | {n:<4} | {exp_str:<12} | {theory_str:<10}")




if __name__ == "__main__":
    boxplots_Tiuki()

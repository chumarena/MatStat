import numpy as np
import pandas as pd
from scipy import stats



def get_mean(data):
    """Выборочное среднее"""
    return np.mean(data)


def get_med(data):
    """Медиана"""
    return np.median(data)


def get_z_R(data):
    """Полусумма экстремальных элементов"""
    return (np.min(data) + np.max(data)) / 2


def get_z_Q(data):
    """Полусумма квартилей"""
    q1 = np.quantile(data, 0.25)
    q3 = np.quantile(data, 0.75)
    return (q1 + q3) / 2


def get_z_tr(data):
    """Усечённое среднее (отбрасываем по 10% с краев)"""
    return stats.trim_mean(data, 0.1)


# Функция для сбора всех оценок
def calculate_all_estimates(data):
    return [
        get_mean(data),
        get_med(data),
        get_z_R(data),
        get_z_Q(data),
        get_z_tr(data)
    ]


# Основная функция
def haract_polozh_i_rasseiv():

    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_colwidth', None)

    # Параметры согласно заданию
    n_sizes = [10, 100, 1000]
    iterations = 1000
    labels = ["Выборочное среднее", "Медиана", "z_R", "z_Q", "z_tr"]


    distributions = {
        "Нормальное N(0, 1)": lambda n: np.random.normal(0, 1, n),
        "Коши C(0, 1)": lambda n: np.random.standard_cauchy(n),
        "Лаплас L(0, 1/sqrt(2))": lambda n: np.random.laplace(0, 1 / np.sqrt(2), n),
        "Пуассон P(5)": lambda n: np.random.poisson(5, n),
        "Равномерное U(-sqrt(3), sqrt(3))": lambda n: np.random.uniform(-np.sqrt(3), np.sqrt(3), n)
    }

    # Теоретические значения E(z) для каждой характеристики
    theoretical_values = {
        "Нормальное N(0, 1)": [0, 0, 0, 0, 0],
        "Коши C(0, 1)": ["не определено", 0, "не определено", 0, 0],
        "Лаплас L(0, 1/sqrt(2))": [0, 0, 0, 0, 0],
        "Пуассон P(5)": [5, 5, "inf", 4.5, 5],
        "Равномерное U(-sqrt(3), sqrt(3))": [0, 0, 0, 0, 0]
    }

    for dist_name, dist_func in distributions.items():
        print(f"\nТаблица для распределения: {dist_name}")
        table_data = []

        for n in n_sizes:
            all_iters_results = []
            for _ in range(iterations):
                sample = dist_func(n)
                all_iters_results.append(calculate_all_estimates(sample))

            all_iters_results = np.array(all_iters_results)
            e_z = np.mean(all_iters_results, axis=0)# среднее
            d_z = np.var(all_iters_results, axis=0)# дисперсия

            row = []
            for i in range(len(labels)):
                # Округляем до 1 знака
                res_str = f"{e_z[i]:.1f} ± {np.sqrt(d_z[i]):.1f}"
                row.append(res_str)
            table_data.append(row)


        df = pd.DataFrame(np.array(table_data).T, columns=[f"n={n}" for n in n_sizes], index=labels)


        df["Теоретическое значение"] = theoretical_values[dist_name]

        print(df)


if __name__ == "__main__":
    haract_polozh_i_rasseiv()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd


x = np.arange(-1.8, 2.1, 0.2)
n = len(x)


a_true, b_true = 2.0, 2.0



epsilon = np.random.normal(0, 1, n)
y_clean = b_true + a_true * x + epsilon

# Добавление выбросов
y_outliers = y_clean.copy()
y_outliers[0] += 10  # y1 <- y1 + 10
y_outliers[-1] -= 10  # y20 <- y20 - 10




def mnk_fit(x, y):
    """Метод наименьших квадратов (аналитически)"""
    x_mean, y_mean = np.mean(x), np.mean(y)
    a = (np.mean(x * y) - x_mean * y_mean) / (np.mean(x ** 2) - x_mean ** 2)
    b = y_mean - a * x_mean
    return a, b


def mnm_loss(params, x, y):
    """Целевая функция для МНМ (сумма модулей)"""
    a, b = params
    return np.sum(np.abs(y - (a * x + b)))


def mnm_fit(x, y):
    """Метод наименьших модулей (через BFGS)"""
    res = minimize(mnm_loss, x0=[1, 1], args=(x, y), method='BFGS')
    return res.x



def get_stats(a_hat, b_hat):
    """Расчет по формуле δz = |z - z_hat| / z * 100%"""
    da = abs(a_true - a_hat)
    db = abs(b_true - b_hat)
    sa = (da / a_true) * 100
    sb = (db / b_true) * 100
    return [a_hat, da, sa, b_hat, db, sb]


def print_results_table(x_data, y_data, title):
    a_ls, b_ls = mnk_fit(x_data, y_data)
    a_lad, b_lad = mnm_fit(x_data, y_data)

    columns = ["Метод", "a", "Δ a", "δ a, %", "b", "Δ b", "δ b, %"]
    data = [
        ["МНК"] + get_stats(a_ls, b_ls),
        ["МНМ"] + get_stats(a_lad, b_lad)
    ]

    df = pd.DataFrame(data, columns=columns)
    print(f"\n{title}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    return (a_ls, b_ls), (a_lad, b_lad)



def plot_lab(x, y, params_ls, params_lad, title):
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle=':', alpha=0.6)

    # Точки и истинный закон
    plt.scatter(x, y, color='lightgray', edgecolors='black', label='Данные')
    plt.plot(x, b_true + a_true * x, '--', color='black', alpha=0.3, label='Истинный закон')

    # Регрессионные прямые (красно-синяя гамма)
    plt.plot(x, params_ls[1] + params_ls[0] * x, color='royalblue', lw=2, label='МНК')
    plt.plot(x, params_lad[1] + params_lad[0] * x, color='crimson', lw=2, label='МНМ')

    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()




# Без выбросов
p_ls_1, p_lad_1 = print_results_table(x, y_clean, "Таблица 1: Без выбросов")
plot_lab(x, y_clean, p_ls_1, p_lad_1, "Линейная регрессия (без выбросов)")

# С выбросами
p_ls_2, p_lad_2 = print_results_table(x, y_outliers, "Таблица 2: С выбросами (y1+10, y20-10)")
plot_lab(x, y_outliers, p_ls_2, p_lad_2, "Линейная регрессия (с выбросами)")

import numpy as np
import pandas as pd
from scipy import stats


def get_mean_and_interval(data, conf=0.95):
    """
    Вычисляет выборочное среднее и доверительный интервал для m.
    """
    n = len(data)
    mean = np.mean(data)
    s = np.std(data, ddof=1)
    alpha = 1 - conf

    # Квантиль распределения Стьюдента
    t_crit = stats.t.ppf(1 - alpha / 2, n - 1)

    # Полуширина интервала по формуле
    margin = (s * t_crit) / np.sqrt(n - 1)
    interval = (mean - margin, mean + margin)

    return mean, interval


def get_variance_and_interval(data, conf=0.95):
    """
    Вычисляет выборочную дисперсию и доверительный интервал для sigma^2.
    """
    n = len(data)
    s2 = np.var(data, ddof=1)
    alpha = 1 - conf

    # Квантили распределения хи-квадрат
    chi2_low = stats.chi2.ppf(alpha / 2, n - 1)
    chi2_high = stats.chi2.ppf(1 - alpha / 2, n - 1)

    # Интервал для дисперсии
    interval = ((s2 * n) / chi2_high, (s2 * n) / chi2_low)

    return s2, interval


def f_test_of_variances(sample1, sample2, alpha=0.05):
    """
    Проверяет гипотезу о равенстве дисперсий (тест Фишера).
    Использует статистику F = s1^2 / s2^2.
    """
    s1_2 = np.var(sample1, ddof=1)
    s2_2 = np.var(sample2, ddof=1)

    # Вычисление статистики F
    if s1_2 > s2_2:
        f_stat = s1_2 / s2_2
    else:
        f_stat = s2_2 / s1_2

    n1, n2 = len(sample1), len(sample2)
    # Критическое значение F_{1-alpha/2}(n-1, m-1)
    f_crit = stats.f.ppf(1 - alpha / 2, n1 - 1, n2 - 1)

    return f_stat, f_crit, f_stat > f_crit




# Генерация данных 
sample_20 = np.random.normal(0, 1, 20)
sample_100 = np.random.normal(0, 1, 100)

# Расчеты для таблицы
m20, m_int20 = get_mean_and_interval(sample_20)
v20, v_int20 = get_variance_and_interval(sample_20)

m100, m_int100 = get_mean_and_interval(sample_100)
v100, v_int100 = get_variance_and_interval(sample_100)


columns = pd.MultiIndex.from_tuples([
    ('m', 'Выборочное'),
    ('m', 'Доверительный интервал'),
    ('σ²', 'Выборочное'),
    ('σ²', 'Доверительный интервал')
])

data_rows = [
    [round(m20, 2), f"({m_int20[0]:.2f}, {m_int20[1]:.2f})",
     round(v20, 2), f"({v_int20[0]:.2f}, {v_int20[1]:.2f})"],
    [round(m100, 2), f"({m_int100[0]:.2f}, {m_int100[1]:.2f})",
     round(v100, 2), f"({v_int100[0]:.2f}, {v_int100[1]:.2f})"]
]

df = pd.DataFrame(data_rows, columns=columns, index=[20, 100])
df.index.name = 'Размер n'

print("Таблица доверительных интервалов:")
print(df.to_string())



# Вычисляем значения через созданную функцию
f_val, f_crit_right, rejected = f_test_of_variances(sample_20, sample_100)

print(f"\nРезультаты F-теста ")
print(f"Расчетное значение F (статистика): {f_val:.4f}")
print(f"Критическое значение F_1-α/2:      {f_crit_right:.4f}")


if rejected:
    print(f"Результат: F > F_1-α/2. Нулевая гипотеза H0 отвергается.")
else:
    print(f"Результат: F <= F_1-α/2. Нулевая гипотеза H0 не отвергается.")

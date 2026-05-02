import numpy as np
from scipy import stats


def calculate_frequencies(data, bin_edges):
    """
    Подсчитывает эмпирические частоты (n_i).
    """
    hist, _ = np.histogram(data, bins=bin_edges)
    return hist


def calculate_theoretical_probs(mu, sigma, bin_edges, n):
    """
    Вычисляет теоретические ожидаемые частоты (n*p_i).
    """
    # p_i = F(a_i) - F(a_i-1)
    p_i = np.diff(stats.norm.cdf(bin_edges, loc=mu, scale=sigma))
    return p_i * n


def refine_intervals(n_observed, n_expected):
    """
    Объединяет интервалы для выполнения условия n*p_i >= 5.
    """
    obs_refined = []
    exp_refined = []

    curr_obs = 0
    curr_exp = 0

    for o, e in zip(n_observed, n_expected):
        curr_obs += o
        curr_exp += e
        if curr_exp >= 5:
            obs_refined.append(curr_obs)
            exp_refined.append(curr_exp)
            curr_obs, curr_exp = 0, 0

    if curr_exp > 0:
        if len(exp_refined) > 0:
            obs_refined[-1] += curr_obs
            exp_refined[-1] += curr_exp
        else:
            obs_refined.append(curr_obs)
            exp_refined.append(curr_exp)

    return np.array(obs_refined), np.array(exp_refined)


def execute_chi_square_test(distribution, n):
    """
    Основная процедура проверки гипотезы.
    """
   
    if distribution == 'Normal':
        sample = np.random.normal(0, 1, n)
    elif distribution == 'Uniform':
        sample = np.random.uniform(-np.sqrt(3), np.sqrt(3), n)
    elif distribution == 'Laplace':
        sample = np.random.laplace(0, 1 / np.sqrt(2), n)

    # Оценка параметров методом максимального правдоподобия
    mu_mmp = np.mean(sample)
    sigma_mmp = np.std(sample)

    # Определение начального количества интервалов k
    k_initial = int(np.ceil(1.72 * (n ** (1 / 3))))

    # Границы интервалов с бесконечными краями
    edges = np.linspace(np.min(sample), np.max(sample), k_initial + 1)
    edges[0], edges[-1] = -np.inf, np.inf

    # Расчет частот и их корректировка
    n_i = calculate_frequencies(sample, edges)
    np_i = calculate_theoretical_probs(mu_mmp, sigma_mmp, edges, n)

    # Объединение малых интервалов
    n_final, np_final = refine_intervals(n_i, np_i)

    # Хи-квадрат
    # Формула: sum((n_i - np_i)^2 / np_i)[cite: 1]
    chi = np.sum((n_final - np_final) ** 2 / np_final)

    print(f"{distribution:7} | n: {n:3} | mu: {mu_mmp:.2f} | sigma: {sigma_mmp:.2f} | k_final: {len(n_final):2} | Chi2: {chi:.2f}")
    return chi



print("Результаты лабораторной работы:")
scenarios = [
    ('Normal', 100),
    ('Uniform', 20),
    ('Uniform', 300),
    ('Laplace', 20),
    ('Laplace', 300)
]

for dist_type, size in scenarios:
    execute_chi_square_test(dist_type, size)

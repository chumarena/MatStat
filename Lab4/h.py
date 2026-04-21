import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


def study_cauchy_bandwidth(n=100):

    dist = stats.cauchy(0, 1)
    x_lim = [-4, 4]
    title = "Cauchy $C(0, 1)$"


    adjusts = [0.1, 1.0, 2.5]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Влияние h на ядерную оценку (Коши, n={n})", fontsize=14)


    sample = dist.rvs(size=n)
    x_range = np.linspace(x_lim[0], x_lim[1], 1000)
    y_theory = dist.pdf(x_range)

    for i, adj in enumerate(adjusts):
        ax = axes[i]


        ax.plot(x_range, y_theory, color='red', lw=2, label='Теория')


        sns.kdeplot(sample, ax=ax, bw_adjust=adj, color='blue', label=f'KDE')

        ax.set_xlim(x_lim)
        ax.set_title(f"h = {adj}h")
        ax.grid(True, ls='--', alpha=0.7)
        ax.legend(fontsize='small')


        if adj < 1:
            ax.set_xlabel("Недосглаживание (зубчатость)")
        elif adj == 1:
            ax.set_xlabel("Стандарт (правило Скотта)")
        else:
            ax.set_xlabel("Пересглаживание (потеря формы)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    study_cauchy_bandwidth()

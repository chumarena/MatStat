import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

SIZES = [20, 60, 100]


def get_distributions():
    s2, s3 = np.sqrt(2), np.sqrt(3)
    return {
        "Normal": (stats.norm(0, 1), [-4, 4], "Normal $N(0, 1)$"),
        "Cauchy": (stats.cauchy(0, 1), [-4, 4], "Cauchy $C(0, 1)$"),
        "Laplace": (stats.laplace(0, 1 / s2), [-4, 4], r"Laplace $L(0, 1/\sqrt{2})$"),
        "Poisson": (stats.poisson(5), [0, 14], "Poisson $P(5)$"),
        "Uniform": (stats.uniform(-s3, 2 * s3), [-4, 4], r"Uniform $U(-\sqrt{3}, \sqrt{3})$")
    }


def plot_dist(dist, x_lim, title, is_cdf=True):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{title}", fontsize=14)
    x_range = np.linspace(*x_lim, 1000)

    for i, n in enumerate(SIZES):
        sample = dist.rvs(size=n)
        ax = axes[i]

        if is_cdf:
            # Теоретическая CDF и ЭФР
            ax.plot(x_range, dist.cdf(x_range), 'red', lw=2, label='Теория')
            x_sorted = np.sort(sample)
            x_plot = np.concatenate(([x_lim[0]], x_sorted, [x_lim[1]]))
            y_plot = np.concatenate(([0], np.arange(1, n + 1) / n, [1]))
            ax.step(x_plot, y_plot, where='post', color='blue', label='Эмпирическое')
        else:
            # Теоретическая плотность и KDE
            if hasattr(dist, 'pmf'):
                x_p = np.arange(x_lim[0], x_lim[1] + 1)
                ax.step(x_p, dist.pmf(x_p), where='mid', color='red', lw=2, label='Теория')
            else:
                ax.plot(x_range, dist.pdf(x_range), 'red', lw=2, label='Теория')
            sns.kdeplot(sample, ax=ax, bw_adjust=1, color='blue', label='Ядерное')

        ax.set(xlim=x_lim, title=f"n = {n}")
        ax.grid(True, ls='--', alpha=0.7)
        ax.legend(fontsize='small')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    dists = get_distributions()
    for name, (dist, x_lim, title) in dists.items():
        plot_dist(dist, x_lim, title, is_cdf=True)
        plot_dist(dist, x_lim, title, is_cdf=False)
from CAT.visual.get_correlation import run_get_correlation, setting_info
import matplotlib.pyplot as plt
import numpy as np

def run_figure1():
    # Data collection
    num_dim = [8, 4, 2, 1, 16, 32]
    pl = [1, 2, 3]
    pearson_corr = [[0.0 for _ in range(len(pl))] for _ in range(len(num_dim))]
    spearman_corr = [[0.0 for _ in range(len(pl))] for _ in range(len(num_dim))]
    kendall_corr = [[0.0 for _ in range(len(pl))] for _ in range(len(num_dim))]
    for i in range(len(setting_info)):
        dim_index = num_dim.index(setting_info[i]['num_dim'])
        pl_index = pl.index(setting_info[i]['pl'])
        pearson_corr[dim_index][pl_index], spearman_corr[dim_index][pl_index], kendall_corr[dim_index][pl_index] = run_get_correlation(i)

    # Plotting
    x = np.arange(len(num_dim))  # Num_dim indices
    width = 0.25  # Width of each bar

    for i in range(len(pl)):
        fig, ax = plt.subplots()
        ax.bar(x - width, [row[i] for row in pearson_corr], width, label='Pearson')
        ax.bar(x, [row[i] for row in spearman_corr], width, label='Spearman')
        ax.bar(x + width, [row[i] for row in kendall_corr], width, label='Kendall')

        ax.set_xlabel('num_dim')
        ax.set_ylabel('Correlation Coefficient')
        ax.set_title(f'Correlation Coefficients for pl={pl[i]}')
        ax.set_xticks(x)
        ax.set_xticklabels(num_dim)
        ax.legend()

        plt.show()

def run_figure_corr(correlation_type='pearson'):  # pearson, spearman, kendall, default is pearson
    # Data collection
    num_dim = [8, 4, 2, 1, 16, 32]
    pl = [1, 2, 3]
    corr = [[0.0 for _ in range(len(pl))] for _ in range(len(num_dim))]
    for i in range(len(setting_info)):
        dim_index = num_dim.index(setting_info[i]['num_dim'])
        pl_index = pl.index(setting_info[i]['pl'])
        if correlation_type == 'pearson':
            corr[dim_index][pl_index], _, _ = run_get_correlation(i)
        elif correlation_type == 'spearman':
            _, corr[dim_index][pl_index], _ = run_get_correlation(i)
        elif correlation_type == 'kendall':
            _, _, corr[dim_index][pl_index] = run_get_correlation(i)
    # Plotting
    x = np.arange(len(num_dim))  # Num_dim indices
    width = 0.25  # Width of each bar
    fig, ax = plt.subplots()
    for i in range(len(pl)):
        ax.bar(x + i * width, [row[i] for row in corr], width, label=f'pl={pl[i]}')
    ax.set_xlabel('num_dim')
    ax.set_ylabel(f'{correlation_type.capitalize()} Correlation Coefficient')
    ax.set_title(f'{correlation_type.capitalize()} Correlation Coefficients')
    ax.set_xticks(x)
    ax.set_xticklabels(num_dim)
    ax.legend()
    plt.show()


if __name__ == '__main__':
    run_figure1()
    # run_figure_corr('pearson')
    # run_figure_corr('spearman')
    # run_figure_corr('kendall')

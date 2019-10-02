from grand import utils
import matplotlib.pylab as plt, numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def axs_to_array(axs):
    return axs if isinstance(axs, (np.ndarray)) else np.array([axs])


def plot_deviations(df_data, df_representatives, df_dev_contexts, dev_threshold, figsize=None, savefig=None, plots=None):
    plots = ["data", "strangeness", "pvalue", "deviation", "threshold"] if plots is None else list(set(plots))
    nb_axs = len(plots) - len([s for s in plots if s in ["pvalue", "deviation", "threshold"]]) + 1

    fig, axes = plt.subplots(nb_axs, sharex="row", figsize=figsize)
    axes = axes if isinstance(axes, (np.ndarray)) else np.array([axes])

    i = 0
    if "data" in plots:
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Feature")
        df_data.plot(ax=axes[i])
        df_representatives.plot(ax=axes[i], color="grey", linestyle="--")
        i += 1

    if "strangeness" in plots:
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Strangeness")
        df_dev_contexts["strangeness"].plot(ax=axes[i])
        i += 1

    if any(s in ["pvalue", "deviation", "threshold"] for s in plots):
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Deviation level")
        axes[i].set_ylim(0, 1)
        if "pvalue" in plots:
            df = df_dev_contexts["pvalue"]
            axes[i].scatter(df.index, df.values, alpha=0.25, marker=".", color="green", label="p-value")
        if "deviation" in plots:
            df = df_dev_contexts["deviation"]
            axes[i].plot(df.index, df.values, label="deviation")
        if "threshold" in plots:
            axes[i].axhline(y=dev_threshold, color='red', linestyle='--')
        axes[i].legend()

    fig.autofmt_xdate()

    if savefig is None:
        plt.draw()
        plt.show()
    else:
        figpathname = utils.create_directory_from_path(savefig)
        plt.savefig(figpathname)

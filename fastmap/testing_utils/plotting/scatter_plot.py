from collections.abc import Iterable
from plot import Plot
import matplotlib.pyplot as plt
import seaborn as sns


class ScatterPlot(Plot):
    def __init__(self, x: Iterable, y: Iterable, xlabel: str = "x", ylabel: str = "y", xlim: tuple[float, float] | None = None, ylim: tuple[float, float] | None = None) -> None:
        super().__init__(x, y, xlabel, ylabel, xlim, ylim)

    def plot(self, show: bool = False, save: str | None = None):

        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=self._x, y=self._y, color='purple', s=1, alpha=0.5)

        plt.xlim(self._xlim)
        plt.ylim(self._ylim)
        plt.xlabel(self._xlabel, fontweight='medium')
        plt.ylabel(self._ylabel, fontweight='medium')

        if show:
            plt.show()

        if save:
            plt.savefig(save)
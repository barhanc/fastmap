from abc import ABC
from collections.abc import Iterable


class Plot(ABC):
    def __init__(self, x: Iterable, y: Iterable, xlabel: str = "x", ylabel: str = "y", xlim: tuple[float, float] | None = None, ylim: tuple[float, float] | None = None) -> None:
        self._x: Iterable = x
        self._y: Iterable = y
        self._xlabel: str = xlabel
        self._ylabel: str = ylabel
        xlim = xlim or (min(x), max(x))
        self._xlim: tuple[float, float] = self.__prepare_limit(x, xlim)
        self._ylim: tuple[float, float] = self.__prepare_limit(y, ylim)

    def plot(self, show: bool = False, save: bool = False, path: str = None):
        pass

    def __prepare_limit(self, data: Iterable, limit: tuple[float, float] | None, padding: float = 0.05) -> tuple[float, float]:
        diff: float = max(data) - min(data)
        if limit:
            return limit[0] - padding * diff, limit[1] + padding * diff
        return min(data) - padding * diff, max(data) + padding * diff
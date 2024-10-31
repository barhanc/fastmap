from abc import ABC, abstractmethod
from collections.abc import Callable
from time import perf_counter
from typing import Any

from fastmap.testing_utils.testing.result import Result
from fastmap.testing_utils.testing.results import Results
from fastmap.testing_utils.testing.utils import T, V


class TestCase(ABC):
    def __init__(self, 
                 f1: Callable[[T], V], 
                 f2: Callable[[T], V],
                 f1_return_mapper: Callable[[V], Any] = lambda x: x,
                 f2_return_mapper: Callable[[V], Any] = lambda x: x,
                 f1_name: str | None = None, 
                 f2_name: str | None = None) -> None:
        self._function_1: Callable[[T], V] = f1
        self._function_2: Callable[[T], V] = f2
        self._f1_return_mapper: Callable[[V], Any] = f1_return_mapper
        self._f2_return_mapper: Callable[[V], Any] = f2_return_mapper
        self._function_1_name: str = f1_name or f1.__name__
        self._function_2_name: str = f2_name or f2.__name__

    @abstractmethod
    def run(self, data: T) -> Results[V]:
        """Run the test case and return the results."""
        pass
    
    @property
    def function_1_name(self) -> str:
        """Return the name of function 1."""
        return self._function_1_name
    
    @property
    def function_2_name(self) -> str:
        """Return the name of function 2."""
        return self._function_2_name
    
    @property
    def type(self) -> str:
        """Return the type of the test."""
        return self.__class__.__name__

    def _time_func(self, func: Callable[[T], V], data: T) -> Result[V]:
        """Helper function to measure the time of a function."""
        start: float = perf_counter()
        result: V = func(*data)
        elapsed_time: float = perf_counter() - start
        return Result(time=elapsed_time, result=result)
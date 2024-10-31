from collections.abc import Callable
from typing import Any
from test_case import TestCase

from fastmap.utils.testing.result import Result
from fastmap.utils.testing.results import Results
from testing_utils import T, V


class EqualityTestCase(TestCase):
    
    def __init__(self, 
                 f1: Callable[[T], V], 
                 f2: Callable[[T], V], 
                 f1_return_mapper: Callable[[V], Any] = lambda x: x, 
                 f2_return_mapper: Callable[[V], Any] = lambda x: x, 
                 f1_name: str | None = None, 
                 f2_name: str | None = None, 
                 epsilon: float = 1e-6) -> None:
        super().__init__(f1, f2, f1_name, f2_name)
        self._f1_mapper = f1_return_mapper
        self._f2_mapper = f2_return_mapper
        self.__epsilon: float = epsilon

    def run(self, data: T) -> Results:
        result1: Result[V] = self._time_func(self._function_1, data)
        result2: Result[V] = self._time_func(self._function_2, data)
        assert abs(self._f1_mapper(result1.result) - self._f1_mapper(result2.result)) < self.__epsilon, f"{self.function_1_name} != {self.function_2_name}"
        return Results(time1=result1.time, time2=result2.time)

from collections.abc import Callable
from typing import Any, override
from testing.test_case import TestCase

from testing.result import Result
from testing.results import Results
from testing.utils import T, V

class ApproxRatioTestCase(TestCase):
    def __init__(self, 
                 f1: Callable[[T], V], 
                 f2: Callable[[T], V], 
                 f1_return_mapper: Callable[[V], Any] = lambda x: x,
                 f2_return_mapper: Callable[[V], Any] = lambda x: x,
                 f1_name: str = "Function 1", 
                 f2_name: str = "Function 2", 
                 ) -> None:
        super().__init__(f1, f2, f1_return_mapper, f2_return_mapper, f1_name, f2_name)

    @override
    def run(self, data: T) -> Results:
        result1: Result[V] = self._time_func(self._function_1, data)
        result2: Result[V] = self._time_func(self._function_2, data)
        approx_ratio: float = max(self._f1_return_mapper(result1.result)/self._f2_return_mapper(result2.result), self._f2_return_mapper(result2.result)/self._f1_return_mapper(result1.result))
        return Results(time1=result1.time, time2=result2.time, result=approx_ratio)
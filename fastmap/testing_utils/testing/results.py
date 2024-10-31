from dataclasses import dataclass
from typing import Generic, TypeVar

from fastmap.testing_utils.testing.result import Result

R = TypeVar('R')


@dataclass
class Results(Generic[R]):
    time1: float
    time2: float
    result: R = None
    failed: bool = False

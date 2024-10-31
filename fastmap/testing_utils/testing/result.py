from dataclasses import dataclass
from typing import Generic, TypeVar

R = TypeVar('R')


@dataclass(frozen=True)
class Result(Generic[R]):
    time: float
    result: R

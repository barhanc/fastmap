

from collections.abc import Sequence
from fastmap.testing.test import Test
from fastmap.testing.utils import fail, success


class OverallTest:
    def __init__(self, tests: Sequence[Test], verbose: bool = False, time: bool = False) -> None:
        self.__tests: Sequence[Test] = tests
        self.__passed_tests: int = 0
        self.__failed_tests: int = 0
        self.__verbose: bool = verbose
        self.__log_time: bool = time

    def run(self) -> None:
        for test in self.__tests:
            self.__run_test(test)
        result: str = f'Tests passed: {self.__passed_tests}/{len(self.__tests)}'
        if self.__failed_tests:
            print(fail(result))
        else:
            print(success(result))

    def __run_test(self, test: Test) -> None:
        try:
            test.run()
            self.__passed_tests += 1
            if self.__verbose:
                print(success(f'Test {test.type}({test.function_1_name}, {test.function_2_name}) passed'))
        except Exception as e:
            print(fail(f'Test {test.type}({test.function_1_name}, {test.function_2_name}) failed'))
            if self.__verbose:
                print(f'Error: {e}')
            self.__failed_tests += 1
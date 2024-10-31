from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

from collections.abc import Iterable

from .csv_props import CSVProps
from .results import Results
from .test_case import TestCase
from .time_graph_props import TimeGraphProps
from .utils import T, V


class Test:
    def __init__(self, test: Test | TestCase, data: Iterable[T], time_graph_props: TimeGraphProps | None = None, csv_props: CSVProps | None = None) -> None:
        self.__test: Test | TestCase = test
        self.__data: Iterable[T] = data
        self.__time_graph_props: TimeGraphProps | None = time_graph_props
        self.__csv_props: CSVProps | None = csv_props

    def run(self) -> list[Results[V]] | list[list[Results[V]]]:
        """Run the tests and return the results."""
        results: list[Results[V]] | list[list[Results[V]]] = [self.__test.run(d) for d in self.__data]
        self.__save_time_graph(results)
        self.__save_csv(results)
        return results
    
    @property
    def function_1_name(self) -> str:
        """Return the name of function 1."""
        return self.__test.__function_1_name
    
    @property
    def function_2_name(self) -> str:
        """Return the name of function 2."""
        return self.__test.__function_2_name
    
    @property
    def type(self) -> str:
        """Return the type of the test."""
        return self.__test.type
    
    def __save_time_graph(self, results: list[Results[V]] | list[list[Results[V]]]) -> None:
        """Save the time graph of the tested functions."""
        if not self.__time_graph_props:
            return
        
        if all(isinstance(res, list) for res in results):
            mean_times1 = [np.mean([res.result1.time for res in result_group]) for result_group in results]
            mean_times2 = [np.mean([res.result2.time for res in result_group]) for result_group in results]
        else:
            mean_times1 = [np.mean(res.result1.time) for res in results]
            mean_times2 = [np.mean(res.result2.time) for res in results]

        plt.figure(figsize=(12, 6))

        plt.plot(self.__time_graph_props.steps, mean_times1, label=self.function_1_name, color='blue')
        
        plt.plot(self.__time_graph_props.steps, mean_times2, label=self.function_2_name, color='orange')

        plt.title(self.__time_graph_props.name or 'Execution Times Comparison')
        plt.xlabel(self.__time_graph_props.xlabel)
        plt.ylabel(self.__time_graph_props.ylabel or 'Time (s)')
        plt.legend(self.function_1_name, self.function_2_name)
        plt.grid()

        plt.tight_layout()
        plt.savefig(self.__time_graph_props.save_path)
        plt.close()

    # TODO: This method needs to be improved
    def __save_csv(self, results: list[Results[V]] | list[list[Results[V]]]) -> None:
        """Save results to a CSV file."""
        if not self.__csv_props:
            return
        
        with open(self.__csv_props.file_path, self.__csv_props.file_mode) as file:
            file.write(f'{self.function_1_name}{self.__csv_props.delimiter}{self.function_2_name}\n')
            for res in results:
                file.write(f'{res.result1.result}{self.__csv_props.delimiter}{res.result2.result}\n')
    


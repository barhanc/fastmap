import os
import importlib.util
import argparse
from pathlib import Path
from testing.overall_test import OverallTest
from testing.test import Test

def autodiscover_and_run_tests(verbose: bool, time: bool):
    """Discover and run Test instances."""
    tests: list[Test] = []
    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith('.py') and file != __file__:
                file_path = os.path.join(root, file)
                find_tests_in_module(file_path, tests)
    overall_test = OverallTest(tests, verbose, time)
    overall_test.run()

def find_tests_in_module(file_path: str, tests: list[Test]):
    """Find Test instances in a module and add them to the tests list."""
    module_name = Path(file_path).stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name, obj in module.__dict__.items():
        if isinstance(obj, Test):
            tests.append(obj)

def main():
    parser = argparse.ArgumentParser(description='Discover and run tests.')
    parser.add_argument('--verbose', action='store_true', help='Increase output verbosity')
    parser.add_argument('--time', action='store_true', help='Show time differences')
    args = parser.parse_args()

    autodiscover_and_run_tests(verbose=args.verbose, time=args.time)

if __name__ == '__main__':
    main()

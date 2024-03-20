import sys

sys.path.append("./fastmap/fast/Fastor/")
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

__version__ = "0.0.1"

setup(
    name="fastmap",
    version=__version__,
    license="MIT",
    description="Optimized implementations of algorithms computing distance measures for Maps of Elections.",
    packages=["fastmap"],
    install_requires=[
        "pybind11",
        "numpy",
        "scipy",
        "tqdm",
        "cvxpy",  # For development only (or not if You come up with a nice MILP form.)
        "mapel",  # For development only
    ],
    python_requires=">=3.11",
    ext_modules=[
        Pybind11Extension(
            "fastmap.fast",
            sources=["fastmap/fast/bfcm.cpp", "fastmap/fast/lap/lap.cpp"],
            extra_compile_args=["-std=c++17", "-Ofast", "-Wall"],
            define_macros=[("VERSION_INFO", __version__)],
        ),
    ],
)

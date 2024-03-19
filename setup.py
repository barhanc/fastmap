from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


setup(
    name="fastmap",
    version="0.0.1",
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
            "fastmap.bfcm",
            ["fastmap/bfcm/rectangular_lsap/rectangular_lsap.cpp", "fastmap/bfcm/bfcm.cpp"],
            extra_compile_args=["-std=c++17", "-Wall", "-Ofast"],
        ),
    ],
)

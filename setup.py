from setuptools import setup, find_packages, Extension
import numpy

__version__ = "0.0.1"

setup(
    name="fastmap",
    version=__version__,
    packages=find_packages(),
    license="MIT",
    description="Optimized implementations of algorithms computing distance measures for Maps of Elections.",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    install_requires=[
        "pybind11",
        "numpy",
        "scipy",
        "tqdm",
        "cvxpy",  # For development only (or not if You come up with a nice MILP form, doesnt work on Mac OS)
        "mapel",  # For development only
    ],
    python_requires=">=3.11",
    ext_modules=[
        Extension(
            name="fastmap.bfcm",
            sources=[
                "fastmap/lap/lap.c",
                "fastmap/module.c",
            ],
            extra_compile_args=[
                "-Wall",
                "-Ofast",
                "-fopenmp",
            ],
            extra_link_args=["-lgomp"],
        )
    ],
    include_dirs=[numpy.get_include()],
)

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

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
            sources=[
                "fastmap/fast/lap/lap.cpp",
                "fastmap/fast/module.cpp",
            ],
            extra_compile_args=["-O2"],
            include_dirs=[],
        ),
    ],
    cmdclass={"build_ext": build_ext},
)

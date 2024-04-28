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
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "tqdm",
        "cvxpy",  # For development only
        "mapel",  # For comparison only
    ],
    python_requires=">=3.10",
    ext_modules=[
        Extension(
            name="fastmap._spear",
            sources=["fastmap/example/pyspear.c"],
            extra_compile_args=["-Ofast"],
            include_dirs=["fastmap/lap/", "fastmap/bap/"],
        ),
        Extension(
            name="fastmap._hamm",
            sources=["fastmap/example/pyhamm.c"],
            extra_compile_args=["-Ofast"],
            include_dirs=["fastmap/lap/", "fastmap/bap/"],
        ),
        Extension(
            name="fastmap._swap",
            sources=["fastmap/example/pyswap.c"],
            extra_compile_args=["-Ofast"],
            include_dirs=["fastmap/lap/"],
        ),
    ],
    include_dirs=[numpy.get_include()],
)

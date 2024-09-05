import numpy as np

from setuptools import setup, find_packages, Extension
from numpy.core._multiarray_umath import __cpu_features__ as cpu_has

__version__ = "0.0.1"

CFLAGS = ["-Ofast"]

if cpu_has["AVX2"]:
    fastmap_lap = "fastmap/lap/lapavx2"
    CFLAGS.append("-mavx2")
else:
    fastmap_lap = "fastmap/lap/lapbase"

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
    ],
    python_requires=">=3.10",
    extras_require={
        "testing": [
            "mapel",
            "tqdm",
        ]
    },
    ext_modules=[
        Extension(
            name="fastmap._spear",
            sources=["fastmap/pyspear.c"],
            extra_compile_args=CFLAGS,
            include_dirs=[fastmap_lap, "fastmap/bap/", "fastmap/utils/"],
        ),
        Extension(
            name="fastmap._hamm",
            sources=["fastmap/pyhamm.c"],
            extra_compile_args=CFLAGS,
            include_dirs=[fastmap_lap, "fastmap/bap/", "fastmap/utils/"],
        ),
        Extension(
            name="fastmap._swap",
            sources=["fastmap/pyswap.c"],
            extra_compile_args=CFLAGS,
            include_dirs=[fastmap_lap],
        ),
        Extension(
            name="fastmap._pairwise",
            sources=["fastmap/pypairwise.c"],
            extra_compile_args=CFLAGS,
            include_dirs=[fastmap_lap, "fastmap/qap/"],
        ),
    ],
    include_dirs=[np.get_include()],
)

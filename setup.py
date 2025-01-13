import numpy as np

from setuptools import setup, find_packages, Extension
from numpy._core._multiarray_umath import __cpu_features__ as cpu_has

__version__ = "0.0.1"

CFLAGS = ["-Ofast"]

if cpu_has["AVX2"]:
    CFLAGS.append("-mavx2")

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
            "pytest",
            "mapel",
            "tqdm",
        ]
    },
    ext_modules=[
        Extension(
            name="fastmap._spear",
            sources=["fastmap/pyspear.c"],
            extra_compile_args=CFLAGS,
            include_dirs=["fastmap/lap/", "fastmap/bap/", "fastmap/utils/"],
        ),
        Extension(
            name="fastmap._hamm",
            sources=["fastmap/pyhamm.c"],
            extra_compile_args=CFLAGS,
            include_dirs=["fastmap/lap/", "fastmap/bap/", "fastmap/utils/"],
        ),
        Extension(
            name="fastmap._swap",
            sources=["fastmap/pyswap.c"],
            extra_compile_args=CFLAGS,
            include_dirs=["fastmap/lap/"],
        ),
        Extension(
            name="fastmap._pairwise",
            sources=["fastmap/pypairwise.c"],
            extra_compile_args=CFLAGS,
            include_dirs=["fastmap/lap/", "fastmap/qap/", "fastmap/utils/"],
        ),
        Extension(
            name="fastmap._swap_distance_between_potes",
            sources=["fastmap/kkemeny/swap_distance_between_potes.c"],
            extra_compile_args=CFLAGS,
        ),
        Extension(
            name="fastmap._local_search_kkemeny_single_k",
            sources=["fastmap/kkemeny/local_search_kkemeny_single_k.c"],
            extra_compile_args=CFLAGS,
            include_dirs=["fastmap/kkemeny/"],
        ),
        Extension(
            name="fastmap._simulated_annealing",
            sources=["fastmap/kkemeny/simulated_annealing.c"],
            extra_compile_args=CFLAGS,
        ),
    ],
    include_dirs=[np.get_include()],
)

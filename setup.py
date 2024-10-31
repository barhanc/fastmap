import numpy as np

from setuptools import setup, find_packages, Extension
from numpy.core._multiarray_umath import __cpu_features__ as cpu_has

__version__ = "0.0.1"

CFLAGS = ["-Ofast"]

if cpu_has["AVX2"]:
    CFLAGS.append("-mavx2")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fastmap",
    version=__version__,
    packages=find_packages(include=["fastmap", "testing_utils"]),
    license="MIT",
    description="Optimized implementations of algorithms computing distance measures for Maps of Elections.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
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
            name="fastmap._agreement_index",
            sources=["fastmap/agreement_index.c"],
            extra_compile_args=CFLAGS,
            include_dirs=["fastmap/kkemeny/"],
        ),
        Extension(
            name="fastmap._polarization_index",
            sources=["fastmap/polarization_index.c"],
            extra_compile_args=CFLAGS,
            include_dirs=["fastmap/kkemeny/"],
        ),
        Extension(
            name="fastmap._diversity_index",
            sources=["fastmap/diversity_index.c"],
            extra_compile_args=CFLAGS,
            include_dirs=["fastmap/kkemeny/"],
        ),
        Extension(
            name="fastmap._kemeny_ranking",
            sources=["fastmap/kemeny_ranking.c"],
            extra_compile_args=CFLAGS,
            include_dirs=["fastmap/kkemeny/"]
        ),
        Extension(
            name="fastmap._local_search_kkemeny_single_k",
            sources=["fastmap/local_search_kkemeny_single_k.c"],
            extra_compile_args=CFLAGS,
            include_dirs=["fastmap/kkemeny/"]
        ),
        Extension(
            name="fastmap._local_search_kkemeny",
            sources=["fastmap/local_search_kkemeny.c"],
            extra_compile_args=CFLAGS,
            include_dirs=["fastmap/kkemeny/"]
        ),
        Extension(
            name="fastmap._polarization_1by2Kemenys",
            sources=["fastmap/polarization_1by2Kemenys.c"],
            extra_compile_args=CFLAGS,
            include_dirs=["fastmap/kkemeny/"]
        ),
        Extension(
            name="fastmap._greedy_kmeans_summed",
            sources=["fastmap/greedy_kmeans_summed.c"],
            extra_compile_args=CFLAGS,
            include_dirs=["fastmap/kkemeny/"]
        ),
        Extension(
            name="fastmap._greedy_kKemenys_summed",
            sources=["fastmap/greedy_kkemenys_summed.c"],
            extra_compile_args=CFLAGS,
            include_dirs=["fastmap/kkemeny/"]
        ),
        Extension(
            name="fastmap._greedy_2kKemenys_summed",
            sources=["fastmap/greedy_2kKemenys_summed.c"],
            extra_compile_args=CFLAGS,
            include_dirs=["fastmap/kkemeny/"]
        ),
        Extension(
            name="fastmap._greedy_kKemenys_divk_summed",
            sources=["fastmap/greedy_kKemenys_divk_summed.c"],
            extra_compile_args=CFLAGS,
            include_dirs=["fastmap/kkemeny/"]
        ),
    ],
    include_dirs=[np.get_include()],
    options={"build_ext": {"verbose": True}}
)

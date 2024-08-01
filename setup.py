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
            extra_compile_args=["-Ofast"],
            include_dirs=["fastmap/lap/", "fastmap/bap/", "fastmap/utils/"],
        ),
        Extension(
            name="fastmap._hamm",
            sources=["fastmap/pyhamm.c"],
            extra_compile_args=["-Ofast"],
            include_dirs=["fastmap/lap/", "fastmap/bap/", "fastmap/utils/"],
        ),
        Extension(
            name="fastmap._swap",
            sources=["fastmap/pyswap.c"],
            extra_compile_args=["-Ofast"],
            include_dirs=["fastmap/lap/"],
        ),
        Extension(
            name="fastmap._pairwise",
            sources=["fastmap/pypairwise.c"],
            extra_compile_args=["-Ofast"],
            include_dirs=["fastmap/lap/", "fastmap/qap/"],
        ),
    ],
    include_dirs=[numpy.get_include()],
)

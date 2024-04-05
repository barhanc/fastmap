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
        "numpy",
        "scipy",
        "tqdm",
        "cvxpy",  # For development only (doesnt work on Mac OS)
        "mapel",  # For comparison only
    ],
    python_requires=">=3.10",
    ext_modules=[
        Extension(
            name="fastmap.bfcm",
            sources=["fastmap/bfcm.c"],
            extra_compile_args=["-Ofast"],
            include_dirs=["fastmap/lap/"],
            # extra_link_args=["-lgomp"],
        )
    ],
    include_dirs=[numpy.get_include()],
)

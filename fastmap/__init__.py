"""
Fastmap is a small Python library that provides optimized C implementations (as well as convenient
Python wrappers) of algorithms computing structural similarity between elections used in the Map of
Elections framework, such as the isomorphic swap and Spearman distances, Hamming distance, pairwise
distance as well as distances based on diversity, agreement and polarization of the elections.
"""

from fastmap._wrapper import spearman
from fastmap._wrapper import hamming
from fastmap._wrapper import swap
from fastmap._wrapper import pairwise

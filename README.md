# Brief description

The map of elections framework is a way of visualizing datasets of elections. A single election E =
(C,V) consists of a set of candidates C and a collection of voters V, where each voter expresses
preferences about the candidates (in the ordinal setting, the voters rank the candidates from the
most to the least appealing ones; in the approval setting they indicate which of the candidates are
sufficiently good). A map of election is a way of presenting a set of elections as points in 2D
plane, so that the Euclidean distances between the elections correspond to their structural
similarity.

There is a number of ways of computing structural similarity of elections, such as using the
isomorphic swap or Spearman distances, using the Hamming/Jaccard-based distances, using distances
based on diversity, agreement, and polarization of the elections and so on. The goal of this work is
to provide highly optimized implementations of these algorithms. The problem is that the distance
computation is often NP-hard, so we need to optimize brute-force algorithms and seek heuristic
solutions (which might be "good enough" to get initial maps of elections).

The goal of this project is to provide such highly optimized implementations, as well as heuristics
for computing the distances. The implementation---if sufficiently good---will become part of the
Python MapEl package.

# Installation

The current recommended way to install *fastmap* is from source. Note that you will need a C
compiler (GCC/Clang) installed to compile native CPython extensions which should be standard on
Linux and macOS. For Windows users we recommend using WSL2.
```shell
$ git clone https://github.com/barhanc/fastmap.git
$ cd fastmap
$ python3 -m venv venv
$ source venv/bin/activate
(venv) $ pip install -e .
```

# Running tests

```shell
(venv) $ pip install -e '.[testing]'
```
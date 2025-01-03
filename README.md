# Brief description

The map of elections framework is a way of visualizing datasets of elections. A single election
$E=(C,V)$ consists of a set of candidates $C$ and a collection of voters $V$, where each voter
expresses preferences about the candidates (in the ordinal setting, the voters rank the candidates
from the most to the least appealing ones; in the approval setting they indicate which of the
candidates are sufficiently good). A map of election is a way of presenting a set of elections as
points in 2D plane, so that the Euclidean distances between the elections correspond to their
structural similarity.

There is a number of ways of computing structural similarity of elections, such as using the
isomorphic swap or Spearman distances, using the Hamming/Jaccard-based distances, using distances
based on diversity, agreement, and polarization of the elections and so on. The goal of this work is
to provide highly optimized implementations of these algorithms. The problem is that the distance
computation is often NP-hard, so we need to optimize brute-force algorithms and seek heuristic
solutions (which might be "good enough" to get initial maps of elections).

The goal of this project is to provide such highly optimized implementations, as well as heuristics
for computing the distances.

# Installation

The current recommended way to install *fastmap* is from source. Note that you will need a C
compiler (GCC/Clang) installed to compile native CPython extensions which should be available by
default on Linux and macOS. For Windows users we recommend using WSL2.

## From source

```shell
git clone https://github.com/barhanc/fastmap.git
cd fastmap
python3 -m venv venv
source venv/bin/activate
pip install -e .
```
## Direct (main)

```shell
pip install git+https://github.com/barhanc/fastmap.git
```

# Benchmarking

In the [example.py](/examples/example.py) file we provide a minimal reproducible example which
generates a map of elections with 8 candidates and 96 voters that requires computing 46056
isomorphic swap distances. In the code we use the `concurrent.futures` module to utilze multiple
cores when computing the distances and use the `sklearn.manifold.MDS` method to obtain the 2D
embedding. We provide the elections' matrices generated using `mapel.elections` module in the
`args.pickle` file to allow full reproducibility. Below we show the resulting maps and the time
required to generate them for the brute force method and the heuristic AA method.

![alt text](/examples/map9658.png "Map of elections using fastmap BF implementation")

*Map of elections generated using fastmap BF implementation | Time: ~2h (Intel i9-12900K)*

![alt text](/examples/map4022.png "Map of elections using fastmap AA heuristic")

*Map of elections generated using fastmap AA heuristic | Time: ~1min (Intel i9-12900K) / ~8min (Apple M1)*

As we can see the maps are nearly identical while the computing time using the implemented heuristic
is vastly smaller. For comparison we also include the map generated using Mapel implementation of
the brute-force algorithm and the corresponding time.

![alt text](/examples/map2758.png "Map of elections using Mapel BF implementation")

*Map of elections generated using Mapel BF implementation | Time: ~3h (Intel i9-12900K)*

The map is obviously exactly the same as the one generated using fastmap BF implementation but we
can see that the computing time is approximately 50% longer.
# from utils.testing.equality_test_case import EqualityTestCase
# from mapel.elections import generate_election_from_votes
# import mapel.elections.features.diversity as mapel
# import fastmap

# from utils.testing.data.data_generator import VotesGenerator

# Test

# making utils discoverable
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mapel.elections.features.diversity as mapel
from mapel.elections import generate_election_from_votes
import fastmap
from fastmap.testing_utils.testing.equality_test_case import EqualityTestCase

from fastmap.testing_utils.testing.test_case import TestCase
from fastmap.testing_utils.testing.equality_test_case import EqualityTestCase
from fastmap.testing_utils.testing.test import Test
from fastmap.testing_utils.data.data_generator import VotesGenerator


votes = [[0, 1, 2], [2, 1, 0], [0, 2, 1]]


agreement_index_test = EqualityTestCase(mapel.agreement_index, fastmap.agreement_index, lambda x: x['value'], lambda x: x['value'], "mapel.agreement_index", "fastmap.agreement_index")
r = agreement_index_test.run([generate_election_from_votes(votes)])
# print(r)

polarization_index_test = EqualityTestCase(mapel.polarization_index, fastmap.polarization_index, lambda x: x['value'], lambda x: x['value'], "mapel.polarization_index", "fastmap.polarization_index")
r = polarization_index_test.run([generate_election_from_votes(votes)])
# print(r)

diversity_index_test = EqualityTestCase(mapel.diversity_index, fastmap.diversity_index, lambda x: x['value'], lambda x: x['value'], "mapel.diversity_index", "fastmap.diversity_index")
r = diversity_index_test.run([generate_election_from_votes(votes)])
# print(r)

kemeny_ranking_test = EqualityTestCase(mapel.kemeny_ranking, fastmap.kemeny_ranking, lambda x: x[1], lambda x: x[1], "mapel.kemeny_ranking", "fastmap.kemeny_ranking")
r = kemeny_ranking_test.run([generate_election_from_votes(votes)])
print(r)

local_search_kKemeny_single_k_test = EqualityTestCase(mapel.local_search_kKemeny_single_k, fastmap.local_search_kKemeny_single_k, lambda x: x['value'], lambda x: x['value'], "mapel.local_search_kKemeny_single_k", "fastmap.local_search_kKemeny_single_k")
r = local_search_kKemeny_single_k_test.run((generate_election_from_votes(votes), 1, 1))
print(r)

local_search_kKemeny_test = EqualityTestCase(mapel.local_search_kKemeny, fastmap.local_search_kKemeny, lambda x: x['value'][0], lambda x: x['value'][0], "mapel.local_search_kKemeny", "fastmap.local_search_kKemeny")
r = local_search_kKemeny_test.run((generate_election_from_votes(votes), 1))
# print(r)

polarization_1by2Kemenys_test = EqualityTestCase(mapel.polarization_1by2Kemenys, fastmap.polarization_1by2Kemenys, lambda x: x['value'], lambda x: x['value'], "mapel.polarization_1by2Kemenys", "fastmap.polarization_1by2Kemenys")
r = polarization_1by2Kemenys_test.run([generate_election_from_votes(votes)])
# print(r)

greedy_kmeans_summed_test = EqualityTestCase(mapel.greedy_kmeans_summed, fastmap.greedy_kmeans_summed, lambda x: x['value'], lambda x: x['value'], "mapel.greedy_kmeans_summed", "fastmap.greedy_kmeans_summed")
r = greedy_kmeans_summed_test.run([generate_election_from_votes(votes)])
# print(r)

greedy_kKemenys_summed_test = EqualityTestCase(mapel.greedy_kKemenys_summed, fastmap.greedy_kKemenys_summed, lambda x: x, lambda x: x['value'], "mapel.greedy_kKemenys_summed", "fastmap.greedy_kKemenys_summed")
r = greedy_kKemenys_summed_test.run([generate_election_from_votes(votes)])
# print(r)

greedy_kKemenys_divk_summed_test = EqualityTestCase(mapel.greedy_kKemenys_divk_summed, fastmap.greedy_kKemenys_divk_summed, lambda x: x['value'], lambda x: x['value'], "mapel.greedy_kKemenys_divk_summed", "fastmap.greedy_kKemenys_divk_summed")
r = greedy_kKemenys_divk_summed_test.run([generate_election_from_votes(votes)])
# print(r)

greedy_2kKemenys_summed_test = EqualityTestCase(mapel.greedy_2kKemenys_summed, fastmap.greedy_2kKemenys_summed, lambda x: x['value'], lambda x: x['value'], "mapel.greedy_2kKemenys_summed", "fastmap.greedy_2kKemenys_summed")
r = greedy_2kKemenys_summed_test.run([generate_election_from_votes(votes)])
# print(r)

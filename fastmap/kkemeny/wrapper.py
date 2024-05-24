import ctypes
import random
import time
import numpy as np

num_voters = 20
num_candidates = 11

# Defining the 2D ctypes array type for votes
votes_ctypes_type = (ctypes.c_int * num_candidates) * num_voters

# Converting numpy data to this format
votes_ctypes = votes_ctypes_type()

votes_ptr_type = ctypes.POINTER(ctypes.POINTER(ctypes.c_int))
votes_ptr = (ctypes.POINTER(ctypes.c_int) * num_voters)()

votes = np.array(
    [
        [3, 6, 0, 7, 2, 4, 5, 1, 10, 8, 9],
        [2, 3, 9, 0, 5, 1, 10, 7, 8, 4, 6],
        [3, 6, 2, 5, 0, 1, 10, 9, 7, 4, 8],
        [5, 0, 7, 3, 6, 8, 9, 1, 4, 2, 10],
        [2, 4, 1, 9, 0, 10, 8, 6, 5, 3, 7],
        [9, 4, 2, 1, 10, 5, 6, 0, 7, 8, 3],
        [10, 3, 2, 0, 5, 6, 8, 4, 7, 9, 1],
        [9, 8, 2, 0, 4, 7, 1, 3, 10, 5, 6],
        [5, 1, 0, 3, 2, 4, 6, 10, 9, 8, 7],
        [10, 7, 6, 4, 0, 9, 2, 8, 1, 3, 5],
        [9, 7, 3, 0, 4, 8, 1, 2, 5, 10, 6],
        [6, 4, 5, 10, 1, 9, 7, 3, 8, 0, 2],
        [4, 10, 0, 8, 1, 3, 2, 7, 6, 9, 5],
        [3, 10, 2, 6, 9, 8, 1, 7, 5, 4, 0],
        [2, 6, 9, 8, 7, 10, 1, 4, 0, 5, 3],
        [0, 1, 7, 8, 5, 3, 4, 2, 10, 6, 9],
        [5, 8, 2, 3, 1, 7, 0, 4, 9, 6, 10],
        [2, 9, 8, 10, 3, 6, 1, 7, 4, 5, 0],
        [10, 9, 7, 0, 8, 4, 1, 6, 2, 5, 3],
        [10, 4, 2, 1, 9, 3, 0, 6, 7, 5, 8],
    ],
    dtype=int,
)

for i in range(num_voters):
    for j in range(num_candidates):
        votes_ctypes[i][j] = votes[i][j]

for i in range(num_voters):
    votes_ptr[i] = ctypes.cast(votes_ctypes[i], ctypes.POINTER(ctypes.c_int))

lib_path = "./libkemeny.so"
lib = ctypes.CDLL(lib_path)

lib.create_pairwise_matrix.argtypes = [
    ctypes.c_int,  # num_voters
    ctypes.c_int,  # num_candidates
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),  # votes
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double))  # matrix
]
lib.create_pairwise_matrix.restype = None

lib.kemeny_ranking.argtypes = [
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.c_int,
]
lib.kemeny_ranking.restype = None

lib.polarization_index.argtypes = [
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),  # votes
    ctypes.c_int,  # num_voters
    ctypes.c_int   # num_candidates
]
lib.polarization_index.restype = ctypes.c_double

lib.agreement_index.argtypes = [
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),  # votes
    ctypes.c_int,  # num_voters
    ctypes.c_int   # num_candidates
]
lib.agreement_index.restype = ctypes.c_double

lib.diversity_index.argtypes = [
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),  # votes
    ctypes.c_int,  # num_voters
    ctypes.c_int   # num_candidates
]
lib.diversity_index.restype = ctypes.c_double

lib.local_search_kKemeny_single_k.argtypes = [
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),  # int **votes
    ctypes.c_int,                                  # int k
    ctypes.c_int,                                  # int l
    ctypes.c_int,                                  # int votes_num
    ctypes.POINTER(ctypes.c_int)                   # int *starting
]
lib.local_search_kKemeny_single_k.restype = ctypes.c_int

lib.local_search_kKemeny.argtypes = [
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),  # votes
    ctypes.c_int,  # num_voters
    ctypes.c_int,  # num_candidates
    ctypes.c_int,  # l
    ctypes.POINTER(ctypes.c_int)  # starting
]
lib.local_search_kKemeny.restype = ctypes.POINTER(ctypes.c_double)

lib.kemeny_ranking.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_int)), ctypes.c_int, ctypes.c_int]
lib.kemeny_ranking.restype = ctypes.c_double

def kemeny_ranking(votes):
    num_voters = len(votes)
    num_candidates = len(votes[0])
    
    # Convert votes to ctypes array
    votes_ctypes_type = (ctypes.POINTER(ctypes.c_int) * num_voters)
    votes_ctypes = votes_ctypes_type()
    
    for i in range(num_voters):
        row = (ctypes.c_int * num_candidates)(*votes[i])
        votes_ctypes[i] = ctypes.cast(row, ctypes.POINTER(ctypes.c_int))
    
    # Call the C function
    result = lib.kemeny_ranking(votes_ctypes, num_voters, num_candidates)

    return result

def local_search_kKemeny_single_k(votes, k, l, starting=None):
    num_voters = len(votes)
    num_candidates = len(votes[0])
    
    # Convert votes to ctypes array
    votes_ctypes_type = (ctypes.POINTER(ctypes.c_int) * num_voters)
    votes_ctypes = votes_ctypes_type()
    
    for i in range(num_voters):
        row = (ctypes.c_int * num_candidates)(*votes[i])
        votes_ctypes[i] = ctypes.cast(row, ctypes.POINTER(ctypes.c_int))
    
    # Convert starting to ctypes array or set to None
    if starting is not None:
        starting_ctypes = (ctypes.c_int * len(starting))(*starting)
    else:
        starting_ctypes = None
    
    # Call the C function
    result = lib.local_search_kKemeny_single_k(votes_ctypes, k, l, num_voters, starting_ctypes)
    
    return result

def local_search_kKemeny(votes, l, starting=None):
    num_voters = len(votes)
    num_candidates = len(votes[0])
    
    # Convert votes to ctypes array
    votes_ctypes_type = (ctypes.POINTER(ctypes.c_int) * num_voters)
    votes_ctypes = votes_ctypes_type()
    
    for i in range(num_voters):
        row = (ctypes.c_int * num_candidates)(*votes[i])
        votes_ctypes[i] = ctypes.cast(row, ctypes.POINTER(ctypes.c_int))
    
    # Convert starting to ctypes array or set to None
    if starting is not None:
        starting_ctypes = (ctypes.c_int * len(starting))(*starting)
    else:
        starting_ctypes = None
    
    # Call the C function
    result_ptr = lib.local_search_kKemeny(votes_ctypes, num_voters, num_candidates, l, starting_ctypes)
    
    # Convert the result to a Python list
    result = [result_ptr[i] for i in range(num_voters)]
    
    # Free the allocated memory for the result in C
    lib.free(result_ptr)
    
    return result

def agreement_index(votes):
    num_voters = len(votes)
    num_candidates = len(votes[0])
    
    # Convert votes to ctypes array
    votes_ctypes_type = (ctypes.POINTER(ctypes.c_int) * num_voters)
    votes_ctypes = votes_ctypes_type()
    
    for i in range(num_voters):
        row = (ctypes.c_int * num_candidates)(*votes[i])
        votes_ctypes[i] = ctypes.cast(row, ctypes.POINTER(ctypes.c_int))
    
    # Call the C function
    result = lib.agreement_index(votes_ctypes, num_voters, num_candidates)
    
    return result

def polarization_index(votes):
    num_voters = len(votes)
    num_candidates = len(votes[0])
    
    # Convert votes to ctypes array
    votes_ctypes_type = (ctypes.POINTER(ctypes.c_int) * num_voters)
    votes_ctypes = votes_ctypes_type()
    
    for i in range(num_voters):
        row = (ctypes.c_int * num_candidates)(*votes[i])
        votes_ctypes[i] = ctypes.cast(row, ctypes.POINTER(ctypes.c_int))
    
    # Call the C function
    result = lib.polarization_index(votes_ctypes, num_voters, num_candidates)
    
    return result

if __name__ == "__main__":

    print("-" * 50)
    def generate_unique_vote(num_candidates: int) -> list[int]:
        candidates = list(range(num_candidates))
        random.shuffle(candidates)
        return candidates


    def generate_votes(num_voters: int, num_candidates: int) -> list[list[int]]:
        votes = []
        for _ in range(num_voters):
            votes.append(generate_unique_vote(num_candidates))
        return votes

    # start_time_single_k = time.time()
    # res = local_search_kKemeny_single_k(votes, 2, 1, None)
    # end_time_single_k = time.time()
    # print(f"Result: {res}")
    # print(f"local_search_kKemeny_single_k: {end_time_single_k - start_time_single_k} seconds")

    # print("-" * 50)

    # start_time_local_search = time.time()
    # res = local_search_kKemeny(votes, 1, None)
    # end_time_local_search = time.time()
    # print(f"Result: {res}")
    # print(f"local_search_kKemeny: {end_time_local_search - start_time_local_search} seconds")

    # print("-" * 50)

    # start_time_polarization = time.time()
    # no_candidates = [2*i for i in range(1, 3)]
    # no_votes = 1000
    # vote_sets = [[generate_votes(no_votes, num_candidates) for _ in range(3)] for num_candidates in no_candidates]
    # for i in range(len(vote_sets)):
    #     for j in range(len(vote_sets[i])):
    #         polarization_index(vote_sets[i][j])
    #         print(f"bagno {i*len(vote_sets[i])+j}")

    no_candidates = [i for i in range(2, 5)]
    no_votes = 10
    vote_sets = [[generate_votes(no_votes, num_candidates) for _ in range(10)] for num_candidates in no_candidates]
    for i in range(len(vote_sets)):
        for j in range(len(vote_sets[i])):
            kemeny_ranking(vote_sets[i][j])
    end_time_polarization = time.time()
    # print(res)
    # print(f"Polarization: {end_time_polarization - start_time_polarization:.5f} seconds")

    # print("-" * 50)

    # start_time_ranking = time.time()
    # lib.kemeny_ranking(comparison_matrix, num_candidates)
    # end_time_ranking = time.time()
    # print(f"Kemeny Ranking: {end_time_ranking - start_time_ranking:.5f} seconds")

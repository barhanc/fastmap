import time
import numpy as np
import itertools


def pairwise_matrix(num_candidates, votes):
    votes_num = len(votes)
    matrix = np.zeros([num_candidates, num_candidates])
    for v in range(votes_num):
        for c1 in range(num_candidates):
            for c2 in range(c1 + 1, num_candidates):
                matrix[int(votes[v][c1])][int(votes[v][c2])] += 1
    for i in range(num_candidates):
        for j in range(i + 1, num_candidates):
            matrix[i][j] /= votes_num
            matrix[j][i] = 1.0 - matrix[i][j]
    return matrix


def kemeny_ranking(votes):
    m = len(votes[0])
    wmg = pairwise_matrix(m, votes)
    best_d = np.infty
    for test_ranking in itertools.permutations(list(range(m))):
        dist = 0
        for i in range(m):
            for j in range(i + 1, m):
                dist = dist + wmg[test_ranking[j], test_ranking[i]]
            if dist > best_d:
                break
        if dist < best_d:
            best = test_ranking
            best_d = dist
    print(best, best_d)
    return best, best_d

def dist_to_Kemeny_ranking(m, votes):
    return kemeny_ranking(m, votes)[1]/m

def local_search_kKemeny_single_k(votes, k, l, starting=None) -> int:
    if starting is None:
        starting = list(range(k))
    distances = calculate_vote_swap_dist(votes)

    n = len(votes)

    d = distances_to_rankings(starting, distances)

    iter = 0
    check = 0

    while(check):
        iter += 1
        rest = [i for i in range(n) if i not in starting]
        for j in range(l):
            starting, d, check = find_improvement(distances, d, starting, rest, n, k, j+1)
            if check:
                break
    return d


def calculate_vote_swap_dist(votes):
    votes_num = len(votes)
    distances = np.zeros((votes_num, votes_num))
    for v1 in range(votes_num):
        for v2 in range(votes_num):
            distances[v1][v2] = swap_distance_between_potes(votes[v1], votes[v2])
    return distances


def swap_distance_between_potes(v1, v2):
    swap_distance = 0
    for i, j in itertools.combinations(v1, 2):
        if (v1[i] > v1[j] and v2[i] < v2[j]) or (v1[i] < v1[j] and v2[i] > v2[j]):
            swap_distance += 1
    return swap_distance


def distances_to_rankings(rankings, distances):
    dists = distances[rankings]
    return np.sum(dists.min(axis=0))


def find_improvement(distances, d, starting, rest, n, k, l):
    for cut in itertools.combinations(range(k), l):
        for paste in itertools.combinations(rest, l):
            ranks = []
            j = 0
            for i in range(k):
                if i in cut:
                    ranks.append(paste[j])
                    j = j + 1
                else:
                    ranks.append(starting[i])
            if len(set(ranks)) == len(ranks):
                d_new = distances_to_rankings(ranks, distances)
                if d > d_new:
                    return ranks, d_new, True
    return starting, d, False


def greedy_kKemenys_summed(votes) -> dict:
    num_voters = len(votes)
    res = [0] * num_voters
    distances = calculate_vote_swap_dist(votes)
    best = np.argmin(distances.sum(axis=1))
    best_vec = distances[best]
    res[0] = best_vec.sum()
    distances = np.vstack((distances[:best], distances[best + 1:]))

    for i in range(1, num_voters):
        relatives = distances - best_vec
        relatives = relatives * (relatives < 0)
        best = np.argmin(relatives.sum(axis=1))
        best_vec = best_vec + relatives[best]
        res[i] = best_vec.sum()
        print(f"res[{i}]: {res[i]}")
        distances = np.vstack((distances[:best], distances[best + 1:]))

    return sum(res)


# def greedy_kKemenys_divk_summed(votes) -> dict:
#     num_voters = len(votes)
#     num_candidates = len(votes[0])
#     res = [0] * num_voters
#     distances = calculate_vote_swap_dist(votes)
#     best = np.argmin(distances.sum(axis=1))
#     best_vec = distances[best]
#     res[0] = best_vec.sum()
#     distances = np.vstack((distances[:best], distances[best + 1:]))

#     for i in range(1, num_voters):
#         relatives = distances - best_vec
#         relatives = relatives * (relatives < 0)
#         best = np.argmin(relatives.sum(axis=1))
#         best_vec = best_vec + relatives[best]
#         res[i] = best_vec.sum() / (i + 1)
#         distances = np.vstack((distances[:best], distances[best + 1:]))

#     # res[0] = 0 # for disregarding one Kemeny (AN = ID)
#     max_dist = (num_candidates) * (num_candidates - 1) / 2
#     return {'value': sum(res) / num_voters / max_dist}

def local_search_kKemeny(votes, l, starting=None) -> dict:
    num_voters = len(votes)
    num_candidates = len(votes[0])
    max_dist = num_candidates * (num_candidates - 1) / 2
    res = []
    for k in range(1, num_voters):
        # print(k)
        if starting is None:
            d = local_search_kKemeny_single_k(votes, k, l)
        else:
            d = local_search_kKemeny_single_k(votes, k, l, starting[:k])
        d = d / max_dist / num_voters
        if d > 0:
            res.append(d)
        else:
            break
    for k in range(len(res), num_voters):
        res.append(0)

    return {'value': res}

def restore_order(x):
    for i in range(len(x)):
        for j in range(len(x) - i, len(x)):
            if x[j] >= x[-i - 1]:
                x[j] += 1
    return x

def calculate_cand_dom_dist(votes):
    num_candidates = len(votes[0])
    distances = pairwise_matrix(num_candidates, votes)
    distances = np.abs(distances - 0.5)
    np.fill_diagonal(distances, 0)
    return distances

def agreement_index(votes) -> dict:
    num_candidates = len(votes[0])
    distances = calculate_cand_dom_dist(votes)
    # print(distances)
    return {'value': distances.sum() / (num_candidates - 1) / num_candidates * 2}

def diversity_index(votes) -> dict:
    num_voters = len(votes)
    num_candidates = len(votes[0])
    max_dist = num_candidates * (num_candidates - 1) / 2
    res = [0] * num_voters
    chosen_votes = []
    distances = calculate_vote_swap_dist(votes)
    best = np.argmin(dist:=distances.sum(axis=1))
    print(dist)
    chosen_votes.append(best)
    best_vec = distances[best]
    print(distances[best])
    res[0] = best_vec.sum() / max_dist / num_voters
    distances = np.vstack((distances[:best], distances[best + 1:]))
    print(f"res[0]: {res[0]}")
    print(f"best: {best}")

    for i in range(1, num_voters):
        relatives = distances - best_vec
        relatives = relatives * (relatives < 0)
        best = np.argmin(relatives.sum(axis=1))
        print(f"best: {best}")
        chosen_votes.append(best)
        best_vec = best_vec + relatives[best]
        res[i] = best_vec.sum() / max_dist / num_voters
        print(f"res[{i}]: {res[i]}")
        distances = np.vstack((distances[:best], distances[best + 1:]))

    print(chosen_votes)
    chosen_votes = restore_order(chosen_votes)

    print(chosen_votes)
    res_1 = local_search_kKemeny(votes, 1, chosen_votes)['value']
    res_2 = local_search_kKemeny(votes, 1)['value']
    print(res_1)
    print(res_2)
    res = [min(d_1, d_2) for d_1, d_2 in zip(res_1, res_2)]
    print(res)
    return {'value': sum([x / (i + 1) for i, x in enumerate(res)])}


def polarization_index(votes) -> dict:
    num_voters = len(votes)
    num_candidates = len(votes[0])
    distances = calculate_vote_swap_dist(votes)
    best_1 = np.argmin(distances.sum(axis=1))
    best_vec = distances[best_1]
    first_kemeny = best_vec.sum()
    distances = np.vstack((distances[:best_1], distances[best_1 + 1:]))

    relatives = distances - best_vec
    relatives = relatives * (relatives < 0)
    best_2 = np.argmin(relatives.sum(axis=1))

    if best_1 <= best_2:
        best_2 = best_2 + 1

    chosen = [best_1, best_2]
    chosen.sort()

    second_kemeny_1 = local_search_kKemeny_single_k(votes, 2, 1, starting=chosen)
    second_kemeny_2 = local_search_kKemeny_single_k(votes, 2, 1)
    second_kemeny = min(second_kemeny_1, second_kemeny_2)

    max_dist = (num_candidates) * (num_candidates - 1) / 2
    return {'value': 2 * (first_kemeny - second_kemeny) / num_voters / max_dist}

if __name__ == "__main__":

    votes = [
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
    ]

#     print("-" * 50)

#     start_time_agreement = time.time()
#     result = diversity_index(votes)
#     end_time_agreement = time.time()
#     print(f"Result: {result}")
#     print(f"Agreement Index: {end_time_agreement - start_time_agreement} seconds")

#     # num_candidates = len(votes[0])
#     # start_time_single_k = time.time()
#     # res = local_search_kKemeny_single_k(votes, 2, 1)
#     # end_time_single_k = time.time()
#     # print(f"Result: {res}")
#     # print(f"local_search_kKemeny_single_k: {end_time_single_k - start_time_single_k} seconds")
    
#     # print("-" * 50)
    
#     # start_time_local_search = time.time()
#     # res = local_search_kKemeny(votes, 1)
#     # end_time_local_search = time.time()
#     # print(f"Result: {res}")
#     # print(f"local_search_kKemeny: {end_time_local_search - start_time_local_search} seconds")
    
#     # print("-" * 50)
    
#     # start_time_polarization = time.time()
#     # result = polarization_index(votes)
#     # end_time_polarization = time.time()
#     # print(f"Result: {result}")
#     # print(f"Polarization Index: {end_time_polarization - start_time_polarization} seconds")
    
#     # print("-" * 50)
    
    start_time_ranking = time.time()
    result = kemeny_ranking(votes)
    end_time_ranking = time.time()
    print(f"Best ranking: {result[0]}")
    print(f"Best distance: {result[1]}")
    print(f"Kemeny Ranking: {end_time_ranking - start_time_ranking} seconds")
#     # result = diversity_index(votes)

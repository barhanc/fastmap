#ifndef UTILS_H
#define UTILS_H

#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <stdbool.h>
#include <string.h>


static inline void create_pairwise_matrix(int num_voters, int num_candidates, int **votes, double **matrix) {
    for (int i = 0; i < num_candidates; i++) {
        for (int j = 0; j < num_candidates; j++) {
            matrix[i][j] = 0.0;
        }
    }

    for (int v = 0; v < num_voters; v++) {
        for (int c1 = 0; c1 < num_candidates; c1++) {
            for (int c2 = c1 + 1; c2 < num_candidates; c2++) {
                int candidate1 = votes[v][c1];
                int candidate2 = votes[v][c2];
                matrix[candidate1][candidate2] += 1.0;
            }
        }
    }

    for (int i = 0; i < num_candidates; i++) {
        for (int j = i + 1; j < num_candidates; j++) {
            matrix[i][j] /= num_voters;
            matrix[j][i] = 1.0 - matrix[i][j];
        }
    }
}

static inline void calculate_cand_dom_dist(int **votes, int num_voters, int num_candidates, double **distances) {

    double **pairwise_matrix = (double **)malloc(num_candidates * sizeof(double *));
    for (int i = 0; i < num_candidates; i++) {
        pairwise_matrix[i] = (double *)malloc(num_candidates * sizeof(double));
    }

    create_pairwise_matrix(num_voters, num_candidates, votes, pairwise_matrix);

    for (int i = 0; i < num_candidates; i++) {
        for (int j = 0; j < num_candidates; j++) {
            if (i != j) {
                distances[i][j] = fabs(pairwise_matrix[i][j] - 0.5);
            } else {
                distances[i][j] = 0.0;
            }
        }
    }

    for (int i = 0; i < num_candidates; i++) {
        free(pairwise_matrix[i]);
    }
    free(pairwise_matrix);
}

static inline double calculate_distance(int *arr, double **wmg, int m) {
    double dist = 0;
    for (int i = 0; i < m; i++) {
        for (int j = i + 1; j < m; j++) {
            dist += wmg[arr[j]][arr[i]];
        }
    }
    return dist;
}

static inline void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

static inline void generate_next_permutation(int *arr, int n, int pos) {
    int i;
    for (i = n - 1; i > pos; i--) {
        if (arr[i] > arr[pos]) {
            break;
        }
    }
    swap(&arr[pos], &arr[i]);
}

static inline void kemeny_ranking(int **votes, int num_voters, int num_candidates, int *best, double *best_d) {
    double **matrix = (double **)malloc(num_candidates * sizeof(double *));
    for (int i = 0; i < num_candidates; i++) {
        matrix[i] = (double *)malloc(num_candidates * sizeof(double));
    }

    create_pairwise_matrix(num_voters, num_candidates, votes, matrix);
    int *arr = (int *)malloc(num_candidates * sizeof(int));
    for (int i = 0; i < num_candidates; i++) {
        arr[i] = i;
    }

    *best_d = INFINITY;

    while (1) {
        double dist = calculate_distance(arr, matrix, num_candidates);

        if (dist < *best_d) {
            for (int i = 0; i < num_candidates; i++) {
                best[i] = arr[i];
            }
            *best_d = dist;
        }

        int pos = -1;
        for (int i = num_candidates - 2; i >= 0; i--) {
            if (arr[i] < arr[i + 1]) {
                pos = i;
                break;
            }
        }

        if (pos == -1) {
            break;
        }

        generate_next_permutation(arr, num_candidates, pos);

        int left = pos + 1, right = num_candidates - 1;
        while (left < right) {
            swap(&arr[left], &arr[right]);
            left++;
            right--;
        }
    }

    for (int i = 0; i < num_candidates; i++) {
        free(matrix[i]);
    }
    free(matrix);
    free(arr);
}

int swap_distance_between_votes(int *v1, int *v2, int n) {
    int swap_distance = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if ((v1[i] > v1[j] && v2[i] < v2[j]) || (v1[i] < v1[j] && v2[i] > v2[j])) {
                swap_distance++;
            }
        }
    }
    return swap_distance;
}

void compute_potes(int **votes, int votes_num, int n, int **potes) {
    for (int v = 0; v < votes_num; v++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (votes[v][j] == i) {
                    potes[v][i] = j;
                    break;
                }
            }
        }
    }
}

void calculate_vote_swap_dist(int **votes, int votes_num, int **distances, int n) {
    int **potes = (int **)malloc(votes_num * sizeof(int *));
    for (int i = 0; i < votes_num; i++) {
        potes[i] = (int *)malloc(n * sizeof(int));
    }

    compute_potes(votes, votes_num, n, potes);

    for (int v1 = 0; v1 < votes_num; v1++) {
        for (int v2 = 0; v2 < votes_num; v2++) {
            distances[v1][v2] = swap_distance_between_votes(potes[v1], potes[v2], n);
        }
    }

    for (int i = 0; i < votes_num; i++) {
        free(potes[i]);
    }
    free(potes);
}

static inline int distances_to_rankings(int *rankings, int **distances, int k, int votes_num) {
    int sum = 0;
    for (int i = 0; i < votes_num; i++) {
        int min_distance = distances[rankings[0]][i];
        for (int j = 1; j < k; j++) {
            if (distances[rankings[j]][i] < min_distance) {
                min_distance = distances[rankings[j]][i];
            }
        }
        sum += min_distance;
    }
    return sum;
}

static inline bool find_improvement(int **distances, int *d, int *starting, int *rest, int votes_num, int k, int l) {
    int *ranks = (int *)malloc(k * sizeof(int));
    bool found_improvement = false;

    for (int i = 0; i < (1 << k); i++) {
        if (__builtin_popcount(i) != l) continue;

        int cut[l];
        int cut_index = 0;
        for (int j = 0; j < k; j++) {
            if ((i & (1 << j)) != 0) {
                cut[cut_index++] = j;
            }
        }
        
        for (int m = 0; m < (1 << (votes_num - k)); m++) {
            if (__builtin_popcount(m) != l) continue;

            int paste[l];
            int paste_index = 0;
            for (int n = 0; n < (votes_num - k); n++) {
                if ((m & (1 << n)) != 0) {
                    paste[paste_index++] = rest[n];
                }
            }

            int j = 0;
            for (int i = 0; i < k; i++) {
                if (j < l && cut[j] == i) {
                    ranks[i] = paste[j++];
                } else {
                    ranks[i] = starting[i];
                }
            }

            bool valid = true;
            for (int a = 0; a < k; a++) {
                for (int b = a + 1; b < k; b++) {
                    if (ranks[a] == ranks[b]) {
                        valid = false;
                        break;
                    }
                }
                if (!valid) break;
            }

            if (valid) {
                int d_new = distances_to_rankings(ranks, distances, k, votes_num);
                if (*d > d_new) {
                    *d = d_new;
                    for (int a = 0; a < k; a++) {
                        starting[a] = ranks[a];
                    }
                    found_improvement = true;
                    break;
                }
            }
        }

        if (found_improvement) break;
    }

    free(ranks);
    return found_improvement;
}

static inline void restore_order(int *x, int len) {
    for (int i = 0; i < len; i++) {
        for (int j = len - i; j < len; j++) {
            if (x[j] >= x[len - i - 1]) {
                x[j] += 1;
            }
        }
    }
}

static inline int local_search_kKemeny_single_k(int **votes, int k, int l, int votes_num, int cand_num, int *starting) {
    bool starting_allocated = false;
    if (!starting) {
        starting = (int *)malloc(k * sizeof(int));
        starting_allocated = true;
        for (int i = 0; i < k; i++) {
            starting[i] = i;
        }
    }

    // Allocate distances
    int **distances = (int **)malloc(votes_num * sizeof(int *));
    for (int i = 0; i < votes_num; i++) {
        distances[i] = (int *)malloc(votes_num * sizeof(int));
    }

    // Calculate distances
    calculate_vote_swap_dist(votes, votes_num, distances, cand_num);

    // Initial distance computation
    int d = distances_to_rankings(starting, distances, k, votes_num);

    bool check = true;
    while (check) {
        // Allocate rest array
        int *rest = (int *)malloc((votes_num - k) * sizeof(int));
        int rest_index = 0;

        for (int i = 0; i < votes_num; i++) {
            bool is_starting = false;
            for (int j = 0; j < k; j++) {
                if (starting[j] == i) {
                    is_starting = true;
                    break;
                }
            }
            if (!is_starting) {
                rest[rest_index++] = i;
            }
        }

        check = find_improvement(distances, &d, starting, rest, votes_num, k, l);

        free(rest); // Free rest array
    }

    // Free distances array
    for (int i = 0; i < votes_num; i++) {
        free(distances[i]);
    }
    free(distances);

    if (starting_allocated) {
        free(starting);
    }

    return d;
}

double *local_search_kKemeny(int **votes, int num_voters, int num_candidates, int l, int *starting) {
    double max_dist = num_candidates * (num_candidates - 1) / 2.0;
    double *res = (double *)malloc(num_voters * sizeof(double));
    if (res == NULL) {
        printf("Memory allocation failed\n");
        return NULL;
    }

    int filled_count = 0;
    for (int k = 1; k < num_voters; k++) {  // Loop from 1 to num_voters - 1
        double d;
        if (starting == NULL) {
        d = local_search_kKemeny_single_k(votes, k, l, num_voters, num_candidates, NULL) / (max_dist * num_voters);
        } else {
            int *starting_k = (int *)malloc(k * sizeof(int));
            if (starting_k == NULL) {
                printf("Memory allocation failed\n");
                free(res);
                return NULL;
            }

            for (int i = 0; i < k; i++) {
                starting_k[i] = starting[i];
            }

            d = local_search_kKemeny_single_k(votes, k, l, num_voters, num_candidates, starting_k) / (max_dist * num_voters);
            free(starting_k);
        }

        if (d > 0) {
            res[k - 1] = d;
            filled_count++;
        } else {
            break;
        }
    }

    // Fill remaining entries in `res` with 0 if any
    for (int k = filled_count; k < num_voters; k++) {
        res[k] = 0.0;
    }

    return res;
}


static inline void update_indexes(int* indexes, int index, int no_start_indexes) {
    for (int i = index; i < no_start_indexes-1; i++) {
        indexes[i] = indexes[i+1];
    }
}

static inline double polarization_index(int **votes, int num_voters, int num_candidates) {
    int **distances = (int **)malloc(num_voters * sizeof(int *));
    for (int i = 0; i < num_voters; i++) {
        distances[i] = (int *)malloc(num_voters * sizeof(int));
    }
    calculate_vote_swap_dist(votes, num_voters, distances, num_candidates);

    int best_1 = 0;
    int min_sum = INT_MAX;
    for (int i = 0; i < num_voters; i++) {
        int sum = 0;
        for (int j = 0; j < num_voters; j++) {
            sum += distances[i][j];
        }
        if (sum < min_sum) {
            min_sum = sum;
            best_1 = i;
        }
    }

    int *best_vec = distances[best_1];
    int first_kemeny = 0;
    for (int i = 0; i < num_voters; i++) {
        first_kemeny += best_vec[i];
    }

    int **new_distances = (int **)malloc((num_voters - 1) * sizeof(int *));
    for (int i = 0, k = 0; i < num_voters; i++) {
        if (i == best_1) continue;
        new_distances[k] = (int *)malloc(num_voters * sizeof(int));
        memcpy(new_distances[k], distances[i], num_voters * sizeof(int));
        k++;
    }

    int **relatives = (int **)malloc((num_voters - 1) * sizeof(int *));
    for (int i = 0; i < num_voters - 1; i++) {
        relatives[i] = (int *)malloc(num_voters * sizeof(int));
        for (int j = 0; j < num_voters; j++) {
            relatives[i][j] = new_distances[i][j] - best_vec[j];
            if (relatives[i][j] >= 0) {
                relatives[i][j] = 0;
            }
        }
    }

    int best_2 = 0;
    min_sum = INT_MAX;
    for (int i = 0; i < num_voters - 1; i++) {
        int sum = 0;
        for (int j = 0; j < num_voters; j++) {
            sum += relatives[i][j];
        }
        if (sum < min_sum) {
            min_sum = sum;
            best_2 = i;
        }
    }

    if (best_2 >= best_1) {
        best_2 += 1;
    }

    int chosen[2] = {best_1, best_2};
    if (chosen[0] > chosen[1]) {
        int temp = chosen[0];
        chosen[0] = chosen[1];
        chosen[1] = temp;
    }

    int second_kemeny_1 = local_search_kKemeny_single_k(votes, 2, 1, num_voters, num_candidates, chosen);
    int second_kemeny_2 = local_search_kKemeny_single_k(votes, 2, 1, num_voters, num_candidates, NULL);
    int second_kemeny = second_kemeny_1 < second_kemeny_2 ? second_kemeny_1 : second_kemeny_2;

    double max_dist = (num_candidates) * (num_candidates - 1) / 2.0;
    double value = 2 * (first_kemeny - second_kemeny) / (double)(num_voters) / max_dist;

    for (int i = 0; i < num_voters; i++) {
        if (distances[i] != NULL) free(distances[i]);
    }
    free(distances);

    for (int i = 0; i < num_voters - 1; i++) {
        free(new_distances[i]);
    }
    free(new_distances);

    for (int i = 0; i < num_voters - 1; i++) {
        free(relatives[i]);
    }
    free(relatives);

    return value;
}

static inline double diversity_index(int** votes, int num_voters, int num_candidates) {
    int max_dist = num_candidates * (num_candidates - 1) / 2;
    double* res = (double*)malloc(num_voters * sizeof(double));
    int** distances = (int**)malloc(num_voters * sizeof(int*));
    for (int i = 0; i < num_voters; ++i) {
        distances[i] = (int*)malloc(num_voters * sizeof(int));
    }

    calculate_vote_swap_dist(votes, num_voters, distances, num_candidates);

    int* chosen_votes = (int*)malloc(num_voters * sizeof(int));
    int* indexes_left = (int*)malloc(num_voters * sizeof(int));

    for (int i = 0; i < num_voters; i++) {
        indexes_left[i] = i;
    }

    double* dist_array = (double*)malloc(num_voters * sizeof(double)); 
    for (int i = 0; i < num_voters; ++i) {
        dist_array[i] = 0;
        for (int j = 0; j < num_voters; ++j) {
            dist_array[i] += distances[i][j];
        }
    }

    int best = 0;
    for (int i = 1; i < num_voters; ++i) {
        if (dist_array[i] < dist_array[best]) {
            best = i;
        }
    }

    chosen_votes[0] = best;
    update_indexes(indexes_left, best, num_voters);
    double* best_vec = (double*)malloc(num_voters * sizeof(double));
    for (int i = 0; i < num_voters; ++i) {
        best_vec[i] = distances[best][i];
    }

    res[0] = 0;
    for (int i = 0; i < num_voters; ++i) {
        res[0] += best_vec[i];
    }
    res[0] = res[0] / max_dist / num_voters;

    free(dist_array);

    for (int i = 1; i < num_voters; ++i) {
        int** relatives = (int**)malloc((num_voters - i) * sizeof(int*));
        for (int j = 0; j < num_voters - i; ++j) {
            relatives[j] = (int*)malloc(num_voters * sizeof(int));
        }

        for (int j = 0; j < num_voters - i; ++j) {
            for (int k = 0; k < num_voters; ++k) {
                relatives[j][k] = distances[indexes_left[j]][k] - best_vec[k];
                if (relatives[j][k] > 0) relatives[j][k] = 0;
            }
        }

        int* dist_new = (int*)malloc((num_voters - i) * sizeof(int));
        for (int j = 0; j < num_voters - i; ++j) {
            dist_new[j] = 0;
            for (int k = 0; k < num_voters; ++k) {
                dist_new[j] += relatives[j][k];
            }
        }

        best = 0;
        for (int j = 1; j < num_voters - i; ++j) {
            if (dist_new[j] < dist_new[best]) {
                best = j;
            }
        }
        chosen_votes[i] = best;
        update_indexes(indexes_left, best, num_voters);
        for (int j = 0; j < num_voters; ++j) {
            best_vec[j] += relatives[best][j];
        }

        res[i] = 0;
        for (int j = 0; j < num_voters; ++j) {
            res[i] += best_vec[j];
        }
        res[i] = res[i] / max_dist / num_voters;

        free(dist_new);
        for (int j = 0; j < num_voters - i; ++j) {
            free(relatives[j]);
        }
        free(relatives);
    }

    restore_order(chosen_votes, num_voters);

    double *res_1 = local_search_kKemeny(votes, num_voters, num_candidates, 1, chosen_votes);
    double *res_2 = local_search_kKemeny(votes, num_voters, num_candidates, 1, NULL);

    double *min_values = (double *)malloc(num_voters * sizeof(double));
    if (min_values == NULL) {
        free(res);
        free(chosen_votes);
        for (int i = 0; i < num_voters; i++) {
            free(distances[i]);
        }
        free(distances);
        free(res_1);
        free(res_2);
    }
    for (int i = 0; i < num_voters; i++) {
        min_values[i] = res_1[i] < res_2[i] ? res_1[i] : res_2[i];
    }

    double diversity_index_value = 0;
    for (int i = 0; i < num_voters; i++) {
        diversity_index_value += min_values[i] / (i + 1);
    }

    free(best_vec);
    free(chosen_votes);
    free(indexes_left);
    for (int i = 0; i < num_voters; ++i) {
        free(distances[i]);
    }
    free(distances);
    free(res);
    free(min_values);
    free(res_1);
    free(res_2);

    return diversity_index_value;
}

static inline double polarization_1by2Kemenys(int** votes, int num_voters, int num_candidates) {
    double res = 0.0;
    int* indexes_left = (int*)malloc(num_voters * sizeof(int));
    for (int i = 0; i < num_voters; i++) {
        indexes_left[i] = i;
    }

    int** distances = (int**)malloc(num_voters * sizeof(int*));
    for (int i = 0; i < num_voters; ++i) {
        distances[i] = (int*)malloc(num_voters * sizeof(int));
    }

    calculate_vote_swap_dist(votes, num_voters, distances, num_candidates);

    double* dist_array = (double*)malloc(num_voters * sizeof(double)); 
    for (int i = 0; i < num_voters; ++i) {
        dist_array[i] = 0;
        for (int j = 0; j < num_voters; ++j) {
            dist_array[i] += distances[i][j];
        }
    }

    int best = 0;
    for (int i = 1; i < num_voters; ++i) {
        if (dist_array[i] < dist_array[best]) {
            best = i;
        }
    }

    double* best_vec = (double*)malloc(num_voters * sizeof(double));
    for (int i = 0; i < num_voters; ++i) {
        best_vec[i] = distances[best][i];
    }

    for (int i = 0; i < num_voters; ++i) {
        res += best_vec[i];
    }

    double first_kemeny = res;

    update_indexes(indexes_left, best, num_voters);
    free(dist_array);

    int** relatives = (int**)malloc((num_voters - 1) * sizeof(int*));
    for (int j = 0; j < num_voters - 1; ++j) {
        relatives[j] = (int*)malloc(num_voters * sizeof(int));
    }

    for (int j = 0; j < num_voters - 1; ++j) {
        for (int k = 0; k < num_voters; ++k) {
            relatives[j][k] = distances[indexes_left[j]][k] - best_vec[k];
            if (relatives[j][k] > 0) relatives[j][k] = 0;
        }
    }

    int* dist_new = (int*)malloc((num_voters - 1) * sizeof(int));
    for (int j = 0; j < num_voters - 1; ++j) {
        dist_new[j] = 0;
        for (int k = 0; k < num_voters; ++k) {
            dist_new[j] += relatives[j][k];
        }
    }

    best = 0;
    for (int j = 1; j < num_voters - 1; ++j) {
        if (dist_new[j] < dist_new[best]) {
            best = j;
        }
    }
    update_indexes(indexes_left, best, num_voters);
    for (int j = 0; j < num_voters; ++j) {
        best_vec[j] += relatives[best][j];
    }

    res = 0.0;
    for (int j = 0; j < num_voters; ++j) {
        res += best_vec[j];
    }
    double second_kemeny = res;

    free(dist_new);
    for (int j = 0; j < num_voters - 1; ++j) {
        free(relatives[j]);
    }
    free(relatives);

    free(best_vec);
    free(indexes_left);
    for (int i = 0; i < num_voters; ++i) {
        free(distances[i]);
    }
    free(distances);

    int max_dist = num_candidates * (num_candidates - 1) / 2;

    return (first_kemeny - second_kemeny) / num_voters / max_dist;
}

static inline double greedy_kmeans_summed(int** votes, int num_voters, int num_candidates) {
    double* res = (double*)malloc(num_voters * sizeof(double));
    double sum = 0.0;
    int* indexes_left = (int*)malloc(num_voters * sizeof(int));
    for (int i = 0; i < num_voters; i++) {
        indexes_left[i] = i;
    }

    int** distances = (int**)malloc(num_voters * sizeof(int*));
    for (int i = 0; i < num_voters; ++i) {
        distances[i] = (int*)malloc(num_voters * sizeof(int));
    }

    calculate_vote_swap_dist(votes, num_voters, distances, num_candidates);
    for (int i = 0; i < num_voters; i++) {
        for (int j = 0; j < num_voters; j++) {
            distances[i][j] = distances[i][j] * distances[i][j];
        }
    }

    double* dist_array = (double*)malloc(num_voters * sizeof(double)); 
    for (int i = 0; i < num_voters; ++i) {
        dist_array[i] = 0;
        for (int j = 0; j < num_voters; ++j) {
            dist_array[i] += distances[i][j];
        }
    }

    int best = 0;
    for (int i = 1; i < num_voters; ++i) {
        if (dist_array[i] < dist_array[best]) {
            best = i;
        }
    }

    double* best_vec = (double*)malloc(num_voters * sizeof(double));
    for (int i = 0; i < num_voters; ++i) {
        best_vec[i] = distances[best][i];
    }

    res[0] = 0;
    for (int i = 0; i < num_voters; ++i) {
        res[0] += best_vec[i];
    }

    sum += res[0];

    update_indexes(indexes_left, best, num_voters);
    free(dist_array);

    for (int i = 1; i < num_voters; ++i) {
        int** relatives = (int**)malloc((num_voters - i) * sizeof(int*));
        for (int j = 0; j < num_voters - i; ++j) {
            relatives[j] = (int*)malloc(num_voters * sizeof(int));
        }

        for (int j = 0; j < num_voters - i; ++j) {
            for (int k = 0; k < num_voters; ++k) {
                relatives[j][k] = distances[indexes_left[j]][k] - best_vec[k];
                if (relatives[j][k] > 0) relatives[j][k] = 0;
            }
        }

        int* dist_new = (int*)malloc((num_voters - i) * sizeof(int));
        for (int j = 0; j < num_voters - i; ++j) {
            dist_new[j] = 0;
            for (int k = 0; k < num_voters; ++k) {
                dist_new[j] += relatives[j][k];
            }
        }

        best = 0;
        for (int j = 1; j < num_voters - i; ++j) {
            if (dist_new[j] < dist_new[best]) {
                best = j;
            }
        }
        update_indexes(indexes_left, best, num_voters);
        for (int j = 0; j < num_voters; ++j) {
            best_vec[j] += relatives[best][j];
        }

        res[i] = 0.0;
        for (int j = 0; j < num_voters; ++j) {
            res[i] += best_vec[j];
        }
        sum += res[i];

        free(dist_new);
        for (int j = 0; j < num_voters - i; ++j) {
            free(relatives[j]);
        }
        free(relatives);
    }

    free(best_vec);
    free(indexes_left);
    for (int i = 0; i < num_voters; ++i) {
        free(distances[i]);
    }
    free(distances);
    free(res);

    return sum;
}

static inline double greedy_kKemenys_summed(int** votes, int num_voters, int num_candidates) {
    double* res = (double*)malloc(num_voters * sizeof(double));
    double sum = 0.0;
    int* indexes_left = (int*)malloc(num_voters * sizeof(int));
    for (int i = 0; i < num_voters; i++) {
        indexes_left[i] = i;
    }

    int** distances = (int**)malloc(num_voters * sizeof(int*));
    for (int i = 0; i < num_voters; ++i) {
        distances[i] = (int*)malloc(num_voters * sizeof(int));
    }

    calculate_vote_swap_dist(votes, num_voters, distances, num_candidates);

    double* dist_array = (double*)malloc(num_voters * sizeof(double)); 
    for (int i = 0; i < num_voters; ++i) {
        dist_array[i] = 0;
        for (int j = 0; j < num_voters; ++j) {
            dist_array[i] += distances[i][j];
        }
    }

    int best = 0;
    for (int i = 1; i < num_voters; ++i) {
        if (dist_array[i] < dist_array[best]) {
            best = i;
        }
    }

    double* best_vec = (double*)malloc(num_voters * sizeof(double));
    for (int i = 0; i < num_voters; ++i) {
        best_vec[i] = distances[best][i];
    }

    res[0] = 0;
    for (int i = 0; i < num_voters; ++i) {
        res[0] += best_vec[i];
    }

    sum += res[0];

    update_indexes(indexes_left, best, num_voters);
    free(dist_array);

    for (int i = 1; i < num_voters; ++i) {
        int** relatives = (int**)malloc((num_voters - i) * sizeof(int*));
        for (int j = 0; j < num_voters - i; ++j) {
            relatives[j] = (int*)malloc(num_voters * sizeof(int));
        }

        for (int j = 0; j < num_voters - i; ++j) {
            for (int k = 0; k < num_voters; ++k) {
                relatives[j][k] = distances[indexes_left[j]][k] - best_vec[k];
                if (relatives[j][k] > 0) relatives[j][k] = 0;
            }
        }

        int* dist_new = (int*)malloc((num_voters - i) * sizeof(int));
        for (int j = 0; j < num_voters - i; ++j) {
            dist_new[j] = 0;
            for (int k = 0; k < num_voters; ++k) {
                dist_new[j] += relatives[j][k];
            }
        }

        best = 0;
        for (int j = 1; j < num_voters - i; ++j) {
            if (dist_new[j] < dist_new[best]) {
                best = j;
            }
        }
        update_indexes(indexes_left, best, num_voters);
        for (int j = 0; j < num_voters; ++j) {
            best_vec[j] += relatives[best][j];
        }

        res[i] = 0.0;
        for (int j = 0; j < num_voters; ++j) {
            res[i] += best_vec[j];
        }
        sum += res[i];

        free(dist_new);
        for (int j = 0; j < num_voters - i; ++j) {
            free(relatives[j]);
        }
        free(relatives);
    }

    free(best_vec);
    free(indexes_left);
    for (int i = 0; i < num_voters; ++i) {
        free(distances[i]);
    }
    free(distances);
    free(res);

    return sum;
}

static inline double greedy_kKemenys_divk_summed(int** votes, int num_voters, int num_candidates) {
    double* res = (double*)malloc(num_voters * sizeof(double));
    double sum = 0.0;
    int* indexes_left = (int*)malloc(num_voters * sizeof(int));
    for (int i = 0; i < num_voters; i++) {
        indexes_left[i] = i;
    }

    int** distances = (int**)malloc(num_voters * sizeof(int*));
    for (int i = 0; i < num_voters; ++i) {
        distances[i] = (int*)malloc(num_voters * sizeof(int));
    }

    calculate_vote_swap_dist(votes, num_voters, distances, num_candidates);

    double* dist_array = (double*)malloc(num_voters * sizeof(double)); 
    for (int i = 0; i < num_voters; ++i) {
        dist_array[i] = 0;
        for (int j = 0; j < num_voters; ++j) {
            dist_array[i] += distances[i][j];
        }
    }

    int best = 0;
    for (int i = 1; i < num_voters; ++i) {
        if (dist_array[i] < dist_array[best]) {
            best = i;
        }
    }

    double* best_vec = (double*)malloc(num_voters * sizeof(double));
    for (int i = 0; i < num_voters; ++i) {
        best_vec[i] = distances[best][i];
    }

    res[0] = 0;
    for (int i = 0; i < num_voters; ++i) {
        res[0] += best_vec[i];
    }

    sum += res[0];

    update_indexes(indexes_left, best, num_voters);
    free(dist_array);

    for (int i = 1; i < num_voters; ++i) {
        int** relatives = (int**)malloc((num_voters - i) * sizeof(int*));
        for (int j = 0; j < num_voters - i; ++j) {
            relatives[j] = (int*)malloc(num_voters * sizeof(int));
        }

        for (int j = 0; j < num_voters - i; ++j) {
            for (int k = 0; k < num_voters; ++k) {
                relatives[j][k] = distances[indexes_left[j]][k] - best_vec[k];
                if (relatives[j][k] > 0) relatives[j][k] = 0;
            }
        }

        int* dist_new = (int*)malloc((num_voters - i) * sizeof(int));
        for (int j = 0; j < num_voters - i; ++j) {
            dist_new[j] = 0;
            for (int k = 0; k < num_voters; ++k) {
                dist_new[j] += relatives[j][k];
            }
        }

        best = 0;
        for (int j = 1; j < num_voters - i; ++j) {
            if (dist_new[j] < dist_new[best]) {
                best = j;
            }
        }
        update_indexes(indexes_left, best, num_voters);
        for (int j = 0; j < num_voters; ++j) {
            best_vec[j] += relatives[best][j];
        }

        res[i] = 0.0;
        for (int j = 0; j < num_voters; ++j) {
            res[i] += best_vec[j];
        }
        sum += res[i] / (i + 1);

        free(dist_new);
        for (int j = 0; j < num_voters - i; ++j) {
            free(relatives[j]);
        }
        free(relatives);
    }

    free(best_vec);
    free(indexes_left);
    for (int i = 0; i < num_voters; ++i) {
        free(distances[i]);
    }
    free(distances);
    free(res);
    int max_dist = num_candidates * (num_candidates - 1) / 2;
    return sum / num_voters / max_dist;
}


static inline double greedy_2kKemenys_summed(int** votes, int num_voters, int num_candidates) {
    double* res = (double*)malloc(num_voters * sizeof(double));
    double sum = 0.0;
    int* indexes_left = (int*)malloc(num_voters * sizeof(int));
    for (int i = 0; i < num_voters; i++) {
        indexes_left[i] = i;
    }

    int** distances = (int**)malloc(num_voters * sizeof(int*));
    for (int i = 0; i < num_voters; ++i) {
        distances[i] = (int*)malloc(num_voters * sizeof(int));
    }

    calculate_vote_swap_dist(votes, num_voters, distances, num_candidates);

    double* dist_array = (double*)malloc(num_voters * sizeof(double)); 
    for (int i = 0; i < num_voters; ++i) {
        dist_array[i] = 0;
        for (int j = 0; j < num_voters; ++j) {
            dist_array[i] += distances[i][j];
        }
    }

    int best = 0;
    for (int i = 1; i < num_voters; ++i) {
        if (dist_array[i] < dist_array[best]) {
            best = i;
        }
    }

    double* best_vec = (double*)malloc(num_voters * sizeof(double));
    for (int i = 0; i < num_voters; ++i) {
        best_vec[i] = distances[best][i];
    }

    res[0] = 0;
    for (int i = 0; i < num_voters; ++i) {
        res[0] += best_vec[i];
    }

    sum += res[0];

    update_indexes(indexes_left, best, num_voters);
    free(dist_array);

    int k = 2;

    for (int i = 1; i < num_voters; ++i) {
        int** relatives = (int**)malloc((num_voters - i) * sizeof(int*));
        for (int j = 0; j < num_voters - i; ++j) {
            relatives[j] = (int*)malloc(num_voters * sizeof(int));
        }

        for (int j = 0; j < num_voters - i; ++j) {
            for (int k = 0; k < num_voters; ++k) {
                relatives[j][k] = distances[indexes_left[j]][k] - best_vec[k];
                if (relatives[j][k] > 0) relatives[j][k] = 0;
            }
        }

        int* dist_new = (int*)malloc((num_voters - i) * sizeof(int));
        for (int j = 0; j < num_voters - i; ++j) {
            dist_new[j] = 0;
            for (int k = 0; k < num_voters; ++k) {
                dist_new[j] += relatives[j][k];
            }
        }

        best = 0;
        for (int j = 1; j < num_voters - i; ++j) {
            if (dist_new[j] < dist_new[best]) {
                best = j;
            }
        }
        update_indexes(indexes_left, best, num_voters);
        for (int j = 0; j < num_voters; ++j) {
            best_vec[j] += relatives[best][j];
        }
        
        if (i + 1 == k) {
            res[i] = 0.0;
            for (int j = 0; j < num_voters; ++j) {
                res[i] += best_vec[j];
            }
            sum += res[i];
            k = k * 2;
        }

        free(dist_new);
        for (int j = 0; j < num_voters - i; ++j) {
            free(relatives[j]);
        }
        free(relatives);
    }

    free(best_vec);
    free(indexes_left);
    for (int i = 0; i < num_voters; ++i) {
        free(distances[i]);
    }
    free(distances);
    free(res);
    int max_dist = num_candidates * (num_candidates - 1) / 2;
    return sum / num_voters / max_dist / 2;
}

#endif // UTILS_H

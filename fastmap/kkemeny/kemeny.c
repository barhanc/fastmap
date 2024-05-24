#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <math.h>
#include <time.h>
#include <string.h>

void create_pairwise_matrix(int num_voters, int num_candidates, int** votes, double** matrix) {
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

double calculate_distance(int *arr, double **wmg, int m) {
    double dist = 0;
    for (int i = 0; i < m; i++) {
        for (int j = i + 1; j < m; j++) {
            dist += wmg[arr[j]][arr[i]];
        }
    }
    return dist;
}

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void generate_next_permutation(int *arr, int n, int pos) {
    int i;
    for (i = n - 1; i > pos; i--) {
        if (arr[i] > arr[pos]) {
            break;
        }
    }
    swap(&arr[pos], &arr[i]);
}

int kemeny_ranking(int **votes, int num_voters, int num_candidates) {
    double **matrix = (double **)malloc(num_candidates * sizeof(double *));
    for (int i = 0; i < num_candidates; i++) {
        matrix[i] = (double *)malloc(num_candidates * sizeof(double));
    }

    create_pairwise_matrix(num_voters, num_candidates, votes, matrix);
    int *arr = (int *)malloc(num_candidates * sizeof(int));
    for (int i = 0; i < num_candidates; i++) {
        arr[i] = i;
    }

    double best_d = INFINITY;
    int best[num_candidates];
    for (int i = 0; i < num_candidates; i++) {
        best[i] = arr[i];
    }

    while (1) {
        double dist = calculate_distance(arr, matrix, num_candidates);

        if (dist < best_d) {
            for (int i = 0; i < num_candidates; i++) {
                best[i] = arr[i];
            }
            best_d = dist;
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

    // printf("Best ranking: ");
    // for (int i = 0; i < num_candidates; i++) {
    //     printf("%d ", best[i]);
    // }
    // printf("\nBest distance: %f\n", best_d);

    free(arr);
    return best_d;
}

float dist_to_Kemeny_mean(int **votes, int m, int n) {
    return kemeny_ranking(votes, m, n) / m;
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

void calculate_vote_swap_dist(int **votes, int votes_num, int **distances, int n) {
    for (int v1 = 0; v1 < votes_num; v1++) {
        for (int v2 = 0; v2 < votes_num; v2++) {
            distances[v1][v2] = swap_distance_between_votes(votes[v1], votes[v2], n);
        }
    }
}

// space for vectorization
int distances_to_rankings(int *rankings, int **distances, int k, int votes_num) {
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

// TODO - Segment fault to fix
bool find_improvement(int **distances, int *d, int *starting, int *rest, int votes_num, int k, int l) {
    printf("Starting\n");
    int *ranks = (int *)malloc(k * sizeof(int));
    printf("Creating ranks\n");
    bool found_improvement = false;
    printf("cut/paste before allocated\n");
    int *cut = (int *)malloc(l * sizeof(int));
    int *paste = (int *)malloc(l * sizeof(int));
    printf("cut/paste allocated\n");

    for (int i = 0; i < (1 << k); i++) {
        if (__builtin_popcount(i) != l) continue;

        int index = 0;
        for (int j = 0; j < k; j++) {
            if ((i & (1 << j)) != 0) {
                cut[index++] = j;
                printf("Cut[%d]: %d\n", index - 1, j);
            }
        }

        index = 0;
        for (int j = 0; j < k; j++) {
            if (!(i & (1 << j))) {
                ranks[index++] = starting[j];
                printf("Ranks[%d]: %d\n", index - 1, starting[j]);
            }
        }
        int rest_idx = 0;
        for (int j = 0; j < (votes_num - k); j++) {
            ranks[index++] = rest[rest_idx++];
            printf("Ranks1[%d]: %d\n", index - 1, rest[rest_idx - 1]);
        }

        index = 0;
        int cut_idx = 0;
        int paste_idx = 0;
        for (int m = 0; m < k; m++) {
            if (cut_idx < l && m == cut[cut_idx]) {
                ranks[m] = rest[paste_idx++];
                printf("Ranks2[%d]: %d\n", m, rest[paste_idx - 1]);
                cut_idx++;
            } else {
                ranks[m] = starting[m];
                printf("Ranks3[%d]: %d\n", m, starting[m]);
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
            printf("Special case distances_to_rankings\n");
            int d_new = distances_to_rankings(ranks, distances, k, votes_num);
            printf("Special case distances_to_rankings_after\n");
            if (*d > d_new) {
                *d = d_new;
                for (int a = 0; a < k; a++) {
                    starting[a] = ranks[a];
                }
                found_improvement = true;
                printf("Special case free before\n");
                break;
            }
        }

        printf("cut/paste before freed\n");
    }

    printf("Ranks before freed: ");
    free(ranks);
    printf("Ranks freed\n");
    printf("cut content before freed: ");
        for (int c = 0; c < l; c++) {
            printf("%d ", cut[c]);
        }
    printf("\n");
    if(cut != NULL) free(cut);
    printf("cut freed\n");
    free(paste);
    printf("cut/paste freed\n");

    return found_improvement;
}

void restore_order(int *x, int len) {
    for (int i = 0; i < len; i++) {
        for (int j = len - i; j < len; j++) {
            if (x[j] >= x[len - i - 1]) {
                x[j] += 1;
            }
        }
    }
}

int local_search_kKemeny_single_k(int **votes, int k, int l, int votes_num, int *starting) {
    bool starting_allocated = false;
    if (!starting) {
        starting = (int *)malloc(k * sizeof(int));
        starting_allocated = true;
        for (int i = 0; i < k; i++) {
            starting[i] = i;
        }
    }

    int **distances = (int **)malloc(votes_num * sizeof(int *));
    for (int i = 0; i < votes_num; i++) {
        distances[i] = (int *)malloc(votes_num * sizeof(int));
    }

    calculate_vote_swap_dist(votes, votes_num, distances, 11);

    int d = distances_to_rankings(starting, distances, k, votes_num);
    bool check = true;

    while (check) {
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

        printf("Calling find_improvement\n");
        check = find_improvement(distances, &d, starting, rest, votes_num, k, l);
        // check = false;

        printf("Free at 289\n");
        free(rest);
    }

    for (int i = 0; i < votes_num; i++) {
        printf("Free at 298\n");
        free(distances[i]);
        printf("Free at 300\n");
    }
    printf("Free at 302\n");
    free(distances);

    if (starting_allocated) {
        printf("Free at 306\n");
        free(starting);
    }
    printf("Returning d\n");

    return d;
}


double *local_search_kKemeny(int **votes, int num_voters, int num_candidates, int l, int *starting) {

    if (starting != NULL) {
        printf("Starting: ");
        for (int i = 0; i < num_voters; i++) {
            printf("%d ", starting[i]);
        }
        printf("\n");
    }
    double max_dist = num_candidates * (num_candidates - 1) / 2;

    double *res = (double *)malloc(num_voters * sizeof(double));
    if (res == NULL) {
        printf("Memory allocation failed\n");
        return NULL;
    }

    for (int k = 1; k <= num_voters; k++) {
        double d;
        if (starting == NULL) {
            d = local_search_kKemeny_single_k(votes, k, l, num_voters, NULL) / max_dist / num_voters;
        } else {
            int *starting_k = (int *)malloc(k * sizeof(int));
            if (starting_k == NULL) {
                printf("Memory allocation failed\n");
                // printf("Free at 421\n");
                free(res);
                return NULL;
            }
            for (int i = 0; i < k; i++) {
                starting_k[i] = starting[i];
            }
            d = local_search_kKemeny_single_k(votes, k, l, num_voters, starting_k) / max_dist / num_voters;
            // printf("Free at 429\n");
            free(starting_k);
            // printf("Free at 431\n");
        }
        if (d > 0) {
            res[k - 1] = d;
        } else {
            break;
        }
    }

    // printf("Line 443\n");
    for (int k = num_voters; k < num_voters; k++) {
        res[k - 1] = 0;
    }

    return res;
}

double diversity_index(int **votes, int num_voters, int num_candidates) {
    double max_dist = num_candidates * (num_candidates - 1) / 2.0;

    double *res = (double *)malloc(num_voters * sizeof(double));
    if (res == NULL) {
        printf("Memory allocation failed\n");
    }
    for (int i = 0; i < num_voters; i++) {
        res[i] = 0;
    }

    int *chosen_votes = (int *)malloc(num_voters * sizeof(int));
    if (chosen_votes == NULL) {
        printf("Memory allocation failed\n");
        free(res);
    }

    int **distances = (int **)malloc(num_voters * sizeof(int *));
    if (distances == NULL) {
        printf("Memory allocation failed\n");
        free(res);
        free(chosen_votes);
    }
    for (int i = 0; i < num_voters; i++) {
        distances[i] = (int *)malloc(num_voters * sizeof(int));
        if (distances[i] == NULL) {
            free(res);
            free(chosen_votes);
            for (int j = 0; j < i; j++) {
                free(distances[j]);
            }
            free(distances);
        }
    }

    calculate_vote_swap_dist(votes, num_voters, distances, num_candidates);

    int best = 0;
    int best_sum = 0;
    for (int i = 0; i < num_voters; i++) {
        int sum = 0;
        for (int j = 0; j < num_voters; j++) {
            sum += distances[i][j];
        }
        if (i == 0 || sum < best_sum) {
            best = i;
            best_sum = sum;
        }
    }

    chosen_votes[0] = best; 
    int *best_vec = distances[best];
    for (int i = 0; i < num_voters; i++) {
        printf("%d ", best_vec[i]);
    }

    double best_vec_sum = 0;
    for (int i = 0; i < num_voters; i++) {
        best_vec_sum += best_vec[i];
    }

    res[0] = best_vec_sum / max_dist / num_voters;
    printf("res[0]: %f\n", res[0]);
    printf("best: %d\n", chosen_votes[0]);

    for (int i = 1; i < num_voters; i++) {
        int **relatives = (int **)malloc((num_voters - i) * sizeof(int *));
        for (int j = 0; j < num_voters - i; j++) {
            relatives[j] = (int *)malloc(num_voters * sizeof(int));
        }

        for (int j = 0; j < num_voters - i; j++) {
            for (int k = 0; k < num_voters; k++) {
                relatives[j][k] = distances[j][k] - best_vec[k];
                if (relatives[j][k] > 0) {
                    relatives[j][k] = 0;
                }
            }
        }

        best = 0;
        best_sum = 0;
        for (int j = 0; j < num_voters - i; j++) {
            int sum = 0;
            for (int k = 0; k < num_voters; k++) {
                sum += relatives[j][k];
            }
            printf("%d \n", sum);
            if (j == 0 || sum < best_sum) {
                best = j;
                best_sum = sum;
            }
        }

        printf("Best_sum[%d]: %d\n", i, best_sum);
        chosen_votes[i] = best;

        for (int j = 0; j < num_voters; j++) {
            best_vec[j] += relatives[best][j];
        }

        best_vec_sum = 0;
        for (int j = 0; j < num_voters; j++) {
            best_vec_sum += best_vec[j];
        }

        res[i] = best_vec_sum / max_dist / num_voters;
        printf("res[%d]: %f\n", i, res[i]);
        printf("best: %d\n", chosen_votes[i]);

        int **new_distances = (int **)malloc((num_voters - i - 1) * sizeof(int *));
        for (int j = 0, k = 0; j < num_voters - i; j++) {
            if (j != best) {
                new_distances[k++] = distances[j];
            } else {
                free(distances[j]);
            }
        }
        free(distances);
        distances = new_distances;

        for (int j = 0; j < num_voters - i; j++) {
            free(relatives[j]);
        }
        free(relatives);
    }

    printf("Chosen votes before restoring order: ");
    for (int i = 0; i < num_voters; i++) {
        printf("%d ", chosen_votes[i]);
    }
    printf("\n");

    restore_order(chosen_votes, num_voters);

    printf("Chosen votes after restoring order: ");
    for (int i = 0; i < num_voters; i++) {
        printf("%d ", chosen_votes[i]);
    }
    printf("\n");

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

    free(res); 
    free(chosen_votes);
    for (int i = 0; i < num_voters; i++) {
        free(distances[i]);
    }
    free(distances);
    free(min_values);
    free(res_1);
    free(res_2);

    return diversity_index_value;
}


double polarization_index(int **votes, int num_voters, int num_candidates) {
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

    int second_kemeny_1 = local_search_kKemeny_single_k(votes, 2, 1, num_voters, chosen);
    int second_kemeny_2 = local_search_kKemeny_single_k(votes, 2, 1, num_voters, NULL);
    int second_kemeny = second_kemeny_1 < second_kemeny_2 ? second_kemeny_1 : second_kemeny_2;

    double max_dist = (num_candidates) * (num_candidates - 1) / 2.0;
    double value = 2 * (first_kemeny - second_kemeny) / (double)(num_voters) / max_dist;

    for (int i = 0; i < num_voters; i++) {
        printf("Free at 739\n");
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

// Function to calculate the pairwise dominance distance
void calculate_cand_dom_dist(int **votes, int num_voters, int num_candidates, double **distances) {
    double **pairwise_matrix = (double **)malloc(num_candidates * sizeof(double *));
    for (int i = 0; i < num_candidates; i++) {
        pairwise_matrix[i] = (double *)malloc(num_candidates * sizeof(double));
    }

    create_pairwise_matrix(num_voters, num_candidates, votes, pairwise_matrix);

    // for (int i = 0; i < num_candidates; i++) {
    //     for (int j = 0; j < num_candidates; j++) {
    //         printf("%f ", pairwise_matrix[i][j]);
    //     }
    //     printf("\n");
    // }

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

double agreement_index(int **votes, int num_voters, int num_candidates) {
    double **distances = (double **)malloc(num_candidates * sizeof(double *));
    for (int i = 0; i < num_candidates; i++) {
        distances[i] = (double *)malloc(num_candidates * sizeof(double));
    }

    calculate_cand_dom_dist(votes, num_voters, num_candidates, distances);

    double sum_distances = 0;
    for (int i = 0; i < num_candidates; i++) {
        for (int j = 0; j < num_candidates; j++) {
            sum_distances += distances[i][j];
        }
    }

    for (int i = 0; i < num_candidates; i++) {
        free(distances[i]);
    }
    free(distances);

    return sum_distances / ((num_candidates - 1) * num_candidates) * 2;
}


int main() {
    int num_voters = 20;
    int num_candidates = 11;

    int example_votes[20][11] = {
        {3, 6, 0, 7, 2, 4, 5, 1, 10, 8, 9},
        {2, 3, 9, 0, 5, 1, 10, 7, 8, 4, 6},
        {3, 6, 2, 5, 0, 1, 10, 9, 7, 4, 8},
        {5, 0, 7, 3, 6, 8, 9, 1, 4, 2, 10},
        {2, 4, 1, 9, 0, 10, 8, 6, 5, 3, 7},
        {9, 4, 2, 1, 10, 5, 6, 0, 7, 8, 3},
        {10, 3, 2, 0, 5, 6, 8, 4, 7, 9, 1},
        {9, 8, 2, 0, 4, 7, 1, 3, 10, 5, 6},
        {5, 1, 0, 3, 2, 4, 6, 10, 9, 8, 7},
        {10, 7, 6, 4, 0, 9, 2, 8, 1, 3, 5},
        {9, 7, 3, 0, 4, 8, 1, 2, 5, 10, 6},
        {6, 4, 5, 10, 1, 9, 7, 3, 8, 0, 2},
        {4, 10, 0, 8, 1, 3, 2, 7, 6, 9, 5},
        {3, 10, 2, 6, 9, 8, 1, 7, 5, 4, 0},
        {2, 6, 9, 8, 7, 10, 1, 4, 0, 5, 3},
        {0, 1, 7, 8, 5, 3, 4, 2, 10, 6, 9},
        {5, 8, 2, 3, 1, 7, 0, 4, 9, 6, 10},
        {2, 9, 8, 10, 3, 6, 1, 7, 4, 5, 0},
        {10, 9, 7, 0, 8, 4, 1, 6, 2, 5, 3},
        {10, 4, 2, 1, 9, 3, 0, 6, 7, 5, 8},
    };

    int **votes = (int **)malloc(num_voters * sizeof(int *));
    for (int i = 0; i < num_voters; i++) {
        votes[i] = (int *)malloc(num_candidates * sizeof(int));
    }

    for (int i = 0; i < num_voters; i++) {
        for (int j = 0; j < num_candidates; j++) {
            votes[i][j] = example_votes[i][j];
        }
    }

    clock_t start_time = clock();
    kemeny_ranking(votes, num_voters, num_candidates);
    // int d = local_search_kKemeny_single_k(votes, 11, 20, num_voters, NULL);
    // for (int i = 0; i < 20; i++) {
    //     double d_index = polarization_index(votes, num_voters, num_candidates);
    // }
    // double d_index = polarization_index(votes, num_voters, num_candidates);
    clock_t end_time = clock();

    // printf("Value of d: %d\n", d);
    // printf("Polarization index: %f\n", d_index);

    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC; 

    printf("Execution time: %.6f seconds\n", elapsed_time);

    return 0;
}

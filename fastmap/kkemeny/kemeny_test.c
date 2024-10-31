#include <stdio.h>
#include "kemeny.h"

int main() {

    int num_voters = 7;
    int num_candidates = 7;

    // int example_votes[3][6] = {
    //     {0, 1, 2, 3, 4, 5},
    //     {5, 4, 3, 2, 1, 0},
    //     {0, 2, 4, 1, 3, 5}
    // };

    int example_votes[7][7] = {
        {0, 3, 5, 1, 6, 4, 2},
        {4, 0, 5, 1, 2, 6, 3},
        {6, 3, 4, 0, 5, 1, 2},
        {0, 4, 6, 5, 2, 1, 3},
        {1, 4, 5, 3, 0, 6, 2}, 
        {0, 1, 6, 5, 4, 2, 3}, 
        {4, 6, 1, 2, 0, 3, 5}
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

    int result = local_search_kKemeny_single_k(votes, 1, 1, num_voters, num_candidates, NULL);
    printf("Result: %d\n", result);

    return 0;
}
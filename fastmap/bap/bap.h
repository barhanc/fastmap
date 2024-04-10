#include <stdint.h>
#include <stdlib.h>

#include "lap.h"

#define swap(type, x, y) \
    {                    \
        type SWAP = x;   \
        x = y;           \
        y = SWAP;        \
    }

/**
 * @brief Brute-force algorithm solving Bilinear Assignment Problem (BAP)
 * ```
 *  min_{σ ∈ S_nc} min_{v ∈ S_nv} sum_{i=1,..,nv} sum_{k=1,..,nc} d(i,v(i),k,σ(k))
 * ```
 * where d(i,j,k,l) is the cost tensor, S_n denotes the set of all permutations of the set {1,..,n}
 * and integers nv, nc describe the size of the problem instance. The algorithm generates all
 * permutations of S_nc using Heap's algorithm and for every generated permutation σ solves the
 * following Linear Assignment Problem (LAP) in order to find the optimal permutation v ∈ S_nv
 * ```
 *   min_{v ∈ S_nv} sum_{i=1,..,nv} ( sum_{k=1,..,nc} d(i,v(i),k,σ(k)) ) .
 * ```
 * NOTE: You must define a macro `#define d(i,j,k,l) ...` which computes the cost tensor (see:
 * fastmap/example/).
 *
 * @param nv problem size parameter
 * @param nc problem size parameter
 * @return int32_t
 */
static int32_t
bap_bf (const size_t nv, const size_t nc)
{
    // Cost matrix for LAP
    int32_t *cost = calloc (nv * nv, sizeof (int32_t));
    for (size_t i = 0; i < nv; i++)
        for (size_t j = 0; j < nv; j++)
            for (size_t k = 0; k < nc; k++)
                cost[i * nv + j] += d (i, j, k, k);

    // Stack pointer and encoding of the stack in iterative version of Heap's algorithm. See:
    // https://en.wikipedia.org/wiki/Heap%27s_algorithm
    size_t alpha = 1, *stack = calloc (nc, sizeof (size_t));

    // Permutation array initialized to identity permutation
    size_t *sigma = calloc (nc, sizeof (size_t));
    for (size_t i = 0; i < nc; i++)
        sigma[i] = i;

    // Auxiliary variables required for J-V LAP algorithm
    int32_t *a = calloc (nv, sizeof (int32_t)), *b = calloc (nv, sizeof (int32_t));
    int32_t *x = calloc (nv, sizeof (int32_t)), *y = calloc (nv, sizeof (int32_t));

    int32_t best_res = lap (nv, cost, a, b, x, y);

    while (alpha < nc)
    {
        if (stack[alpha] < alpha)
        {
            if (alpha % 2 == 0)
            {
                for (size_t i = 0; i < nv; i++)
                    for (size_t j = 0; j < nv; j++)
                    {
                        cost[i * nv + j] += d (i, j, alpha, sigma[0]) + d (i, j, 0, sigma[alpha]);
                        cost[i * nv + j] -= d (i, j, 0, sigma[0]) + d (i, j, alpha, sigma[alpha]);
                    }
                swap (size_t, sigma[0], sigma[alpha]);
            }
            else
            {
                for (size_t i = 0; i < nv; i++)
                    for (size_t j = 0; j < nv; j++)
                    {
                        cost[i * nv + j] += d (i, j, alpha, sigma[stack[alpha]]) + d (i, j, stack[alpha], sigma[alpha]);
                        cost[i * nv + j] -= d (i, j, alpha, sigma[alpha]) + d (i, j, stack[alpha], sigma[stack[alpha]]);
                    }
                swap (size_t, sigma[alpha], sigma[stack[alpha]]);
            }

            int32_t res = lap (nv, cost, a, b, x, y);
            best_res = (res < best_res ? res : best_res);
            stack[alpha]++;
            alpha = 1;
        }
        else
        {
            stack[alpha] = 0;
            alpha++;
        }
    }

    free (cost), free (stack), free (sigma);
    free (a), free (b), free (x), free (y);
    return best_res;
}

/**
 * @brief
 *
 * @param nv problem size parameter
 * @param nc problem size parameter
 * @return int32_t
 */
static int32_t
bap_cd (const size_t nv, const size_t nc)
{
    // Cost matrices for LAP
    int32_t *cost_nv = calloc (nv * nv, sizeof (int32_t));
    int32_t *cost_nc = calloc (nc * nc, sizeof (int32_t));

    // Auxiliary variables required for J-V LAP algorithm
    int32_t *rowsol_nv = calloc (nv, sizeof (int32_t));
    int32_t *colsol_nv = calloc (nv, sizeof (int32_t));
    int32_t *rowsol_nc = calloc (nc, sizeof (int32_t));
    int32_t *colsol_nc = calloc (nc, sizeof (int32_t));
    int32_t *x_nv = calloc (nv, sizeof (int32_t)), *y_nv = calloc (nv, sizeof (int32_t));
    int32_t *x_nc = calloc (nc, sizeof (int32_t)), *y_nc = calloc (nc, sizeof (int32_t));

    // Permutation arrays initilized to identity
    size_t *sigma_nv = calloc (nv, sizeof (size_t));
    size_t *sigma_nc = calloc (nc, sizeof (size_t));
    for (size_t i = 0; i < nv; i++)
        sigma_nv[i] = i;
    for (size_t i = 0; i < nc; i++)
        sigma_nc[i] = i;

    // ====================================================
    // Shuffle to random initilize
    // TODO: Try randomized greedy init instead
    // ====================================================

    for (size_t i = nv - 1; i > 0; i--)
    {
        size_t j = rand () % i;
        swap (size_t, sigma_nv[i], sigma_nv[j]);
    }
    for (size_t i = nc - 1; i > 0; i--)
    {
        size_t j = rand () % i;
        swap (size_t, sigma_nc[i], sigma_nc[j]);
    }

    // ====================================================
    // Coordinate-descent-like refinment
    // ====================================================

    int32_t cost_prev = 0, cost_next = -1;
    for (size_t i = 0; i < nv; i++)
        for (size_t k = 0; k < nc; k++)
            cost_prev += d (i, sigma_nv[i], k, sigma_nc[k]);

    size_t iters = 0;
    while (1)
    {
        iters++;
        for (size_t i = 0; i < nv; i++)
            for (size_t j = 0; j < nv; j++)
            {
                int32_t acc = 0;
                for (size_t k = 0; k < nc; k++)
                    acc += d (i, j, k, sigma_nc[k]);
                cost_nv[i * nv + j] = acc;
            }

        cost_next = lap (nv, cost_nv, rowsol_nv, colsol_nv, x_nv, y_nv);
        for (size_t i = 0; i < nv; i++)
            sigma_nv[i] = rowsol_nv[i];

        for (size_t i = 0; i < nc; i++)
            for (size_t j = 0; j < nc; j++)
            {
                int32_t acc = 0;
                for (size_t k = 0; k < nv; k++)
                    acc += d (k, sigma_nv[k], i, j);
                cost_nc[i * nc + j] = acc;
            }

        cost_next = lap (nc, cost_nc, rowsol_nc, colsol_nc, x_nc, y_nc);
        for (size_t i = 0; i < nc; i++)
            sigma_nc[i] = rowsol_nc[i];

        if (cost_prev == cost_next)
            break;

        cost_prev = cost_next;
    }
    printf ("Iters %d\n", iters);

    free (sigma_nv);
    free (sigma_nc);
    free (cost_nv);
    free (cost_nc);

    free (rowsol_nv), free (rowsol_nc);
    free (colsol_nv), free (colsol_nc);
    free (x_nv), free (y_nv);
    free (x_nc), free (y_nc);

    return cost_next;
}
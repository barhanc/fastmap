#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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
 * NOTE: Before including this header-file you must define a macro `#define d(i,j,k,l) ...` which is
 * used to compute the cost tensor (see: fastmap/example/).
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

    // Indices of elements of permutation sigma which are swapped in the given iteration of Heap's
    // algorithm
    size_t p = 0, q = 0;

    // Auxiliary variables required for J-V LAP algorithm
    int32_t *a = calloc (nv, sizeof (int32_t));
    int32_t *b = calloc (nv, sizeof (int32_t));
    int32_t *x = calloc (nv, sizeof (int32_t));
    int32_t *y = calloc (nv, sizeof (int32_t));

    int32_t best_res = lap (nv, cost, a, b, x, y);

    while (alpha < nc)
    {
        if (stack[alpha] < alpha)
        {
            if (alpha % 2 == 0)
                p = 0, q = alpha;
            else
                p = alpha, q = stack[alpha];

            for (size_t i = 0; i < nv; i++)
                for (size_t j = 0; j < nv; j++)
                    cost[i * nv + j] += d (i, j, p, sigma[q])
                                        + d (i, j, q, sigma[p])
                                        - d (i, j, p, sigma[p])
                                        - d (i, j, q, sigma[q]);

            swap (size_t, sigma[p], sigma[q]);

            int32_t res = lap (nv, cost, a, b, x, y);
            best_res = res < best_res ? res : best_res;

            stack[alpha]++;
            alpha = 1;
        }
        else
        {
            stack[alpha] = 0;
            alpha++;
        }
    }

    free (cost);
    free (stack);
    free (sigma);
    free (a);
    free (b);
    free (x);
    free (y);

    return best_res;
}

/**
 * @brief Implementation of the Alternating Algorithm described in arXiv:1707.07057 which is a
 * heuristic to solve the Bilinear Assignment Problem (BAP)
 * ```
 *  min_{σ ∈ S_nc} min_{v ∈ S_nv} sum_{i=1,..,nv} sum_{k=1,..,nc} d(i,v(i),k,σ(k))
 * ```
 * where d(i,j,k,l) is the cost tensor, S_n denotes the set of all permutations of the set {1,..,n}
 * and integers nv, nc describe the size of the problem instance. The algorithm first generates a
 * feasible solution to the BAP by sampling from a uniform distribution two permutations σ, v and
 * then performs a coordinate-descent-like refinment by interchangeably fixing one of the
 * permutations, solving the corresponding Linear Assignment Problem (LAP) and updating the other
 * permutation with the matching found in LAP doing so until convergence.
 *
 * NOTE: Before including this header-file you must define a macro `#define d(i,j,k,l) ...` which is
 * used to compute the cost tensor (see: fastmap/example/).
 *
 * @param nv problem size parameter
 * @param nc problem size parameter
 * @return int32_t
 */
static int32_t
bap_aa (const size_t nv, const size_t nc)
{
    // Cost matrices for LAP
    int32_t *cost_nv = calloc (nv * nv, sizeof (int32_t));
    int32_t *cost_nc = calloc (nc * nc, sizeof (int32_t));

    // Auxiliary variables required for J-V LAP algorithm
    int32_t *rowsol_nv = calloc (nv, sizeof (int32_t));
    int32_t *colsol_nv = calloc (nv, sizeof (int32_t));
    int32_t *rowsol_nc = calloc (nc, sizeof (int32_t));
    int32_t *colsol_nc = calloc (nc, sizeof (int32_t));
    int32_t *x_nv = calloc (nv, sizeof (int32_t));
    int32_t *y_nv = calloc (nv, sizeof (int32_t));
    int32_t *x_nc = calloc (nc, sizeof (int32_t));
    int32_t *y_nc = calloc (nc, sizeof (int32_t));

    // Permutation arrays randomly initialized
    size_t *sigma_nv = calloc (nv, sizeof (size_t));
    size_t *sigma_nc = calloc (nc, sizeof (size_t));

    for (size_t i = 0; i < nv; i++)
        sigma_nv[i] = i;

    for (size_t i = 0; i < nc; i++)
        sigma_nc[i] = i;

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

    // Minimum costs found in the previous and current iteration
    int32_t res_prev = 0, res_curr = -1;
    for (size_t i = 0; i < nv; i++)
        for (size_t k = 0; k < nc; k++)
            res_prev += d (i, sigma_nv[i], k, sigma_nc[k]);

    // Coordinate-descent-like refinment
    while (1)
    {
        // TODO: This is too slow (it's 2-3 OoM slower than lap() with the same theoretical time
        // complexity when nv=nc).
        // NOTE: It's basically a more advanced matmul. Notice that the structure is the same but we
        // are actually contracting a 4-D tensor which is defined by a macro.
        memset (cost_nv, 0, nv * nv * sizeof (*cost_nv));
        for (size_t k = 0; k < nc; k++)
            for (size_t i = 0; i < nv; i++)
                for (size_t j = 0; j < nv; j++)
                    cost_nv[i * nv + j] += d (i, j, k, sigma_nc[k]);

        res_curr = lap (nv, cost_nv, rowsol_nv, colsol_nv, x_nv, y_nv);
        for (size_t i = 0; i < nv; i++)
            sigma_nv[i] = rowsol_nv[i];

        // TODO: This is too slow (it's 2-3 OoM slower than lap() with the same theoretical time
        // complexity when nv=nc).
        // NOTE: It's basically a more advanced matmul. Notice that the structure is the same but we
        // are actually contracting a 4-D tensor which is defined by a macro.
        memset (cost_nc, 0, nc * nc * sizeof (*cost_nc));
        for (size_t i = 0; i < nc; i++)
            for (size_t j = 0; j < nc; j++)
                for (size_t k = 0; k < nv; k++)
                    cost_nc[i * nc + j] += d (k, sigma_nv[k], i, j);

        res_curr = lap (nc, cost_nc, rowsol_nc, colsol_nc, x_nc, y_nc);
        for (size_t i = 0; i < nc; i++)
            sigma_nc[i] = rowsol_nc[i];

        if (res_prev == res_curr)
            break;

        res_prev = res_curr;
    }

    free (sigma_nv);
    free (sigma_nc);
    free (cost_nv);
    free (cost_nc);
    free (rowsol_nv);
    free (rowsol_nc);
    free (colsol_nv);
    free (colsol_nc);
    free (x_nv);
    free (y_nv);
    free (x_nc);
    free (y_nc);

    return res_curr;
}
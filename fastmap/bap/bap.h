#include <stdbool.h>
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
 *  min_{σ ∈ S_nc} min_{v ∈ S_nv} sum_{i=0,..,nv-1} sum_{k=0,..,nc-1} d(i,v(i),k,σ(k))
 * ```
 * where d(i,j,k,l) is the cost tensor, S_n denotes the set of all permutations of the set
 * {0,..,n-1} and integers nv, nc describe the size of the problem instance. The algorithm generates
 * all permutations of S_nc using Heap's algorithm and for every generated permutation σ solves the
 * following Linear Assignment Problem (LAP) in order to find the optimal permutation v ∈ S_nv
 * ```
 *   min_{v ∈ S_nv} sum_{i=0,..,nv-1} ( sum_{k=0,..,nc-1} d(i,v(i),k,σ(k)) ) .
 * ```
 * NOTE: Before including this header-file you must define a macro `#define d(i,j,k,l) ...` which is
 * used to compute the cost tensor.
 *
 * @param nv problem size parameter
 * @param nc problem size parameter
 * @return minimal value of the cost function
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
    int32_t *x = calloc (nv, sizeof (int32_t));
    int32_t *y = calloc (nv, sizeof (int32_t));

    int32_t best_res = lap (nv, cost, x, y);

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

            int32_t res = lap (nv, cost, x, y);
            best_res = res < best_res ? res : best_res;

            swap (size_t, sigma[p], sigma[q]);
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
    free (x);
    free (y);

    return best_res;
}

/**
 * @brief Implementation of the Alternating Algorithm described in arXiv:1707.07057 which is a
 * heuristic to solve the Bilinear Assignment Problem (BAP)
 * ```
 *  min_{σ ∈ S_nc} min_{v ∈ S_nv} sum_{i=0,..,nv-1} sum_{k=0,..,nc-1} d(i,v(i),k,σ(k))
 * ```
 * where d(i,j,k,l) is the cost tensor, S_n denotes the set of all permutations of the set
 * {0,..,n-1} and integers nv, nc describe the size of the problem instance. The algorithm first
 * generates a feasible solution to the BAP by sampling from a uniform distribution two permutations
 * σ, v and then performs a coordinate-descent-like refinment by interchangeably fixing one of the
 * permutations, solving the corresponding Linear Assignment Problem (LAP) and updating the other
 * permutation with the matching found in LAP doing so until convergence.
 *
 * NOTE: Before including this header-file you must define a macro `#define d(i,j,k,l) ...` which is
 * used to compute the cost tensor.
 *
 * @param nv problem size parameter
 * @param nc problem size parameter
 * @return approximation of the minimal value of the cost function
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
        memset (cost_nv, 0, nv * nv * sizeof (*cost_nv));
        for (size_t k = 0; k < nc; k++)
            for (size_t i = 0; i < nv; i++)
                for (size_t j = 0; j < nv; j++)
                    cost_nv[i * nv + j] += d (i, j, k, sigma_nc[k]);

        res_curr = lap (nv, cost_nv, rowsol_nv, colsol_nv);
        for (size_t i = 0; i < nv; i++)
            sigma_nv[i] = rowsol_nv[i];

        memset (cost_nc, 0, nc * nc * sizeof (*cost_nc));
        for (size_t i = 0; i < nc; i++)
            for (size_t j = 0; j < nc; j++)
                for (size_t k = 0; k < nv; k++)
                    cost_nc[i * nc + j] += d (k, sigma_nv[k], i, j);

        res_curr = lap (nc, cost_nc, rowsol_nc, colsol_nc);
        for (size_t i = 0; i < nc; i++)
            sigma_nc[i] = rowsol_nc[i];

        if (res_prev == res_curr)
            break;

        res_prev = res_curr;
    }

    free (cost_nv);
    free (cost_nc);
    free (sigma_nv);
    free (sigma_nc);
    free (rowsol_nv);
    free (rowsol_nc);
    free (colsol_nv);
    free (colsol_nc);

    return res_curr;
}

#include "stack.h"

typedef struct Node
{
    bool *available;
    int32_t bound;
    size_t *sigma;
    size_t n;
} Node;

/**
 * @brief Implementation of a branch-and-bound (see: https://en.wikipedia.org/wiki/Branch_and_bound)
 * algorithm solving Bilinear Assignment Problem (BAP)
 * ```
 *  min_{σ ∈ S_nc} min_{v ∈ S_nv} sum_{i=0,..,nv-1} sum_{k=0,..,nc-1} d(i,v(i),k,σ(k))    (0)
 * ```
 * where d(i,j,k,l) is the cost tensor, S_n denotes the set of all permutations of the set
 * {0,..,n-1} and integers nv, nc describe the size of the problem instance. In the algorithm we
 * first compute an upper bound on the cost value using Alternating Algorithm heuristic (see:
 * bap_aa() function) and later perform a search over all permutation prefixes of σ computing the
 * lower bound on the cost value and pruning the part of the tree that certainly does not contain
 * minimum. The lower bound is computed as follows. Assume we are in the node of the tree having a
 * prefix of length n' < nc of permutation σ and let C(σ) be the cost
 * ```
 *  C(σ) = min_{v ∈ S_nv} sum_{i=0,..,nv-1} sum_{k=0,..,nc-1} d(i,v(i),k,σ(k))
 * ```
 * for any permutation σ with the given prefix. It is easy to see that
 * ```
 *  forall σ : C(σ) >= min_{v ∈ S_nv} sum_{i=0,..,nv-1} cost[i,v(i)]                      (1)
 * ```
 * where
 * ```
 *  cost[i,j] = sum_{k=0,..,n'-1} d(i,j,k,σ(k)) + min sum_{k=n',..,nc-1} d(i,j,k,σ(k))    (2)
 * ```
 * where in (2) we minimize over possible assignments of the remaining m = nc - n' elements. It's
 * clear that every element of cost[i,j] can be computed using lap in O(m**3) time and we then may
 * compute the lower bound (1) also using lap in O(nv**3) time. If the lower bound in the node is
 * greater than upper bound, we prune this part of the tree. Note that if we were to visit every
 * node this algorithm needs much more lap calls than simple brute-force (see: bap_bf()) since there
 * are `sum_{k=1,..,nc} (nc choose k) * k! > nc!` nodes and in each node we need to solve one LAP
 * for a cost matrix of size nv x nv and nv**2 LAPs for cost matrices of size O(nc) x O(nc), while
 * brute-force needs to solve only nc! LAPs for cost matrices of size nv x nv.
 *
 * NOTE: Before including this header-file you must define a macro `#define d(i,j,k,l) ...` which is
 * used to compute the cost tensor.
 *
 * @param nv problem size parameter
 * @param nc problem size parameter
 * @param repeats number of times we compute initial upper bound using "aa" heuristic
 * @param max_evals TODO:...
 * @return minimal value of the cost function
 */
static int32_t
bap_bb (const size_t nv, const size_t nc, const int repeats)
{
    // Upper bound on cost
    int32_t B = 0x7FFFFFFF;

    //  1. Using a heuristic, find a solution to the optimization problem. Store its value in B. B
    //     will denote the best solution found so far, and will be used as an upper bound on
    //     candidate solutions.

    for (int i = 0; i < repeats; i++)
    {
        int32_t bound = bap_aa (nv, nc);
        B = bound < B ? bound : B;
    }

    // Cost matrix for LAP
    int32_t *cost_nv = calloc (nv * nv, sizeof (int32_t));
    int32_t *cost_m = calloc (nc * nc, sizeof (int32_t));

    // Auxiliary variables required for J-V LAP algorithm
    int32_t *rowsol_nv = calloc (nv, sizeof (int32_t));
    int32_t *colsol_nv = calloc (nv, sizeof (int32_t));
    int32_t *rowsol_m = calloc (nc, sizeof (int32_t));
    int32_t *colsol_m = calloc (nc, sizeof (int32_t));

    // 2. Initialize a queue to hold a partial solution with none of the variables of the problem
    //    assigned.

    Stack *stack = stack_alloc ();       // LIFO queue
    Node *node = malloc (sizeof (Node)); // Node of the search tree

    node->n = 0;
    node->bound = 0;
    node->sigma = calloc (nc, sizeof (size_t));
    node->available = malloc (nc * sizeof (bool));
    memset (node->available, true, nc * sizeof (bool));

    push (stack, (void *)node);

    // 3. Loop until the queue is empty:
    while (stack->size > 0)
    {
        // 1. Take a node N off the queue.
        node = (Node *)pop (stack);

        // 2. If N represents a single candidate solution x and cost(x) < B, then x is the best
        //    solution so far. Record it and set B = cost(x).
        if (node->n == nc)
        {
            B = node->bound < B ? node->bound : B;
        }
        // 3. Else, branch on N to produce new nodes Ni.
        else
        {
            for (size_t candidate = 0; candidate < nc; candidate++)
            {
                if (!node->available[candidate])
                    continue;

                Node *new_node = malloc (sizeof (Node));

                new_node->sigma = calloc (nc, sizeof (size_t));
                new_node->available = calloc (nc, sizeof (bool));

                memcpy (new_node->sigma, node->sigma, nc * sizeof (size_t));
                memcpy (new_node->available, node->available, nc * sizeof (bool));

                new_node->n = node->n + 1;
                new_node->sigma[node->n] = candidate;
                new_node->available[candidate] = false;

                size_t m = nc - new_node->n;

                memset (cost_nv, 0, nv * nv * sizeof (int32_t));
                memset (cost_m, 0, nc * nc * sizeof (int32_t));
                memset (rowsol_m, 0, nc * sizeof (int32_t));
                memset (colsol_m, 0, nc * sizeof (int32_t));

                for (size_t i = 0; i < nv; i++)
                    for (size_t j = 0; j < nv; j++)
                    {
                        size_t l = 0;
                        for (size_t c = 0; c < nc; c++)
                        {
                            if (!new_node->available[c])
                                continue;

                            for (size_t k = 0; k < m; k++)
                                cost_m[k * m + l] = d (i, j, k + new_node->n, c);
                            l++;
                        }

                        if (m > 0)
                            cost_nv[i * nv + j] = lap (m, cost_m, rowsol_m, colsol_m);

                        for (size_t k = 0; k < new_node->n; k++)
                            cost_nv[i * nv + j] += d (i, j, k, new_node->sigma[k]);
                    }

                new_node->bound = lap (nv, cost_nv, rowsol_nv, colsol_nv);

                // 1. If bound(Ni) > B, do nothing; since the lower bound on this node is greater
                //    than the upper bound of the problem, it will never lead to the optimal
                //    solution, and can be discarded.
                if (new_node->bound >= B)
                {
                    free (new_node->available);
                    free (new_node->sigma);
                    free (new_node);
                }
                // 2. Else, store Ni on the queue.
                else
                {
                    push (stack, (void *)new_node);
                }
            }
        }

        free (node->available);
        free (node->sigma);
        free (node);
    }

    free (stack);

    free (cost_nv);
    free (rowsol_nv);
    free (colsol_nv);

    free (cost_m);
    free (rowsol_m);
    free (colsol_m);

    return B;
}
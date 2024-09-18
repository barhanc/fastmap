#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "lapf.h"

#define swap(type, x, y) \
    {                    \
        type SWAP = x;   \
        x = y;           \
        y = SWAP;        \
    }

/**
 * @brief Function to generate a random doubly stochastic matrix. It takes as an argument a pointer
 * X to the array which will store the doubly stochastic matrix in a linearized row-major fashion.
 * It initializes matrix X with random numbers between 0 and 1 sampled uniformly and then repeatadly
 * makes the matrix row-stochastic and column-stochastic until convergence condition is met. This
 * scheme makes X eventually converge to the random doubly stochastic matrix according to paper:
 *
 * Richard Sinkhorn. "A Relationship Between Arbitrary Positive Matrices and Doubly Stochastic
 * Matrices." Ann. Math. Statist. 35 (2) 876 - 879, June, 1964.
 * https://doi.org/10.1214/aoms/1177703591
 *
 * @param n size of the matrix (must be a square matrix)
 * @param X pointer to the linearized matrix
 * @param eps tolerance for termination. Iteration terminates when all(abs(X.sum(0) - 1) <= eps) and
 * all(abs(X.sum(1) - 1) <= eps).
 */
static void
random_bistochastic (const size_t n, double *X, const double eps)
{
    for (size_t r = 0; r < n; r++)
        for (size_t c = 0; c < n; c++)
            X[r * n + c] = (double)rand () / (double)RAND_MAX;

    double *rsum = calloc (n, sizeof (double));
    double *csum = calloc (n, sizeof (double));
    bool converged = false;

    while (!converged)
    {
        memset (rsum, 0, n * sizeof (*rsum));
        memset (csum, 0, n * sizeof (*csum));

        for (size_t c = 0; c < n; c++)
            for (size_t r = 0; r < n; r++)
                rsum[c] += X[r * n + c];

        for (size_t c = 0; c < n; c++)
            for (size_t r = 0; r < n; r++)
                X[r * n + c] /= rsum[c];

        for (size_t r = 0; r < n; r++)
            for (size_t c = 0; c < n; c++)
                csum[r] += X[r * n + c];

        for (size_t r = 0; r < n; r++)
            for (size_t c = 0; c < n; c++)
                X[r * n + c] /= csum[r];

        memset (rsum, 0, n * sizeof (*rsum));
        memset (csum, 0, n * sizeof (*csum));

        for (size_t r = 0; r < n; r++)
            for (size_t c = 0; c < n; c++)
            {
                rsum[c] += X[r * n + c];
                csum[r] += X[r * n + c];
            }

        converged = true;
        for (size_t i = 0; i < n; i++)
            if (fabs (rsum[i] - 1) > eps || fabs (csum[i] - 1) > eps)
            {
                converged = false;
                break;
            }
    }

    free (rsum);
    free (csum);

    return;
}

/**
 * @brief Approximate algorithm solving Quadratic Assignment Problem (QAP)
 * ```
 *  min_{σ ∈ S_nc} sum_{i=0,..,nc-1} sum_{j=0,..,nc-1} d(i,σ(i),j,σ(j))
 * ```
 * where d(i,j,k,l) is the cost array, S_n denotes the set of all permutations of the set {0,..,n-1}
 * and integer nc describes the size of the problem instance. Using permutation matrix P we may
 * write this problem as a minimization of a function f(P) defined as
 * ```
 *  f(P) = sum_{i,j,k,l} d(i,j,k,l) * P[i,j] * P[k,l]   .
 * ```
 * The implemented algorithm is a generalization to Lawler QAP problem of the algorithm described in
 * paper:
 *
 * Vogelstein JT, Conroy JM, Lyzinski V, Podrazik LJ, Kratzer SG, Harley ET, et al. (2015) Fast
 * Approximate Quadratic Programming for Graph Matching. PLoS ONE 10(4): e0121002.
 * https://doi.org/10.1371/journal.pone.0121002
 *
 * NOTE: Before including this header-file you must define a macro `#define d(i,j,k,l) ...` which is
 * used to compute the cost tensor.
 *
 * @param nc problem size parameter
 * @param maxiter upper bound on iteration count
 * @param tol stopping condition when ||P_{i}-P_{i+1}||_Frobenius < tol
 * @return approximation of the minimal value of the cost function
 */
static double
qap_faq (const size_t nc, const size_t maxiter, const double tol)
{
    // Auxiliary variables required for lap algorithm
    int32_t *rowsol_nc = calloc (nc, sizeof (int32_t));
    int32_t *colsol_nc = calloc (nc, sizeof (int32_t));

    // Gradient of the objective
    double *grad_f = calloc (nc * nc, sizeof (double));

    // Doubly stochastic matrix being a solution to the relaxed problem (rQAP)
    double *P = calloc (nc * nc, sizeof (double));

    // Point minimizing the linear approximation of the objective in Frank--Wolfe algorithm
    double *Q = calloc (nc * nc, sizeof (double));

    // 1: Random initialization
    random_bistochastic (nc, P, 1e-8);

    // 2: While stopping criteria not met do
    for (size_t iter = maxiter; --iter;)
    {
        // 3: Compute the gradient of objective at the current point

        memset (grad_f, 0, nc * nc * sizeof (double));
        for (size_t k = 0; k < nc; k++)
            for (size_t l = 0; l < nc; l++)
                for (size_t i = 0; i < nc; i++)
                    for (size_t j = 0; j < nc; j++)
                        grad_f[k * nc + l] += (d (i, j, k, l) + d (k, l, i, j)) * P[i * nc + j];

        // 4:  Compute the direction Q

        lapf (nc, grad_f, rowsol_nc, colsol_nc);
        memset (Q, 0, nc * nc * sizeof (double));
        for (size_t i = 0; i < nc; i++)
            Q[i * nc + colsol_nc[i]] = 1;

        // 5: Compute the step size alpha

        double a = 0, b = 0, alpha = 0;

        for (size_t i = 0; i < nc; i++)
            for (size_t j = 0; j < nc; j++)
                for (size_t k = 0; k < nc; k++)
                    for (size_t l = 0; l < nc; l++)
                    {
                        double R_ij = Q[i * nc + j] - P[i * nc + j];
                        double R_kl = Q[k * nc + l] - P[k * nc + l];
                        a += d (i, j, k, l) * R_ij * R_kl;
                        b += d (i, j, k, l) * (R_ij * P[k * nc + l] + R_kl * P[i * nc + j]);
                    }

        if (a > 0 && 0 <= -b / (2 * a) && -b / (2 * a) <= 1)
            alpha = -b / (2 * a);
        else
            alpha = a + b > 0 ? 0 : 1;

        // 6: Update P
        double norm2 = 0;
        for (size_t i = 0; i < nc * nc; i++)
        {
            double P_next = (1 - alpha) * P[i] + alpha * Q[i];
            norm2 += (P[i] - P_next) * (P[i] - P_next);
            P[i] = P_next;
        }

        if (norm2 <= tol * tol)
            break;
    }
    // 7: End loop

    // 8: Obtain solution

    for (size_t i = 0; i < nc * nc; i++)
        P[i] *= -1;

    lapf (nc, P, rowsol_nc, colsol_nc);

    double res = 0;
    for (size_t i = 0; i < nc; i++)
        for (size_t j = 0; j < nc; j++)
            res += d (i, rowsol_nc[i], j, rowsol_nc[j]);

    free (rowsol_nc);
    free (colsol_nc);
    free (grad_f);
    free (P);
    free (Q);

    return res;
}

/**
 * @brief Implementation of a coordinate-descent heuristic solving Quadratic Assignment Problem
 * (QAP)
 * ```
 *  min_{σ ∈ S_nc} sum_{i=0,..,nc-1} sum_{j=0,..,nc-1} d(i,σ(i),j,σ(j))
 * ```
 * where d(i,j,k,l) is the cost tensor, S_n denotes the set of all permutations of the set
 * {0,..,n-1} and integer nc describes the size of the problem instance. The algorithm actually
 * solves heuristically the Bilinear Assignment Problem (BAP)
 * ```
 *  min_{v ∈ S_nv} min_{σ ∈ S_nc} sum_{i=0,..,nc-1} sum_{j=0,..,nc-1} d(i,v(i),j,σ(j))
 * ```
 * using the Alternating Algorithm (for details see: 'bap.h' file). After the AA algorithm has
 * converged we return the smaller of the values
 * ```
 *  sum_{i=0,..,nc-1} sum_{j=0,..,nc-1} d(i,σ(i),j,σ(j))
 *  sum_{i=0,..,nc-1} sum_{j=0,..,nc-1} d(i,v(i),j,v(j))
 * ```
 * which provides a heuristic solution to the QAP problem.
 *
 * NOTE: Before including this header-file you must define a macro `#define d(i,j,k,l) ...` which is
 * used to compute the cost tensor.
 *
 * @param nc problem size parameter
 * @return approximation of the minimal value of the cost function
 */
static double
qap_aa (const size_t nc)
{
    // Cost matrices for LAP
    double *cost_nc = calloc (nc * nc, sizeof (double));

    // Auxiliary variables required for J-V LAP algorithm
    int32_t *rowsol_nc = calloc (nc, sizeof (int32_t));
    int32_t *colsol_nc = calloc (nc, sizeof (int32_t));

    // Permutation arrays randomly initialized. Note that permutations `sigma_nc_1` and `sigma_nc_2`
    // are initialized to the same random permutation by design.
    size_t *sigma_nc_1 = calloc (nc, sizeof (size_t));
    size_t *sigma_nc_2 = calloc (nc, sizeof (size_t));

    for (size_t i = 0; i < nc; i++)
        sigma_nc_1[i] = i;

    for (size_t i = nc - 1; i > 0; i--)
    {
        size_t j = rand () % i;
        swap (size_t, sigma_nc_1[i], sigma_nc_1[j]);
    }

    for (size_t i = 0; i < nc; i++)
        sigma_nc_2[i] = sigma_nc_1[i];

    // Minimum costs found in the previous and current iteration of coordinate-descent
    double res_prev = 0, res_curr = -1;

    for (size_t i = 0; i < nc; i++)
        for (size_t j = 0; j < nc; j++)
            res_prev += d (i, sigma_nc_1[i], j, sigma_nc_2[j]);

    // Coordinate-descent-like refinment
    while (1)
    {
        memset (cost_nc, 0, nc * nc * sizeof (*cost_nc));
        for (size_t k = 0; k < nc; k++)
            for (size_t i = 0; i < nc; i++)
                for (size_t j = 0; j < nc; j++)
                    cost_nc[i * nc + j] += d (i, j, k, sigma_nc_2[k]);

        res_curr = lapf (nc, cost_nc, rowsol_nc, colsol_nc);
        for (size_t i = 0; i < nc; i++)
            sigma_nc_1[i] = rowsol_nc[i];

        memset (cost_nc, 0, nc * nc * sizeof (*cost_nc));
        for (size_t i = 0; i < nc; i++)
            for (size_t j = 0; j < nc; j++)
                for (size_t k = 0; k < nc; k++)
                    cost_nc[i * nc + j] += d (k, sigma_nc_1[k], i, j);

        res_curr = lapf (nc, cost_nc, rowsol_nc, colsol_nc);
        for (size_t i = 0; i < nc; i++)
            sigma_nc_2[i] = rowsol_nc[i];

        if (res_prev == res_curr)
            break;

        res_prev = res_curr;
    }

    double best_res_1 = 0;
    for (size_t i = 0; i < nc; i++)
        for (size_t j = 0; j < nc; j++)
            best_res_1 += d (i, sigma_nc_1[i], j, sigma_nc_1[j]);

    double best_res_2 = 0;
    for (size_t i = 0; i < nc; i++)
        for (size_t j = 0; j < nc; j++)
            best_res_2 += d (i, sigma_nc_2[i], j, sigma_nc_2[j]);

    free (cost_nc);
    free (rowsol_nc);
    free (colsol_nc);
    free (sigma_nc_1);
    free (sigma_nc_2);

    return best_res_1 < best_res_2 ? best_res_1 : best_res_2;
}

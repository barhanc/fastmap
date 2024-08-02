#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "lapf.h"

void
_invert_array (double *arr, size_t size)
{
    for (size_t i = 0; i < size; i++)
        arr[i] = 1.0 / arr[i];
}

double *
_doubly_stochastic (const size_t nc, double eps)
{
    double *P = (double *)malloc (nc * nc * sizeof (double));
    double *P_eps = (double *)malloc (nc * nc * sizeof (double));
    double *c = (double *)calloc (nc, sizeof (double));
    double *r = (double *)calloc (nc, sizeof (double));
    double *rowsum = (double *)malloc (nc * sizeof (double));
    double *colsum = (double *)malloc (nc * sizeof (double));

    for (size_t i = 0; i < nc; i++)
        for (size_t j = 0; j < nc; j++)
            P[i * nc + j] = (double)rand () / RAND_MAX;

    for (size_t i = 0; i < nc; i++)
        for (size_t j = 0; j < nc; j++)
            c[i] += P[j * nc + i];

    _invert_array (c, nc);

    for (size_t i = 0; i < nc; i++)
        for (size_t j = 0; j < nc; j++)
            r[i] += P[i * nc + j] * c[j];

    _invert_array (r, nc);

    memcpy (P_eps, P, nc * nc * sizeof (double));

    int break_flag;
    for (size_t iter = 1000; iter--;)
    {
        break_flag = 0;

        memset (rowsum, 0, nc * sizeof (*rowsum));
        memset (colsum, 0, nc * sizeof (*colsum));

        for (size_t i = 0; i < nc; i++)
            for (size_t j = 0; j < nc; j++)
            {
                rowsum[i] += P_eps[i * nc + j];
                colsum[j] += P_eps[i * nc + j];
            }

        for (size_t i = 0; i < nc; i++)
        {
            if (abs (rowsum[i] - 1.0) > eps || abs (colsum[i] - 1.0) > eps)
            {
                break_flag = 1;
                break;
            }
        }

        if (break_flag == 0)
            break;

        memset (c, 0, nc * sizeof (*c));
        for (size_t i = 0; i < nc; i++)
            for (size_t j = 0; j < nc; j++)
                c[i] += r[j] * P[j * nc + i];

        _invert_array (c, nc);

        memset (r, 0, nc * sizeof (*r));
        for (size_t i = 0; i < nc; i++)
            for (size_t j = 0; j < nc; j++)
                r[i] += P[i * nc + j] * c[j];

        _invert_array (r, nc);

        for (size_t i = 0; i < nc; i++)
            for (size_t j = 0; j < nc; j++)
                P_eps[i * nc + j] = r[i] * P[i * nc + j] * c[j];
    }

    free (P);
    free (c);
    free (r);
    free (rowsum);
    free (colsum);

    return P_eps;
}

/**
 * @brief Approximate algorithm solving Quadratic Assignment Problem (QAP)
 * ```
 *  min_{σ ∈ S_nc} sum_{i=0,..,nc-1} sum_{j=0,..,nc-1} d(i,σ(i),j,σ(j))
 * ```
 * where d(i,j,k,l) is the cost tensor, S_n denotes the set of all permutations of the set
 * {0,..,n-1} and integer nc describe the size of the problem instance. Using permutation matrix P
 * we may write this problem as a minimization of a function f(P)
 * ```
 *  f(P) = sum_{i,j,k,l} d(i,j,k,l) * P[i,j] * P[k,l]   .
 * ```
 * The algorithm is based on paper
 * https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0121002.
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

    // 1: Choose an initialization, P = 1 1^T / n (barycenter initialization)
    for (size_t i = 0; i < nc * nc; i++)
        P[i] = 1 / (double)nc;

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

        double vertex = -b / (2 * a);

        if (a > 0 && 0 <= vertex && vertex <= 1)
            alpha = vertex;
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
            res += d (i, colsol_nc[i], j, colsol_nc[j]);

    free (rowsol_nc);
    free (colsol_nc);
    free (grad_f);
    free (P);
    free (Q);

    return res;
}

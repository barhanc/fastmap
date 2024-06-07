#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "lapf.h"

#define _rand_double (double) rand() / RAND_MAX
#define d(i, j, k, l) abs(A[i * nc + j] - B[k * nc + l]) // metric

void
_invert_array(double* arr, size_t size)
{
    for(size_t i=0; i<size; i++)
        arr[i] = 1.0 / arr[i];
}

double*
_doubly_stochastic(const size_t nc, double eps)
{
    double *P = (double*) malloc(nc * nc * sizeof(double));
    double *P_eps = (double*) malloc(nc * nc * sizeof(double));
    double* c = (double*) calloc(nc, sizeof(double));
    double* r = (double*) calloc(nc, sizeof(double));
    double* rowsum = (double*) malloc(nc * sizeof(double));
    double* colsum = (double*) malloc(nc * sizeof(double));

    for(size_t i=0; i<nc; i++)
        for(size_t j=0; j<nc; j++)
            P[i * nc + j] = _rand_double;

    for(size_t i=0; i<nc; i++)
        for(size_t j=0; j<nc; j++)
            c[i] += P[j * nc + i];

    _invert_array(c, nc);
    
    for(size_t i=0; i<nc; i++)
        for(size_t j=0; j<nc; j++)
            r[i] += P[i * nc + j] * c[j];
    
    _invert_array(r, nc);

    memcpy(P_eps, P, nc * nc * sizeof(double));

    int break_flag;
    for (size_t iter = 1000; iter--;)
    {
        break_flag = 0;

        memset (rowsum, 0, nc * sizeof (*rowsum));
        memset (colsum, 0, nc * sizeof (*colsum));

        for(size_t i=0; i<nc; i++)
            for(size_t j=0; j<nc; j++)
            {
                rowsum[i] += P_eps[i * nc + j];
                colsum[j] += P_eps[i * nc + j];
            }
        
        for(size_t i=0; i<nc; i++)
        {
            if (abs(rowsum[i] - 1.0) > eps || abs(colsum[i] - 1.0) > eps)
            {
                break_flag = 1;
                break;
            }
        }

        if (break_flag == 0)
            break;

        memset (c, 0, nc * sizeof (*c));
        for(size_t i=0; i<nc; i++)
            for(size_t j=0; j<nc; j++)
                c[i] += r[j] * P[j * nc + i];

        _invert_array(c, nc);

        memset (r, 0, nc * sizeof (*r));
        for(size_t i=0; i<nc; i++)
            for(size_t j=0; j<nc; j++)
                r[i] += P[i * nc + j] * c[j];
        
        _invert_array(r, nc);

        for(size_t i=0; i<nc; i++)
            for(size_t j=0; j<nc; j++)
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
 *  min_{σ ∈ S_nc} min_{v ∈ S_nc} sum_{i=1,..,nc} sum_{k=1,..,nc} d(i,v(i),k,σ(k))
 * ```
 * where d(i,j,k,l) is the cost tensor, S_n denotes the set of all permutations of the set {1,..,n}
 * and integer nc describe the size of the problem instance. The algorithm is based on paper 
 * https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0121002
 * 
 * NOTE: Before including this header-file you must define a macro `#define d(i,j,k,l) ...` which is
 * used to compute the cost tensor.
 *
 * @param nc problem size parameter
 * @param A first cost matrix
 * @param B second cost matrix
 * @param maxiter upper bound on iteration count
 * @param use_k when set to 1, then P_0 is set to (J + K) / 2
 * @param tol stopping condition when ||P_i - P_(i+1)||_F < tol
 * @return minimal value of the cost function
 */
static int64_t
qap_faq (const size_t nc, const int32_t* A, const int32_t* B, const size_t maxiter, const bool use_k, double tol)
{
    // Auxiliary variables required for lap algorithm
    int32_t *rowsol_nc = (int32_t*) calloc(nc, sizeof (int32_t));
    int32_t *colsol_nc = (int32_t*) calloc(nc, sizeof (int32_t));
    double lapres = 0;

    // Gradient of the objective
    double *grad_f = (double*) calloc(nc * nc, sizeof (double));

    // Point minimizing Taylor series approximation to f(P)
    double *Q = (double*) calloc(nc * nc, sizeof (double));

    // 1: Choose an initialization, P(0) = 1 1^T / n
    double *P = (double*) calloc(nc * nc, sizeof (double));
    double *P_next = (double*) calloc(nc * nc, sizeof (double));

    for (size_t i = 0; i < nc; i++)
        for (size_t j = 0; j < nc; j++)
            P[i * nc + j] = 1 / (double) nc;
    
    // 1.1 Generate other initial P(0) using doubly stochastic matrix
    if (use_k)
    {
        double *K = _doubly_stochastic(nc, 0.001);

        for (size_t i = 0; i < nc; i++)
            for (size_t j = 0; j < nc; j++)
                P[i * nc + j] = (P[i * nc + j] + K[i * nc + j]) / 2;
    }

    // used to calculate Frobenius norm for stopping criteria
    double eval;

    // 2: while stopping criteria not met do
    for (size_t iter = maxiter; --iter;)
    {
        // 3: Compute the gradient of f at the current point
        memset (grad_f, 0, nc * nc * sizeof (*grad_f));
        for (size_t x = 0; x < nc; x++)
            for (size_t y = 0; y < nc; y++)
                for (size_t i = 0; i < nc; i++)
                    for (size_t j = 0; j < nc; j++)
                        grad_f[x * nc + y] = (d(i, x, j, y) + d(x, i, y, j)) * P[i * nc + j];

        // 4:  Compute the direction Q(i)
        lapres = lapf(nc, grad_f, rowsol_nc, colsol_nc);

        memset(Q, 0, nc * nc * sizeof (*Q));
        for (size_t i = 0; i < nc; i++)
            Q[i * nc + colsol_nc[i]] = 1.0;

        // 5: Compute the step size alpha(i)
        double a = 0, b = 0, alpha = 0;
        for (size_t i = 0; i < nc; i++)
            for (size_t j = 0; j < nc; j++)
            {
                int32_t l_1 = colsol_nc[i], l_2 = colsol_nc[j];
                a += d(i, j, l_1, l_2) * Q[i * nc + l_1] * Q[j * nc + l_2];

                for (size_t k = 0; k < nc; k++)
                {
                    b += d(i, j, l_1, k) * P[j * nc + k] * Q[i * nc + l_1];
                    b += d(i, j, k, l_2) * Q[j * nc + l_2] * P[i * nc + k];
                }
            }

        double crit = -b / (2 * a);
        if (0 <= crit && crit <= 1)
            alpha = crit;
        else
            alpha = crit > 1 ? 1.0 : 0.0;

        // 6: Update P(i)
        for (size_t i = 0; i < nc; i++)
            for (size_t j = 0; j < nc; j++)
                P_next[i * nc + j] = P[i * nc + j] + alpha * Q[i * nc + j];

        // 7: end while either by condition or iter cout reached
        eval = 0;
        for (size_t i = 0; i < nc; i++)
            for (size_t j = 0; j < nc; j++)
            {
                eval += (P[i * nc + j] - P_next[i * nc + j]) * (P[i * nc + j] - P_next[i * nc + j]);
                P[i * nc + j] = P_next[i * nc + j];
            }

        if (sqrt(eval) < tol)
            break;       
    }

    // 8: Obtain solution
    lapres = lapf(nc, P, rowsol_nc, colsol_nc);
    int64_t res = 0;
    for (size_t i = 0; i < nc; i++)
        for (size_t j = 0; j < nc; j++)
            res += d (i, j, colsol_nc[i], colsol_nc[j]);

    free (rowsol_nc);
    free (colsol_nc);
    free (grad_f);
    free (Q);
    free (P);
    free (P_next);

    return res;
}


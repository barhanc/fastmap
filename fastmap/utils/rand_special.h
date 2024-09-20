#ifndef RAND_SPECIAL_H
#define RAND_SPECIAL_H

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Generates a random doubly stochastic matrix. It takes as an argument a pointer X to the
 * array which will store the doubly stochastic matrix in a linearized row-major fashion. It
 * initializes matrix X with random numbers between 0 and 1 sampled uniformly and then repeatadly
 * makes the matrix row-stochastic and column-stochastic until convergence condition is met. This
 * scheme makes X eventually converge to the random doubly stochastic matrix according to paper
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

#endif
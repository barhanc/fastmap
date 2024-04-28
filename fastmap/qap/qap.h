#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "lap.h"

#define d(i, j, k, l) 0

static int64_t
qap_faq (const size_t nc, const size_t maxiter)
{
    // Auxiliary variables required for lap algorithm
    int32_t *rowsol_nc = calloc (nc, sizeof (int32_t));
    int32_t *colsol_nc = calloc (nc, sizeof (int32_t));
    int32_t *x_nc = calloc (nc, sizeof (int32_t));
    int32_t *y_nc = calloc (nc, sizeof (int32_t));
    double lapres = 0;

    // Gradient of the objective
    double *grad_f = calloc (nc * nc, sizeof (double));

    // Point minimizing Taylor series approximation to f(P)
    double *Q = calloc (nc * nc, sizeof (double));

    // 1: Choose an initialization, P(0) = 1 1^T / n
    double *P = calloc (nc * nc, sizeof (double));
    for (size_t i = 0; i < nc; i++)
        for (size_t j = 0; j < nc; j++)
            P[i * nc + j] = 1 / (double)nc;

    // 2: while stopping criteria not met do
    for (size_t iter = maxiter; --iter;)
    {
        // 3: Compute the gradient of f at the current point
        memset (grad_f, 0, nc * nc * sizeof (*grad_f));
        for (size_t x = 0; x < nc; x++)
            for (size_t y = 0; y < nc; y++)
                for (size_t i = 0; i < nc; i++)
                    for (size_t j = 0; j < nc; j++)
                        grad_f[x * nc + y] += (d (i, x, j, y) + d (x, i, y, j)) * P[i * nc + j];

        // 4:  Compute the direction Q(i)
        lapres = lap (nc, grad_f, rowsol_nc, colsol_nc, x_nc, y_nc);

        memset (Q, 0, nc * nc * sizeof (*Q));
        for (size_t i = 0; i < nc; i++)
            Q[i * nc + colsol_nc[i]] = 1.0;

        // 5: Compute the step size alpha(i)
        double a = 0, b = 0, alpha = 0;
        for (size_t i = 0; i < nc; i++)
            for (size_t j = 0; j < nc; j++)
                for (size_t k = 0; k < nc; k++)
                    for (size_t l = 0; l < nc; l++)
                    {
                        a += d (i, j, k, l) * Q[i * nc + k] * Q[j * nc + l];
                        b += d (i, j, k, l) * (P[j * nc + l] * Q[i * nc + k] + Q[j * nc + l] * P[i * nc + k]);
                    }

        double crit = -b / (2 * a);
        if (0 <= crit && crit <= 1)
            alpha = crit;
        else
            alpha = a + b > 0 ? 0.0 : 1.0;

        // 6: Update P(i)
        for (size_t i = 0; i < nc; i++)
            for (size_t j = 0; j < nc; j++)
                P[i * nc + j] = P[i * nc + j] + alpha * Q[i * nc + j];
    }
    // 7: end while

    // 8: Obtain solution
    lapres = lap (nc, P, rowsol_nc, colsol_nc, x_nc, y_nc);
    int32_t res = 0;
    for (size_t i = 0; i < nc; i++)
        for (size_t j = 0; j < nc; j++)
            res += d (i, j, colsol_nc[i], colsol_nc[j]);

    free (rowsol_nc);
    free (colsol_nc);
    free (x_nc);
    free (y_nc);
    free (grad_f);
    free (Q);
    free (P);

    return res;
}
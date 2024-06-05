#include <math.h>
#include <stdio.h>
#include <time.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <x86intrin.h>

typedef float v4f32_t __attribute__ ((vector_size (4 * sizeof (float))));

int
main ()
{
    srand (1);
    register size_t i;
    const size_t n = 100000000;
    float *A = malloc (n * sizeof (float));
    float *B = malloc (n * sizeof (float));
    float *C = malloc (n * sizeof (float));
    float *D = malloc (n * sizeof (float));
    float *E = malloc (n * sizeof (float));

    v4f32_t a, b, c;
    __m256 va, vb;

    printf ("Random initialize A and B\n");
    for (size_t i = 0; i < n; i++)
    {
        A[i] = (double)rand () / RAND_MAX;
        B[i] = (double)rand () / RAND_MAX;
    }

    printf ("Scalar add ");
    clock_t t = clock ();
    for (i = 0; i < n; i++)
        C[i] = A[i] + B[i];
    t = clock () - t;
    printf ("%f\n", (double)t / CLOCKS_PER_SEC);

    printf ("GCC Vector Ext. add ");
    t = clock ();
    for (i = 0; i < n; i += 4)
    {
        memcpy (&a, &A[i], 4 * sizeof (float));
        memcpy (&b, &B[i], 4 * sizeof (float));
        c = a + b;
        memcpy (&D[i], &c, 4 * sizeof (float));
    }
    t = clock () - t;
    printf ("%f\n", (double)t / CLOCKS_PER_SEC);

    printf ("AVX2 add ");
    t = clock ();
    for (i = 0; i < n; i += 8)
    {
        va = _mm256_loadu_ps (&A[i]);
        vb = _mm256_loadu_ps (&B[i]);
        _mm256_storeu_ps (&E[i], _mm256_add_ps (va, vb));
    }
    t = clock () - t;
    printf ("%f\n", (double)t / CLOCKS_PER_SEC);

    float eps = 1e-4;
    for (i = 0; i < n; i++)
        if (fabs (C[i] - D[i]) > eps || fabs (C[i] - E[i]) > eps)
        {
            printf ("ERROR! %f %f %f\n", C[i], D[i], E[i]);
            return -1;
        }

    printf ("OK\n");

    free (A);
    free (B);
    free (C);
    free (D);
    free (E);

    return 0;
}